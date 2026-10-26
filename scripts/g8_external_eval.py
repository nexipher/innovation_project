"""
外部泛化评测：专家基线（阶段 1）。

对 `dataset/mnw_div2k/inputs_manifest.json` 的 200 个输入（100 源 × original /
jpeg_q95）逐一跑三个专家，写逐条分数与分组报告。**只做评测**：不训练、不调阈值、
不据此选专家（清单条款）。

设计要点（与 plan §4.9 对齐）：

* 三个专家**独立测量**（PROBE / NPR-A / UnivFD，共 600 次），与 Agent 后续调用的是
  同一套专家类与预处理——可比性来自同一实现，而不是"名字一样"。
* 分析按 **生成器 × 条件**：每组 = 20 Fake + 同一组 20 DIV2K Real（n=40）。单标签组
  算不了 AUROC，所以必须配对 Real；20 张 Real 在四个生成器组间复用，组间不独立。
* 两种条件作用在同一批源上 → **按源配对**：报告 ΔAUROC（jpeg_q95 − original）并
  用按源重采样的 bootstrap 给 CI；不把两种条件当独立样本。
* 阈值用**登记表既有值**判对率（`raw_score.direction` 定向后 ≥ threshold 判 Fake），
  不拟合、不调参。

用法：
  python scripts/g8_external_eval.py --device cuda            # 真跑（需 GPU）
  python scripts/g8_external_eval.py --dry-run                # 只列输入与分组
  python scripts/g8_external_eval.py --experts probe --limit 20
  python scripts/g8_external_eval.py --report-only <scores.jsonl>  # 只重建报告，不重测
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402

MANIFEST = os.path.join(PROJECT_ROOT, "dataset", "mnw_div2k", "inputs_manifest.json")
PLAN = os.path.join(PROJECT_ROOT, "dataset", "mnw_div2k", "eval_plan.json")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "sft_data", "g8_external_eval")


# ---------------------------------------------------------------------------
# 统计（纯函数，CPU 可测）
# ---------------------------------------------------------------------------

def auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """
    Mann-Whitney 形式的 AUROC（`scores` 越大越像 Fake，`labels` 1=Fake）。

    并列分数按平均秩处理；只有单一类别时返回 None（组内单标签就是这种情况，
    必须在调用方阻止——这里返回 None 而不是猜一个数）。
    """
    pairs = [(float(score), int(label)) for score, label in zip(scores, labels)]
    positives = [score for score, label in pairs if label == 1]
    negatives = [score for score, label in pairs if label == 0]
    if not positives or not negatives:
        return None

    def rank(values: List[float]) -> List[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = average
            i = j + 1
        return ranks

    ranked = rank([score for score, _ in pairs])
    positives_rank = sum(ranked[i] for i, pair in enumerate(pairs) if pair[1] == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (positives_rank - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def bootstrap_auroc(scores: Sequence[float], labels: Sequence[int],
                    samples: int = 1000, seed: int = 20261018) -> Optional[dict]:
    """按**条目**重采样的 AUROC CI（配对设计：同一条目在两种条件下成对出现）。"""
    point = auroc(scores, labels)
    if point is None:
        return None
    rng = random.Random(seed)
    n = len(scores)
    values = []
    for _ in range(samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        value = auroc([scores[i] for i in indexes], [labels[i] for i in indexes])
        if value is not None:
            values.append(value)
    values.sort()
    if not values:
        return None
    low = values[int(0.025 * (len(values) - 1))]
    high = values[int(0.975 * (len(values) - 1))]
    return {"auroc": round(point, 4), "ci95": [round(low, 4), round(high, 4)],
            "bootstrap": samples}


def paired_delta(scores_a: Sequence[float], scores_b: Sequence[float],
                 labels: Sequence[int], samples: int = 1000,
                 seed: int = 20261018) -> Optional[dict]:
    """
    同一批条的两种条件：ΔAUROC = AUROC(b) − AUROC(a)，CI 按条目成对重采样。

    配对是关键：两种条件作用在**同一张源图**上，把两组当独立样本会高估方差、
    并让"条件不同"与"图不同"混在一起。
    """
    base = auroc(scores_a, labels)
    other = auroc(scores_b, labels)
    if base is None or other is None:
        return None
    rng = random.Random(seed)
    n = len(labels)
    deltas = []
    for _ in range(samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        a = auroc([scores_a[i] for i in indexes], [labels[i] for i in indexes])
        b = auroc([scores_b[i] for i in indexes], [labels[i] for i in indexes])
        if a is not None and b is not None:
            deltas.append(b - a)
    deltas.sort()
    if not deltas:
        return None
    low = deltas[int(0.025 * (len(deltas) - 1))]
    high = deltas[int(0.975 * (len(deltas) - 1))]
    return {"delta": round(other - base, 4), "ci95": [round(low, 4), round(high, 4)],
            "bootstrap": samples}


# ---------------------------------------------------------------------------
# 专家
# ---------------------------------------------------------------------------

CANDIDATE_NAMES = ("spai", "aide", "aide_sd14", "aide_genimage", "drct")


def _candidate_semantics(name: str) -> dict:
    """Evaluation-only direction/threshold for candidates (not the Agent registry)."""
    path = os.path.join(PROJECT_ROOT, "experts", "candidate_semantics.json")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        return payload["experts"][name]
    except KeyError:
        raise SystemExit(f"candidate_semantics.json 里没有 {name!r}")


def build_expert(name: str, device: str):
    """与 Agent 工具箱**同一实现**；候选专家仅用于离线评测，不接入 Agent。"""
    if name == "spai":
        from experts.spai import SPAIExpert
        return SPAIExpert(device=device)
    if name == "drct":
        from experts.drct import DRCTExpert
        return DRCTExpert(device=device)
    if name == "aide":
        from experts.aide import AIDEExpert
        return AIDEExpert(device=device)
    if name == "aide_sd14":
        from experts.aide import AIDEExpert
        return AIDEExpert(checkpoint="sd14_train", device=device)
    if name == "aide_genimage":
        from experts.aide import AIDEExpert
        return AIDEExpert(checkpoint="GenImage_train", device=device)
    if name == "probe":
        from experts.probe_dino import ProbeDinoExpert
        return ProbeDinoExpert(device=device)
    if name == "npr":
        from experts.npr import NPRExpert
        return NPRExpert(device=device)
    if name == "univfd":
        from experts.univfd import UnivFDExpert
        return UnivFDExpert(device=device)
    raise SystemExit(
        f"未知专家：{name}（probe / npr / univfd / spai / aide / "
        f"aide_sd14 / aide_genimage / drct）")


def expert_orientation(name: str) -> Tuple[str, float, str]:
    """登记表/候选表给的 (分数字段, 阈值, 方向)——不拟合、不现场翻转。"""
    if name in CANDIDATE_NAMES:
        record = _candidate_semantics(name)
    else:
        from utils.score_semantics import entry

        record = entry({"probe": "probe_dino_expert", "npr": "npr_expert",
                        "univfd": "univfd_expert"}[name])
    semantics = record["raw_score"]
    field = semantics["field"]
    direction = semantics["direction"]
    threshold = float(record["threshold"]["value"])
    if direction == "higher_is_fake":
        return field, threshold, direction
    if direction == "higher_is_real":
        # 统一成"越大越像 Fake"：Real 向的分数取相反数
        return field, threshold, direction
    raise SystemExit(f"{name} 没有可用的方向，不能做 AUROC")


def oriented(value: float, direction: str) -> float:
    return value if direction == "higher_is_fake" else -value


def predicts_fake(value: float, direction: str, threshold: float) -> bool:
    """按登记表阈值判 Fake（higher_is_real 时阈值方向相反）。"""
    return value >= threshold if direction == "higher_is_fake" else value < threshold


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _measured(row: dict) -> bool:
    """A row counts only if it was measured and the value is finite."""
    if row.get("status", "measured") != "measured":
        return False
    value = row.get("value")
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _cgroup(name: str):
    try:
        with open(f"/sys/fs/cgroup/memory.{name}", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except OSError:
        return None


def _timing(rows: List[dict], experts: List[str]) -> Dict[str, dict]:
    """First measured call vs steady-state; elapsed_ms is analyze-only."""
    out = {}
    for name in experts:
        values = [float(row["elapsed_ms"]) for row in rows
                  if row.get("expert") == name and _measured(row)
                  and row.get("elapsed_ms") is not None]
        if values:
            steady = values[1:] or values
            out[name] = {"first_ms": values[0],
                         "steady_median_ms": round(statistics_median(steady), 1),
                         "steady_min_ms": min(steady),
                         "steady_max_ms": max(steady),
                         "n": len(values),
                         "scope": "expert.analyze() including preprocessing, "
                                  "forward and .item() sync; image loading excluded"}
    return out


def statistics_median(values: List[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def run(inputs_path: str, out_dir: str, experts: List[str], device: str,
        limit: Optional[int], dry_run: bool, resume: bool) -> dict:
    manifest = json.load(open(inputs_path, encoding="utf-8"))
    entries = manifest["inputs"]
    if limit:
        entries = entries[:limit]

    plan = {name: {"n": 0, "groups": {}} for name in experts}
    for name in experts:
        _field, threshold, direction = expert_orientation(name)
        for entry in entries:
            label = 1 if entry["label"] == "Fake" else 0
            plan[name]["n"] += 1
            key = (entry["group"], entry["condition"])
            plan[name]["groups"].setdefault(key, []).append(label)
    if dry_run:
        for name in experts:
            counts = {f"{g}/{c}": len(v) for (g, c), v in plan[name]["groups"].items()}
            print(f"{name}: {plan[name]['n']} 次测量｜分组 {counts}")
        return {"dry_run": True}

    if device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit(
            "GPU 评测未授权：确认后加 FORENSIC_ALLOW_GPU=1 再跑；"
            "dry-run 不需要授权。")

    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")
    done = set()
    if resume and os.path.exists(scores_path):
        for line in open(scores_path, encoding="utf-8"):
            raw = json.loads(line)
            done.add((raw["variant_id"], raw["expert"]))

    objects = {name: build_expert(name, device) for name in experts}
    for name, expert in objects.items():
        reason = expert.unavailable_reason()
        if reason:
            raise SystemExit(f"{name} 不可用：{reason}")

    from utils.image_utils import ImageUtils

    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cgroup_start = _cgroup("current")
    started = time.time()
    load_s = {}
    load_errors = {}
    for name, expert in objects.items():
        load_started = time.perf_counter()
        try:
            expert.preload()
        except Exception as exc:
            # A failed load is a measurement failure, not a zero/neutral score.
            load_errors[name] = f"{type(exc).__name__}: {exc}"
        load_s[name] = round(time.perf_counter() - load_started, 3)
    written = 0
    with open(scores_path, "a", encoding="utf-8") as handle:
        for entry in entries:
            if all((entry["variant_id"], name) in done for name in experts):
                continue
            image_path = os.path.join(PROJECT_ROOT, entry["path"])
            try:
                image = ImageUtils.load_image(image_path)
                if image is None or getattr(image, "size", 0) == 0:
                    raise ValueError("empty image")
            except Exception as exc:
                # The image itself failed: every expert gets an explicit
                # failure row, never a zero or a neutral score.
                for name in experts:
                    if (entry["variant_id"], name) in done:
                        continue
                    field, threshold, direction = expert_orientation(name)
                    handle.write(json.dumps({
                        "variant_id": entry["variant_id"], "expert": name,
                        "group": entry["group"], "condition": entry["condition"],
                        "label": entry["label"], "source_id": entry.get("source_id", ""),
                        "source_path": entry["source_path"],
                        "field": field, "value": None, "direction": direction,
                        "oriented": None, "status": "failed",
                        "error": f"image_load_failed: {type(exc).__name__}: {exc}",
                        "elapsed_ms": None,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                handle.flush()
                continue
            for name, expert in objects.items():
                if (entry["variant_id"], name) in done:
                    continue
                field, threshold, direction = expert_orientation(name)
                if name in load_errors:
                    handle.write(json.dumps({
                        "variant_id": entry["variant_id"], "expert": name,
                        "group": entry["group"], "condition": entry["condition"],
                        "label": entry["label"], "source_id": entry.get("source_id", ""),
                        "source_path": entry["source_path"],
                        "field": field, "value": None, "direction": direction,
                        "oriented": None, "status": "failed",
                        "error": f"preload_failed: {load_errors[name]}",
                        "threshold": threshold, "predicted_label": None,
                        "elapsed_ms": None,
                    }, ensure_ascii=False) + "\n")
                    written += 1
                    continue
                try:
                    t0 = time.perf_counter()
                    result = expert.analyze(image)
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    value = float(getattr(result, field,
                                          getattr(result, "raw_metric", 0.0)))
                    if not math.isfinite(value):
                        raise ValueError(f"non-finite score: {value!r}")
                    status, error = "measured", ""
                except Exception as exc:
                    elapsed_ms = (time.perf_counter() - t0) * 1000
                    value, status, error = None, "failed", (
                        f"{type(exc).__name__}: {exc}")
                row = {
                    "variant_id": entry["variant_id"], "expert": name,
                    "group": entry["group"], "condition": entry["condition"],
                    "label": entry["label"], "source_id": entry.get("source_id", ""),
                    "source_path": entry["source_path"],
                    "field": field, "value": value, "direction": direction,
                    "oriented": (oriented(value, direction)
                                 if value is not None else None),
                    "status": status, "error": error,
                    "threshold": threshold,
                    "predicted_label": (("Fake" if predicts_fake(value, direction, threshold)
                                        else "Real") if value is not None else None),
                    "elapsed_ms": round(elapsed_ms, 1),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()

    rows = [json.loads(line) for line in open(scores_path, encoding="utf-8")]
    report = build_report(rows, experts, manifest, inputs_path)
    gpu_peak = gpu_reserved = None
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_peak = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        gpu_reserved = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    report["run"] = {
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "device": device, "experts": experts, "rows": len(rows),
        "written_this_run": written,
        "rows_measured": sum(1 for row in rows if _measured(row)),
        "rows_failed": sum(1 for row in rows if row.get("status") == "failed"),
        "load_s": load_s,
        "load_errors": load_errors,
        "timing": _timing(rows, experts),
        "gpu_peak_allocated_gb": gpu_peak,
        "gpu_peak_reserved_gb": gpu_reserved,
        "host_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cgroup_current_start": cgroup_start,
        "cgroup_current_end": _cgroup("current"),
        "cgroup_peak": _cgroup("peak"),
        "cgroup_note": ("memory.current is anon+file+slab; file-backed mmap pages "
                        "are reclaimable. memory.peak is not exposed in this cgroup."),
    }
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    plan_doc = json.load(open(PLAN, encoding="utf-8"))
    plan_doc.setdefault("usage_history", []).append({
        "at": datetime.now().isoformat(timespec="seconds"),
        "action": "expert_baseline", "inputs": len(entries),
        "experts": experts, "report": os.path.relpath(
            os.path.join(out_dir, "report.json"), PROJECT_ROOT),
    })
    json.dump(plan_doc, open(PLAN, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return report


def build_report(rows: List[dict], experts: List[str], manifest: dict,
                 inputs_path: str, pairs: Optional[List[Tuple[str, str]]] = None) -> dict:
    """
    两份视图：

    * `groups`  —— 按 **来源组 × 条件** 的原始分数（单标签，只给分布与阈值判假率，
      **不算 AUROC**：单标签组算不出来）；
    * `analysis` —— 按 **生成器 × 条件** 的配对分析：每组 = 该生成器 20 Fake +
      DIV2K 20 Real（n=40），组内可算 AUROC；20 张 Real 在四个生成器组间复用，
      组间不独立，只作组内解释。两种条件同源 → 再给配对 ΔAUROC。
    """
    by_key: Dict[Tuple[str, str, str], dict] = {}
    for row in rows:
        by_key.setdefault((row["group"], row["condition"], row["variant_id"]), {})[
            row["expert"]] = row

    labels = {key: (1 if per[experts[0]]["label"] == "Fake" else 0)
              for key, per in by_key.items() if experts[0] in per}

    # ---- 来源组视图（单标签） ----
    groups: Dict[str, dict] = {}
    for (group, condition, _variant), per in sorted(by_key.items()):
        bucket = groups.setdefault(group, {}).setdefault(condition, {
            "n": 0, "n_fake": 0, "n_real": 0, "n_measured": 0, "n_failed": 0,
            "per_expert": {
                name: {"values": [], "fake_rate_at_threshold": 0,
                       "elapsed_ms": [], "failures": 0}
                for name in experts}})
        bucket["n"] += 1
        label = labels[(group, condition, _variant)]
        bucket["n_fake" if label else "n_real"] += 1
        for name in experts:
            row = per.get(name)
            if not row:
                continue
            data = bucket["per_expert"][name]
            if not _measured(row):
                data["failures"] += 1
                continue
            data["values"].append(float(row["value"]))
            data["elapsed_ms"].append(row["elapsed_ms"])
            _field, threshold, direction = expert_orientation(name)
            data["fake_rate_at_threshold"] += int(
                predicts_fake(row["value"], direction, threshold))
        measured_any = any(_measured(per[name]) for name in experts if name in per)
        bucket["n_measured"] += int(measured_any)
        bucket["n_failed"] += int(not measured_any)
    for group, conditions in groups.items():
        for condition, bucket in conditions.items():
            for name in experts:
                data = bucket["per_expert"][name]
                if data["values"]:
                    data["fake_rate_at_threshold"] = round(
                        data["fake_rate_at_threshold"] / len(data["values"]), 4)
                    data["mean_score"] = round(
                        sum(data["values"]) / len(data["values"]), 4)
                else:
                    data["fake_rate_at_threshold"] = None
                    data["mean_score"] = None
                data["n_measured"] = len(data["values"])
                del data["values"]

    # ---- 配对分析视图（生成器 × 条件） ----
    if pairs is None:
        real_groups = sorted({key[0] for key, label in labels.items() if label == 0})
        real_group = real_groups[0] if real_groups else None
        fake_groups = sorted({key[0] for key, label in labels.items() if label == 1})
        pairs = [(group, real_group) for group in fake_groups if group != real_group]

    analysis: Dict[str, dict] = {}
    for fake_group, real_group in pairs:
        per_condition: Dict[str, dict] = {}
        item_ids: Dict[str, List[str]] = {}
        for condition in ("original", "jpeg_q95"):
            items = []
            for (group, cond, variant), per in sorted(by_key.items()):
                if cond != condition:
                    continue
                label = labels[(group, cond, variant)]
                if (group == fake_group and label == 1) or \
                        (group == real_group and label == 0):
                    items.append((_source_key(per), per, label))
            if not items:
                continue
            # 配对键是**源**（source_id / source_path），不是 variant_id：
            # variant_id 带 __original / __jpeg_q95 后缀，跨条件永远交集为空。
            item_ids[condition] = [source_key for source_key, _per, _label in items]
            bucket = {"n": len(items),
                      "n_fake": sum(1 for _v, _p, label in items if label == 1),
                      "n_real": sum(1 for _v, _p, label in items if label == 0),
                      "n_sources": len({_source_key(per) for _v, per, _l in items}),
                      "n_measured": sum(1 for _v, per, _l in items
                                        if any(_measured(per[n]) for n in experts
                                               if n in per)),
                      "n_failed": 0,
                      "per_expert": {}}
            for name in experts:
                measured = [(per, label) for _v, per, label in items
                            if name in per and _measured(per[name])]
                failures = sum(1 for _v, per, _l in items
                               if name in per and not _measured(per[name]))
                bucket["n_failed"] += int(failures == len(items))
                if not measured:
                    bucket["per_expert"][name] = {
                        "auroc": None, "threshold_accuracy": None,
                        "tpr": None, "tnr": None, "balanced_accuracy": None,
                        "n_measured": 0, "n_fake_measured": 0,
                        "n_real_measured": 0, "failures": failures,
                    }
                    continue
                scores = [float(per[name]["oriented"]) for per, _l in measured]
                truths = [label for _per, label in measured]
                _field, threshold, direction = expert_orientation(name)
                correct = sum(
                    predicts_fake(per[name]["value"], direction, threshold)
                    == bool(label) for per, label in measured)
                n_fake = sum(1 for _per, label in measured if label == 1)
                n_real = sum(1 for _per, label in measured if label == 0)
                tpr = (sum(1 for per, label in measured
                           if label == 1 and predicts_fake(
                               per[name]["value"], direction, threshold))
                       / n_fake) if n_fake else None
                tnr = (sum(1 for per, label in measured
                           if label == 0 and not predicts_fake(
                               per[name]["value"], direction, threshold))
                       / n_real) if n_real else None
                balanced = ((tpr + tnr) / 2.0
                            if tpr is not None and tnr is not None else None)
                elapsed = [float(per[name]["elapsed_ms"]) for per, _l in measured
                           if per[name].get("elapsed_ms") is not None]
                bucket["per_expert"][name] = {
                    "auroc": bootstrap_auroc(scores, truths),
                    "threshold_accuracy": round(correct / len(scores), 4),
                    "tpr": round(tpr, 4) if tpr is not None else None,
                    "tnr": round(tnr, 4) if tnr is not None else None,
                    "balanced_accuracy": (round(balanced, 4)
                                          if balanced is not None else None),
                    "n_measured": len(scores),
                    "n_fake_measured": n_fake,
                    "n_real_measured": n_real,
                    "failures": failures,
                    "elapsed_ms": ({
                        "n": len(elapsed),
                        "mean": round(sum(elapsed) / len(elapsed), 1),
                        "median": round(statistics_median(elapsed), 1),
                        "min": round(min(elapsed), 1),
                        "max": round(max(elapsed), 1),
                    } if elapsed else None),
                }
            per_condition[condition] = bucket
        # 两种条件按同一 variant 配对（variant_id 唯一，且带来源组前缀）
        if {"original", "jpeg_q95"} <= set(per_condition):
            rows_by_condition: Dict[str, Dict[str, dict]] = {}
            for (group, cond, _variant), per in by_key.items():
                if cond in ("original", "jpeg_q95"):
                    key = _source_key(per)
                    if key in rows_by_condition.setdefault(cond, {}):
                        raise SystemExit(f"同一条件下源重复：{key}")
                    rows_by_condition[cond][key] = per
            joint = sorted(set(rows_by_condition["original"])
                           & set(rows_by_condition["jpeg_q95"])
                           & set(item_ids["original"]) & set(item_ids["jpeg_q95"]))
            for name in experts:
                a, b, truths = [], [], []
                for source_key in joint:
                    orig = rows_by_condition["original"][source_key]
                    jpeg = rows_by_condition["jpeg_q95"][source_key]
                    if name not in orig or name not in jpeg:
                        continue
                    if not (_measured(orig[name]) and _measured(jpeg[name])):
                        continue
                    a.append(float(orig[name]["oriented"]))
                    b.append(float(jpeg[name]["oriented"]))
                    truths.append(1 if orig[name]["label"] == "Fake" else 0)
                if a:
                    per_condition["original"]["per_expert"].setdefault(name, {})[
                        "paired_delta_vs_jpeg_q95"] = paired_delta(a, b, truths)
                    per_condition["original"]["per_expert"][name][
                        "paired_delta_n_pairs"] = len(a)
        analysis[f"{fake_group} vs {real_group}"] = per_condition

    return {
        "schema": "external_eval_report/1",
        "inputs_manifest": os.path.relpath(inputs_path, PROJECT_ROOT),
        "manifest_sha256": _sha256(inputs_path),
        "rules": {"no_threshold_fitting": True, "per_group_only": True,
                  "real_images_reused_across_generators": True,
                  "paired_conditions": True,
                  "single_label_groups_have_no_auroc": True},
        "composition": _composition(manifest),
        "experts": {name: _expert_identity(name) for name in experts},
        "groups": groups,
        "analysis": analysis,
    }


def _expert_identity(name: str) -> dict:
    """报告要能回答"这个分数由哪份权重、哪套预处理产生"——从登记表/候选表抄。"""
    if name in CANDIDATE_NAMES:
        record = _candidate_semantics(name)
        code_key = "aide" if name.startswith("aide") else name
        code = json.load(open(os.path.join(
            PROJECT_ROOT, "experts", "candidate_code.json"),
            encoding="utf-8"))["pinned"][code_key]
        return {
            "weights": record["weights"],
            "raw_score": record["raw_score"],
            "threshold": record["threshold"],
            "preprocess": record["preprocess"],
            "licence": record.get("licence"),
            "code": {"repo": code["repo"], "commit": code["commit"]},
            "output": {
                "transform": ("sigmoid" if name == "spai"
                              else "softmax(logits, dim=1)[:, 1]"),
                "threshold": record["threshold"]["value"],
                "meaning": record["raw_score"]["meaning"],
            },
        }
    from utils.score_semantics import entry

    record = entry({"probe": "probe_dino_expert", "npr": "npr_expert",
                    "univfd": "univfd_expert"}[name])
    weights = record.get("weights") or {}
    if "sha256" in weights:                                   # probe
        weight_identity = {"file": weights.get("file"),
                           "sha256": weights.get("sha256"),
                           "bytes": weights.get("bytes")}
    elif "checkpoints" in weights:                            # npr
        selected = (weights.get("selected") or {}).get("checkpoint_id")
        chosen = next((c for c in weights["checkpoints"] if c["id"] == selected), {})
        weight_identity = {"checkpoint_id": selected,
                           "file": chosen.get("file"),
                           "sha256": chosen.get("sha256"),
                           "bytes": chosen.get("bytes")}
    else:                                                     # univfd
        weight_identity = {"head": weights.get("head"),
                           "backbone": weights.get("backbone")}
    return {
        "weights": weight_identity,
        "raw_score": record["raw_score"],
        "threshold": record["threshold"],
        "preprocess": record["preprocess"],
        "licence": weights.get("licence"),
    }


def _composition(manifest: dict) -> dict:
    """
    每种条件下"标签 × 容器"的计数。

    `original` 条件里 MNW Fake 是 JPEG、DIV2K Real 是 PNG——标签与容器混杂；
    `jpeg_q95` 把两者都重编码为 JPEG q95。把这个计数写进报告，读者不必
    从路径里猜。
    """
    composition: Dict[str, dict] = {}
    for item in manifest.get("inputs", []):
        condition = item.get("condition", "unknown")
        bucket = composition.setdefault(condition, {"n": 0, "by_label_container": {}})
        bucket["n"] += 1
        container = item.get("container") or \
            os.path.splitext(item.get("path", ""))[1].lstrip(".").lower()
        key = f"{item.get('label')}/{container}"
        bucket["by_label_container"][key] = bucket["by_label_container"].get(key, 0) + 1
    return composition


def _source_key(per: dict) -> str:
    """
    同一源在两种条件下的配对键。

    优先 `source_id`（清单里两种条件共用），回退到 `source_path`（清单里也共用）。
    `variant_id` **不能**用：它带 `__original` / `__jpeg_q95` 后缀，跨条件不相等。
    """
    for row in per.values():
        key = row.get("source_id") or row.get("source_path")
        if key:
            return key
    raise SystemExit("分数行既无 source_id 也无 source_path，无法按源配对")


def _sha256(path: str) -> Optional[str]:
    """清单哈希；文件不在（例如单测传了占位路径）就写 null，不编一个值。"""
    import hashlib

    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rebuild_report(inputs_path: str, scores_path: str, experts: List[str]) -> dict:
    """只重建报告：读已有的逐条分数，不重跑任何测量。"""
    rows = [json.loads(line) for line in open(scores_path, encoding="utf-8")]
    manifest = json.load(open(inputs_path, encoding="utf-8"))
    report = build_report(rows, experts, manifest, inputs_path)
    report["run"] = {
        "rebuilt_from": os.path.relpath(scores_path, PROJECT_ROOT),
        "scores_sha256": _sha256(scores_path),
        "scores_mtime": datetime.fromtimestamp(
            os.path.getmtime(scores_path)).isoformat(timespec="seconds"),
        "rebuilt_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "measurement_provenance": "rebuilt_from_scores",
        "note": ("未重新测量；报告由已有 scores.jsonl 重建。"
                 "原始 run 的 device/elapsed 字段未保留，以运行日志与 plan 记录为准。"),
    }
    out_dir = os.path.dirname(scores_path)
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", default=MANIFEST)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--experts", default="probe,npr,univfd")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", metavar="SCORES_JSONL", default=None,
                        help="不测量，只从已有 scores.jsonl 重建报告")
    args = parser.parse_args()

    experts = [name.strip() for name in args.experts.split(",") if name.strip()]
    if args.report_only:
        report = rebuild_report(args.inputs, args.report_only, experts)
    else:
        out_dir = args.out or os.path.join(
            OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
        report = run(args.inputs, out_dir, experts, args.device, args.limit,
                     args.dry_run, args.resume)
        if args.dry_run:
            return 0
        print(f"报告：{os.path.relpath(os.path.join(out_dir, 'report.json'), PROJECT_ROOT)}")

    # 配对分析（生成器 Fake + DIV2K Real）
    for pair, conditions in sorted(report["analysis"].items()):
        for condition, bucket in sorted(conditions.items()):
            line = (f"{pair}/{condition} (n={bucket['n']}, "
                    f"Fake={bucket['n_fake']}, Real={bucket['n_real']})")
            for name in experts:
                data = bucket["per_expert"].get(name) or {}
                auroc_value = data.get("auroc") or {}
                line += (f"｜{name}: AUROC {auroc_value.get('auroc')} "
                         f"CI{auroc_value.get('ci95')} "
                         f"acc {data.get('threshold_accuracy')}")
                delta = data.get("paired_delta_vs_jpeg_q95")
                if delta:
                    line += f" Δ {delta['delta']} CI{delta['ci95']}"
            print(line)
    # 来源组分布（单标签，仅均值与判假率）
    for group, conditions in sorted(report["groups"].items()):
        for condition, bucket in sorted(conditions.items()):
            line = f"{group}/{condition} (n={bucket['n']}, Fake={bucket['n_fake']})"
            for name in experts:
                data = bucket["per_expert"].get(name) or {}
                line += (f"｜{name}: mean {data.get('mean_score')} "
                         f"fake-rate {data.get('fake_rate_at_threshold')}")
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
