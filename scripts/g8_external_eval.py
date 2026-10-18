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
"""

from __future__ import annotations

import argparse
import json
import os
import random
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

def build_expert(name: str, device: str):
    """与 Agent 工具箱**同一实现**（同一类、同一预处理），否则不可比。"""
    if name == "probe":
        from experts.probe_dino import ProbeDinoExpert
        return ProbeDinoExpert(device=device)
    if name == "npr":
        from experts.npr import NPRExpert
        return NPRExpert(device=device)
    if name == "univfd":
        from experts.univfd import UnivFDExpert
        return UnivFDExpert(device=device)
    raise SystemExit(f"未知专家：{name}（probe / npr / univfd）")


def expert_orientation(name: str) -> Tuple[str, float, str]:
    """登记表给的 (分数字段, 阈值, 方向)——定向与判对率都以此为准，不拟合。"""
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

    import cv2

    started = time.time()
    written = 0
    with open(scores_path, "a", encoding="utf-8") as handle:
        for entry in entries:
            if all((entry["variant_id"], name) in done for name in experts):
                continue
            image_path = os.path.join(PROJECT_ROOT, entry["path"])
            image = cv2.imread(image_path)
            if image is None:
                raise SystemExit(f"读不到图像：{entry['path']}")
            for name, expert in objects.items():
                if (entry["variant_id"], name) in done:
                    continue
                field, _threshold, direction = expert_orientation(name)
                t0 = time.perf_counter()
                result = expert.analyze(image)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                value = float(getattr(result, field, getattr(result, "raw_metric", 0.0)))
                handle.write(json.dumps({
                    "variant_id": entry["variant_id"], "expert": name,
                    "group": entry["group"], "condition": entry["condition"],
                    "label": entry["label"], "source_path": entry["source_path"],
                    "field": field, "value": value, "direction": direction,
                    "oriented": oriented(value, direction),
                    "elapsed_ms": round(elapsed_ms, 1),
                }, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()

    rows = [json.loads(line) for line in open(scores_path, encoding="utf-8")]
    report = build_report(rows, experts, manifest, inputs_path)
    report["run"] = {
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "device": device, "experts": experts, "rows": len(rows),
        "written_this_run": written,
    }
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # 使用历史：哪一批、哪个阶段、多少输入
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
                 inputs_path: str) -> dict:
    """按 生成器 × 条件 汇总；每组 20 Fake + 20 Real；两条件按源配对。"""
    from utils.score_semantics import entry

    groups: Dict[str, dict] = {}
    by_key: Dict[Tuple[str, str, str], dict] = {}
    for row in rows:
        by_key.setdefault((row["group"], row["condition"], row["variant_id"]), {})[
            row["expert"]] = row

    report_groups: Dict[str, dict] = {}
    for (group, condition, variant_id), per_expert in sorted(by_key.items()):
        report_groups.setdefault(group, {}).setdefault(condition, {
            "n": 0, "n_fake": 0, "n_real": 0, "per_expert": {
                name: {"scores": [], "labels": [], "elapsed_ms": []} for name in experts
            }})
        bucket = report_groups[group][condition]
        bucket["n"] += 1
        label = 1 if per_expert[experts[0]]["label"] == "Fake" else 0
        bucket["n_fake" if label else "n_real"] += 1
        for name in experts:
            if name not in per_expert:
                continue
            row = per_expert[name]
            bucket["per_expert"][name]["scores"].append(row["oriented"])
            bucket["per_expert"][name]["labels"].append(label)
            bucket["per_expert"][name]["elapsed_ms"].append(row["elapsed_ms"])

    for group, conditions in report_groups.items():
        for condition, bucket in conditions.items():
            for name in experts:
                data = bucket["per_expert"][name]
                if not data["scores"]:
                    continue
                stat = bootstrap_auroc(data["scores"], data["labels"])
                data["auroc"] = stat
                # 登记表阈值判对率（用未定向的原始值）
                _field, threshold, direction = expert_orientation(name)
                correct = 0
                for variant_id, per_expert in by_key.items():
                    if variant_id[0] != group or variant_id[1] != condition:
                        continue
                    row = per_expert.get(name)
                    if not row:
                        continue
                    predicted = predicts_fake(row["value"], direction, threshold)
                    truth = row["label"] == "Fake"
                    correct += predicted == truth
                data["threshold_accuracy"] = round(correct / len(data["scores"]), 4)
                del data["scores"], data["labels"]
        # 条件配对：同一 variant_id 在两个条件下都有分数才配对
        if {"original", "jpeg_q95"} <= set(conditions):
            for name in experts:
                pairs = [(key, per) for key, per in by_key.items()
                         if key[0] == group and key[1] == "original" and name in per]
                a, b, labels = [], [], []
                for key, per in pairs:
                    other = by_key.get((key[0], "jpeg_q95", key[2]), {}).get(name)
                    if not other:
                        continue
                    a.append(per[name]["oriented"])
                    b.append(other["oriented"])
                    labels.append(1 if per[name]["label"] == "Fake" else 0)
                if a:
                    delta = paired_delta(a, b, labels)
                    conditions["original"]["per_expert"].setdefault(name, {})
                    conditions["original"]["per_expert"][name]["paired_delta_vs_jpeg_q95"] = delta
    return {
        "schema": "external_eval_report/1",
        "inputs_manifest": os.path.relpath(inputs_path, PROJECT_ROOT),
        "manifest_sha256": _sha256(inputs_path),
        "rules": {"no_threshold_fitting": True, "per_group_only": True,
                  "real_images_reused_across_generators": True,
                  "paired_conditions": True},
        "groups": report_groups,
    }


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
    args = parser.parse_args()

    experts = [name.strip() for name in args.experts.split(",") if name.strip()]
    out_dir = args.out or os.path.join(
        OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    report = run(args.inputs, out_dir, experts, args.device, args.limit,
                 args.dry_run, args.resume)
    if args.dry_run:
        return 0
    print(f"报告：{os.path.relpath(os.path.join(out_dir, 'report.json'), PROJECT_ROOT)}")
    for group, conditions in sorted(report["groups"].items()):
        for condition, bucket in sorted(conditions.items()):
            line = f"{group}/{condition} (n={bucket['n']}, Fake={bucket['n_fake']})"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
