"""PROBE / SPAI evaluation on the fixed GenImage q95 dev inputs.

The input manifest is produced by ``scripts/audit_genimage_usage.py``.  Every
row carries source and q95 hashes plus usage flags.  The report is computed for

* ``dev_all``: all 6,299 dev inputs, descriptive;
* ``dev_no_known_use``: the 5,619 inputs with no known train/fit/eval/review/
  SFT-data use, the primary selection view.

Real inputs are shared across the eight generator groups.  No pooled AUROC is
reported; per-group CIs are not independent.

Usage:
  python scripts/g8_genimage_eval.py --dry-run
  python scripts/g8_genimage_eval.py --experts probe --device cuda
  python scripts/g8_genimage_eval.py --experts spai --device cuda --resume
  python scripts/g8_genimage_eval.py --report-only --experts probe,spai
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from scripts.g8_external_eval import (  # noqa: E402
    _cgroup,
    _expert_identity,
    _measured,
    _timing,
    bootstrap_auroc,
    build_expert,
    expert_orientation,
    predicts_fake,
)

DEFAULT_MANIFEST = os.path.join(
    PROJECT_ROOT, "dataset", "GenImage_q95", "eval_dev_manifest_20261102.json")
DEFAULT_OUT = os.path.join(
    PROJECT_ROOT, "sft_data", "g8_external_eval", "genimage_dev_20261102")
SET_NAMES = ("dev_all", "dev_no_known_use")
FAKE_GROUPS = ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15",
               "VQDM", "Wukong")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("split") != "dev":
        raise SystemExit(f"expected a dev manifest, got {payload.get('split')!r}")
    return payload


def _select(manifest: dict, limit: int | None):
    rows = manifest["inputs"]
    if limit:
        rows = rows[:limit]
    return rows


def run(manifest_path: str, out_dir: str, experts: list[str], device: str,
        limit: int | None, dry_run: bool, resume: bool) -> dict:
    manifest = _load_manifest(manifest_path)
    rows = _select(manifest, limit)

    if dry_run:
        counts = collections.Counter(
            (row["group"], row["label"], "no_known_use" if row["no_known_use"]
             else "known_use") for row in rows)
        print(json.dumps({"inputs": len(rows), "experts": experts,
                          "counts": {f"{k[0]}/{k[1]}/{k[2]}": v
                                     for k, v in sorted(counts.items())}},
                         ensure_ascii=False, indent=2))
        return {"dry_run": True}

    if device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit(
            "GPU 评测未授权：确认后加 FORENSIC_ALLOW_GPU=1 再跑；dry-run 不需要授权。")

    os.makedirs(out_dir, exist_ok=True)
    scores_path = os.path.join(out_dir, "scores.jsonl")
    done = set()
    if resume and os.path.exists(scores_path):
        for line in open(scores_path, encoding="utf-8"):
            row = json.loads(line)
            done.add((row["variant_id"], row["expert"]))

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
            load_errors[name] = f"{type(exc).__name__}: {exc}"
        load_s[name] = round(time.perf_counter() - load_started, 3)

    written = 0
    with open(scores_path, "a", encoding="utf-8") as handle:
        for entry in rows:
            if all((entry["variant_id"], name) in done for name in experts):
                continue
            image_path = os.path.join(PROJECT_ROOT, entry["path"])
            try:
                image = ImageUtils.load_image(image_path)
                if image is None or getattr(image, "size", 0) == 0:
                    raise ValueError("empty image")
            except Exception as exc:
                for name in experts:
                    if (entry["variant_id"], name) in done:
                        continue
                    field, threshold, direction = expert_orientation(name)
                    handle.write(json.dumps({
                        **{key: entry[key] for key in
                           ("variant_id", "group", "label", "source_id",
                            "source_path", "old_split", "new_split",
                            "usage_flags", "usage_roles", "no_known_use")},
                        "path": entry["path"],
                        "source_sha256": entry["source_sha256"],
                        "sha256": entry["sha256"],
                        "expert": name, "field": field, "value": None,
                        "direction": direction, "oriented": None,
                        "status": "failed",
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
                        **{key: entry[key] for key in
                           ("variant_id", "group", "label", "source_id",
                            "source_path", "old_split", "new_split",
                            "usage_flags", "usage_roles", "no_known_use")},
                        "path": entry["path"],
                        "source_sha256": entry["source_sha256"],
                        "sha256": entry["sha256"],
                        "expert": name, "field": field, "value": None,
                        "direction": direction, "oriented": None,
                        "status": "failed",
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
                    **{key: entry[key] for key in
                       ("variant_id", "group", "label", "source_id",
                        "source_path", "old_split", "new_split",
                        "usage_flags", "usage_roles", "no_known_use")},
                    "path": entry["path"],
                    "source_sha256": entry["source_sha256"],
                    "sha256": entry["sha256"],
                    "expert": name, "field": field, "value": value,
                    "direction": direction,
                    "oriented": None if value is None else value,
                    "status": status, "error": error,
                    "threshold": threshold,
                    "predicted_label": (
                        ("Fake" if predicts_fake(value, direction, threshold)
                         else "Real") if value is not None else None),
                    "elapsed_ms": round(elapsed_ms, 1),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            handle.flush()

    report = build_report(scores_path, manifest, manifest_path, experts)
    gpu_peak = gpu_reserved = None
    if device.startswith("cuda") and torch.cuda.is_available():
        gpu_peak = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        gpu_reserved = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    report["run"] = {
        "started_at": datetime.fromtimestamp(started).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - started, 1),
        "device": device,
        "experts": experts,
        "inputs_this_run": len(rows),
        "written_this_run": written,
        "scores_file": os.path.relpath(scores_path, PROJECT_ROOT),
        "scores_sha256": _sha256(scores_path),
        "load_s": load_s,
        "load_errors": load_errors,
        "gpu_peak_allocated_gb": gpu_peak,
        "gpu_peak_reserved_gb": gpu_reserved,
        "host_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cgroup_current_start": cgroup_start,
        "cgroup_current_end": _cgroup("current"),
        "cgroup_peak": _cgroup("peak"),
    }
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    write_markdown(report, os.path.join(out_dir, "report.md"))
    return report


def _dist(values):
    if not values:
        return None
    ordered = sorted(values)
    import numpy as np
    return {
        "n": len(values),
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values, ddof=0)), 6),
        "min": round(float(ordered[0]), 6),
        "p05": round(float(np.percentile(values, 5)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "p50": round(float(np.percentile(values, 50)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "p95": round(float(np.percentile(values, 95)), 6),
        "max": round(float(ordered[-1]), 6),
    }


def _pool_sha(rows):
    values = sorted(row["sha256"] for row in rows)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _metrics(rows, expert):
    field, threshold, direction = expert_orientation(expert)
    measured = [row for row in rows if _measured(row)]
    failures = len(rows) - len(measured)
    if not measured:
        return {
            "n": len(rows), "n_measured": 0,
            "n_fake": sum(1 for row in rows if row["label"] == "Fake"),
            "n_real": sum(1 for row in rows if row["label"] == "Real"),
            "n_sources": 0, "failures": failures,
            "auroc": None, "tpr": None, "tnr": None,
            "balanced_accuracy": None, "threshold_accuracy": None,
            "threshold": {"value": threshold, "direction": direction,
                          "chosen_on": "registered fixed threshold; not fitted here"},
            "score_distribution": None, "errors": [],
        }
    labels = [1 if row["label"] == "Fake" else 0 for row in measured]
    scores = [float(row["oriented"]) for row in measured]
    fake = [float(row["oriented"]) for row in measured if row["label"] == "Fake"]
    real = [float(row["oriented"]) for row in measured if row["label"] == "Real"]
    tpr = (sum(1 for row in measured
               if row["label"] == "Fake" and predicts_fake(
                   row["value"], row["direction"], row["threshold"]))
           / len(fake)) if fake else None
    tnr = (sum(1 for row in measured
               if row["label"] == "Real" and not predicts_fake(
                   row["value"], row["direction"], row["threshold"]))
           / len(real)) if real else None
    correct = sum(
        predicts_fake(row["value"], row["direction"], row["threshold"])
        == (row["label"] == "Fake") for row in measured)
    errors = []
    for row in measured:
        predicted = "Fake" if predicts_fake(
            row["value"], row["direction"], row["threshold"]) else "Real"
        if predicted != row["label"]:
            errors.append({
                "variant_id": row["variant_id"],
                "source_id": row["source_id"],
                "source_sha256": row["source_sha256"],
                "sha256": row["sha256"],
                "label": row["label"],
                "predicted": predicted,
                "score": round(float(row["value"]), 6),
                "no_known_use": row["no_known_use"],
            })
    return {
        "n": len(rows),
        "n_measured": len(measured),
        "n_fake": len(fake),
        "n_real": len(real),
        "n_sources": len({row["source_id"] for row in measured}),
        "failures": failures,
        "auroc": bootstrap_auroc(scores, labels),
        "threshold": {"value": threshold, "direction": direction,
                      "chosen_on": "registered fixed threshold; not fitted here"},
        "tpr": round(tpr, 6) if tpr is not None else None,
        "tnr": round(tnr, 6) if tnr is not None else None,
        "balanced_accuracy": (round((tpr + tnr) / 2, 6)
                              if tpr is not None and tnr is not None else None),
        "threshold_accuracy": round(correct / len(measured), 6),
        "score_distribution": {"fake": _dist(fake), "real": _dist(real)},
        "errors": errors,
    }


def build_report(scores_path: str, manifest: dict, manifest_path: str,
                 experts: list[str]) -> dict:
    rows = [json.loads(line) for line in open(scores_path, encoding="utf-8")]
    by_variant = collections.defaultdict(dict)
    for row in rows:
        by_variant[row["variant_id"]][row["expert"]] = row
    # Restrict to the manifest input set; stale rows from another manifest are
    # visible as "extra" rather than silently mixed.
    manifest_ids = {row["variant_id"] for row in manifest["inputs"]}
    extras = sorted(set(by_variant) - manifest_ids)
    missing = sorted(manifest_ids - set(by_variant))

    sets = {}
    for set_name in SET_NAMES:
        set_rows = [row for row in manifest["inputs"]
                    if set_name in row["eval_sets"]]
        real_rows = [row for row in set_rows if row["label"] == "Real"]
        per_group = {}
        for group in FAKE_GROUPS:
            fake_rows = [row for row in set_rows
                         if row["group"] == group and row["label"] == "Fake"]
            group_report = {
                "n_fake_inputs": len(fake_rows),
                "n_real_inputs": len(real_rows),
                "real_pool_shared": True,
                "real_pool_sha256": _pool_sha(real_rows),
                "experts": {},
            }
            for name in experts:
                measured_rows = []
                for entry in fake_rows + real_rows:
                    row = by_variant.get(entry["variant_id"], {}).get(name)
                    if row is not None:
                        measured_rows.append(row)
                group_report["experts"][name] = _metrics(measured_rows, name)
            per_group[group] = group_report
        real_only = {}
        for name in experts:
            real_measured = []
            for entry in real_rows:
                row = by_variant.get(entry["variant_id"], {}).get(name)
                if row is not None:
                    real_measured.append(row)
            real_only[name] = _metrics(real_measured, name)
        sets[set_name] = {
            "n_inputs": len(set_rows),
            "n_fake_inputs": sum(1 for row in set_rows if row["label"] == "Fake"),
            "n_real_inputs": len(real_rows),
            "n_no_known_use": sum(1 for row in set_rows if row["no_known_use"]),
            "groups": per_group,
            "real_only": real_only,
            "caveat": ("Real inputs are shared across all eight generator groups; "
                       "group metrics are not independent and no pooled AUROC "
                       "is reported."),
        }

    usage = {}
    for set_name in SET_NAMES:
        subset = [row for row in manifest["inputs"] if set_name in row["eval_sets"]]
        usage[set_name] = {
            "inputs": len(subset),
            "no_known_use": sum(1 for row in subset if row["no_known_use"]),
            "signatures": dict(collections.Counter(
                "|".join(row["usage_flags"]) if row["usage_flags"]
                else "split_only_unknown" for row in subset)),
            "by_group": dict(collections.Counter(row["group"] for row in subset)),
        }

    return {
        "schema": "genimage_external_eval/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "inputs_manifest": os.path.relpath(manifest_path, PROJECT_ROOT),
        "inputs_manifest_sha256": _sha256(manifest_path),
        "manifest_split": manifest.get("split"),
        "rules": {
            "no_threshold_fitting": True,
            "fixed_thresholds": True,
            "per_generator_only": True,
            "no_pooled_auroc": True,
            "real_images_shared_across_groups": True,
            "test_not_used": True,
        },
        "experts": {name: _expert_identity(name) for name in experts},
        "usage": usage,
        "sets": sets,
        "alignment": {
            "manifest_inputs": len(manifest["inputs"]),
            "score_variants": len(by_variant),
            "extra_score_variants": extras,
            "missing_score_variants": missing,
        },
    }


def _fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def write_markdown(report: dict, path: str):
    lines = [
        "# GenImage q95 全局专家评测（PROBE / SPAI）",
        "",
        f"生成时间：{report['created_at']}；输入：`{report['inputs_manifest']}`。",
        "",
        "> 真实图在八个生成器组间共享；组间不独立；不报告跨组合并 AUROC。"
        "阈值固定为登记值，不在本数据上拟合。",
        "",
    ]
    for set_name, data in report["sets"].items():
        lines += [f"## {set_name}", "",
                  f"输入 {data['n_inputs']}（Fake {data['n_fake_inputs']} / "
                  f"Real {data['n_real_inputs']}；无已知使用 "
                  f"{data['n_no_known_use']}）。", ""]
        for expert in report["experts"]:
            lines += [f"### {expert}", "",
                      "| 生成器 | n Fake | n Real | AUROC | TPR | TNR | 平衡准确率 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
            for group, entry in data["groups"].items():
                m = entry["experts"][expert]
                auroc = m["auroc"]["auroc"] if m["auroc"] else None
                lines.append(
                    f"| {group} | {m['n_fake']} | {m['n_real']} | "
                    f"{_fmt(auroc)} | {_fmt(m['tpr'])} | {_fmt(m['tnr'])} | "
                    f"{_fmt(m['balanced_accuracy'])} |")
            real = data["real_only"][expert]
            fake_errors = sum(
                sum(1 for row in entry["experts"][expert]["errors"]
                    if row["label"] == "Fake")
                for entry in data["groups"].values())
            real_errors = sum(
                1 for row in real["errors"] if row["label"] == "Real")
            lines += ["",
                      f"错误（固定阈值）：fake {fake_errors} / "
                      f"real {real_errors}；Real-only n={real['n_measured']}。"
                      "完整错误样本见 JSON。", ""]
    run = report.get("run", {})
    lines += ["## 运行与资源", "",
              f"- device: `{run.get('device')}`；elapsed: {run.get('elapsed_s')} s；"
              f"written: {run.get('written_this_run')}",
              f"- load_s: `{run.get('load_s')}`；GPU peak allocated/reserved: "
              f"{run.get('gpu_peak_allocated_gb')} / {run.get('gpu_peak_reserved_gb')} GB",
              f"- host maxrss: {run.get('host_maxrss_kb')} KB；cgroup "
              f"{run.get('cgroup_current_start')} → {run.get('cgroup_current_end')}",
              "",
              "完整分数、错误样本、分位数和失败列表见同目录 `report.json` 与 "
              "`scores.jsonl`。"]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def rebuild(scores_path: str, manifest_path: str, experts: list[str]) -> dict:
    manifest = _load_manifest(manifest_path)
    report = build_report(scores_path, manifest, manifest_path, experts)
    report["run"] = {
        "rebuilt_from": os.path.relpath(scores_path, PROJECT_ROOT),
        "scores_sha256": _sha256(scores_path),
        "rebuilt_at": datetime.now().isoformat(timespec="seconds"),
        "note": "No measurement was rerun; the report was rebuilt from scores.jsonl.",
    }
    out_dir = os.path.dirname(scores_path)
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    write_markdown(report, os.path.join(out_dir, "report.md"))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--experts", default="probe,spai")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    experts = [name.strip() for name in args.experts.split(",") if name.strip()]
    if args.report_only:
        rebuild(os.path.join(args.out, "scores.jsonl"), args.manifest, experts)
        return 0
    report = run(args.manifest, args.out, experts, args.device, args.limit,
                 args.dry_run, args.resume)
    print(json.dumps({"dry_run": report.get("dry_run", False),
                      "sets": {name: {"inputs": data["n_inputs"],
                                      "no_known_use": data["n_no_known_use"]}
                               for name, data in report.get("sets", {}).items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
