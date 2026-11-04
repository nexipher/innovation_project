"""Expanded TruFor localization evaluation on the fixed AutoSplice dev subset.

The subset is fixed by ``build_autosplice_localization_subset.py`` (100 bases /
220 q95 inputs; 100 Real / 120 Fake).  This script runs the official TruFor
adapter, writes per-image raw npz, and reports:

* per-image IoU/F1/precision/recall against the shared base mask;
* real-image high-response area and false positives;
* image-level TruFor score (TruFor has one);
* source-level aggregation (variants are not independent);
* failures, timing and resources.

It supports CPU and CUDA, but the current 2 GiB cgroup is not sufficient for a
model load; the script is prepared for the restored environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from experts.trufor import HIGH_RESPONSE_THRESHOLD, TruForExpert  # noqa: E402
from scripts.g8_external_eval import bootstrap_auroc  # noqa: E402
from scripts.trufor_autosplice_smoke import (  # noqa: E402
    _environment,
    _load_bgr,
    _mask,
    _metrics,
)

SUBSET = os.path.join(
    PROJECT_ROOT, "sft_data", "g8_localization",
    "autosplice_dev_subset_20261102.json")
DEFAULT_OUT = os.path.join(
    PROJECT_ROOT, "sft_data", "g8_localization",
    "trufor_autosplice_dev_20261102")


def _dist(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std(ddof=0)), 6),
        "min": round(float(a.min()), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
        "max": round(float(a.max()), 6),
    }


def _source_aggregate(checks):
    """Aggregate variants by base; never treat same-source variants as independent."""
    by_source = {}
    for check in checks:
        by_source.setdefault(check["source_id"], []).append(check)
    sources = []
    for sid, rows in sorted(by_source.items()):
        real = [row for row in rows if row["label"] == "Real"]
        fake = [row for row in rows if row["label"] == "Fake"]
        if len(real) != 1:
            raise ValueError(f"{sid}: expected one authentic row, got {len(real)}")
        source = {
            "source_id": sid,
            "variant_count": len(fake),
            "authentic": {
                "variant_id": real[0]["variant_id"],
                "score": real[0]["score"],
                "map_high_fraction": real[0]["map_high_fraction"],
            },
            "forged": [],
        }
        if fake:
            source["forged"] = [{
                "variant_id": row["variant_id"],
                "score": row["score"],
                "iou": row["iou"],
                "f1": row["f1"],
                "precision": row["precision"],
                "recall": row["recall"],
                "map_high_fraction": row["map_high_fraction"],
                "map_inside_mean": row["map_inside_mean"],
                "map_outside_mean": row["map_outside_mean"],
            } for row in sorted(fake, key=lambda row: row["variant_id"])]
            source["forged_summary"] = {
                "score_mean": round(statistics.mean(row["score"] for row in fake), 6),
                "iou_mean": round(statistics.mean(row["iou"] for row in fake), 6),
                "iou_max": round(max(row["iou"] for row in fake), 6),
                "f1_mean": round(statistics.mean(row["f1"] for row in fake), 6),
                "precision_mean": round(statistics.mean(row["precision"] for row in fake), 6),
                "recall_mean": round(statistics.mean(row["recall"] for row in fake), 6),
            }
        sources.append(source)

    forged_sources = [source for source in sources if source["variant_count"]]
    real_sources = [source for source in sources if not source["variant_count"]]
    image_scores = [check["score"] for check in checks]
    image_labels = [1 if check["label"] == "Fake" else 0 for check in checks]
    source_scores = [source["forged_summary"]["score_mean"] for source in forged_sources] + \
                    [source["authentic"]["score"] for source in real_sources]
    source_labels = [1] * len(forged_sources) + [0] * len(real_sources)
    return {
        "sources": sources,
        "n_sources": len(sources),
        "n_forged_sources": len(forged_sources),
        "n_real_sources": len(real_sources),
        "forged_variant_iou": _dist([row["iou"] for row in checks
                                     if row["label"] == "Fake"]),
        "forged_variant_f1": _dist([row["f1"] for row in checks
                                    if row["label"] == "Fake"]),
        "forged_variant_recall": _dist([row["recall"] for row in checks
                                        if row["label"] == "Fake"]),
        "forged_source_iou_mean": _dist(
            [source["forged_summary"]["iou_mean"] for source in forged_sources]),
        "forged_source_f1_mean": _dist(
            [source["forged_summary"]["f1_mean"] for source in forged_sources]),
        "real_score": _dist([source["authentic"]["score"] for source in real_sources]),
        "real_high_fraction": _dist(
            [source["authentic"]["map_high_fraction"] for source in real_sources]),
        "real_false_positive_sources": [
            source for source in real_sources
            if source["authentic"]["map_high_fraction"] > 0.01
        ],
        "auroc_image_score": bootstrap_auroc(image_scores, image_labels),
        "auroc_source_score": bootstrap_auroc(source_scores, source_labels),
        "note": ("Same-source variants are repeated measurements; source-level "
                 "score uses the mean across variants. Image-level AUROC is "
                 "descriptive and not independent across variants."),
    }


def run(subset_path: str, out_dir: str, device: str, batch_size: int) -> dict:
    subset = json.load(open(subset_path, encoding="utf-8"))
    rows = subset["inputs"]
    if len(rows) != 220:
        raise SystemExit(f"expected 220 subset inputs, got {len(rows)}")
    if device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1（仅定位评测）")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault(
        "TRUFOR_RESULTS",
        os.path.join(PROJECT_ROOT, "sft_data", "g8_localization",
                     "trufor_raw_subset_20261102"))
    os.makedirs(out_dir, exist_ok=True)

    expert = TruForExpert(device=device)
    reason = expert.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor 不可用：{reason}")

    images = {}
    load_failures = []
    for row in rows:
        try:
            images[row["variant_id"]] = _load_bgr(
                os.path.join(PROJECT_ROOT, row["path"]))
        except Exception as exc:
            load_failures.append({
                "variant_id": row["variant_id"],
                "source_id": row["source_id"],
                "label": row["label"],
                "error": f"image_load_failed: {type(exc).__name__}: {exc}",
            })

    started = time.perf_counter()
    results = {}
    batch_failures = []
    keys = [row["variant_id"] for row in rows if row["variant_id"] in images]
    for start in range(0, len(keys), batch_size):
        chunk = {key: images[key] for key in keys[start:start + batch_size]}
        try:
            results.update(expert.analyze_many(chunk, progress=True))
        except Exception as exc:
            batch_failures.append({
                "variant_ids": list(chunk),
                "error": f"{type(exc).__name__}: {exc}",
            })
    wall = time.perf_counter() - started

    checks = []
    failures = list(load_failures)
    for row in rows:
        if row["variant_id"] not in images:
            continue
        result = results.get(row["variant_id"])
        if result is None:
            failures.append({
                "variant_id": row["variant_id"],
                "source_id": row["source_id"],
                "label": row["label"],
                "error": "missing_result",
            })
            continue
        try:
            key = expert.cache_key(images[row["variant_id"]])
            arrays = dict(np.load(expert.result_path(key), allow_pickle=True))
            mask = _mask(os.path.join(PROJECT_ROOT, row["mask_path"])) \
                if row["label"] == "Fake" else None
            metrics = _metrics(float(result.raw_metric), arrays["map"],
                               arrays.get("conf"), mask, row["label"])
            metrics.update({
                "variant_id": row["variant_id"],
                "source_id": row["source_id"],
                "label": row["label"],
                "variant": row.get("variant"),
                "path": row["path"],
                "mask_path": row.get("mask_path"),
                "mask_sha256": row.get("mask_sha256"),
                "source_sha256": row.get("source_sha256"),
                "q95_sha256": row.get("sha256"),
            })
            checks.append(metrics)
        except Exception as exc:
            failures.append({
                "variant_id": row["variant_id"],
                "source_id": row["source_id"],
                "label": row["label"],
                "error": f"metric_failed: {type(exc).__name__}: {exc}",
            })

    report = {
        "schema": "trufor_autosplice_eval/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ok": len(failures) == 0 and all(
            check["map_finite"] and check.get("conf_finite", True)
            for check in checks),
        "device": device,
        "subset": os.path.relpath(subset_path, PROJECT_ROOT),
        "subset_sha256": None,
        "weights": expert.provenance().as_dict(),
        "environment": _environment(),
        "sample": {
            "bases": subset["counts"]["bases"],
            "inputs": subset["counts"]["inputs"],
            "real": subset["counts"]["real"],
            "fake": subset["counts"]["fake"],
            "mask": "shared by base; no per-variant mask",
        },
        "timing": {
            "wall_s": round(wall, 2),
            "per_input_s": round(wall / max(1, len(checks)), 3),
            "adapter": expert.timing_summary(),
        },
        "failures": failures,
        "batch_failures": batch_failures,
        "maps": {
            "high_response_threshold": HIGH_RESPONSE_THRESHOLD,
            "score_meaning": "TruFor detection head sigmoid; higher = more manipulated",
            "map_meaning": "per-pixel P(forged); not a calibrated probability",
        },
        "checks": checks,
        "aggregate": _source_aggregate(checks) if checks else None,
    }
    import hashlib
    report["subset_sha256"] = hashlib.sha256(open(subset_path, "rb").read()).hexdigest()
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    write_markdown(report, os.path.join(out_dir, "report.md"))
    print(json.dumps({
        "ok": report["ok"],
        "device": device,
        "checked": len(checks),
        "failures": len(failures),
        "aggregate": {
            "forged_variant_iou_mean": (report["aggregate"]["forged_variant_iou"]["mean"]
                                        if report["aggregate"] else None),
            "forged_variant_f1_mean": (report["aggregate"]["forged_variant_f1"]["mean"]
                                       if report["aggregate"] else None),
            "real_high_fraction_mean": (report["aggregate"]["real_high_fraction"]["mean"]
                                        if report["aggregate"] else None),
            "auroc_source_score": (report["aggregate"]["auroc_source_score"]
                                   if report["aggregate"] else None),
        },
    }, ensure_ascii=False, indent=2))
    return report


def write_markdown(report, path):
    a = report.get("aggregate")
    lines = [
        "# TruFor AutoSplice q95 定位评测",
        "",
        f"装置：`{report['device']}`；输入：`{report['subset']}`；"
        f"检查 {len(report['checks'])}/220，失败 {len(report['failures'])}。",
        "",
        "> mask 为 base 级共享；同源多变体按 base 聚合，不当作独立源。",
        "",
    ]
    if a:
        lines += [
            "| 项目 | 值 |",
            "| --- | --- |",
            f"| forged variant IoU | mean {a['forged_variant_iou']['mean']:.3f}，"
            f"median {a['forged_variant_iou']['p50']:.3f} |",
            f"| forged variant F1 | mean {a['forged_variant_f1']['mean']:.3f}，"
            f"median {a['forged_variant_f1']['p50']:.3f} |",
            f"| forged source IoU | mean {a['forged_source_iou_mean']['mean']:.3f} |",
            f"| real high-response | mean {a['real_high_fraction']['mean']:.3f}，"
            f">1% sources {len(a['real_false_positive_sources'])}/{a['n_real_sources']} |",
            f"| image score AUROC (descriptive) | "
            f"{a['auroc_image_score']['auroc'] if a['auroc_image_score'] else None} |",
            f"| source score AUROC | "
            f"{a['auroc_source_score']['auroc'] if a['auroc_source_score'] else None} |",
            "",
        ]
    lines += [
        "完整逐图指标、源级聚合、失败和资源字段见同目录 `report.json`。",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default=SUBSET)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run(args.subset, args.out, args.device, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
