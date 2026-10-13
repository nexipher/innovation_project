#!/usr/bin/env python3
"""
G2-c: distill calibration measurements into calibration/reliability_table.json.

Inputs:
  calibration/set/raw_values.json      per-sample expert measurements (G2-b run)
  calibration/set/manifest.json        cell/label table
  calibration/g2_expert_report.json    AUROC, confound drop, polarity alignment

Output:
  calibration/reliability_table.json   runtime lookup: raw metric -> P(Fake),
                                       per-expert separation, applicability label

The table is a *conditional lookup*, not a trained model: matched cells
(png + jpeg_q95/85/70) are pooled, the raw metric is split into equal-count
quantile bins, and each bin records the empirical P(Fake). Runtime tokens then
carry a calibrated likelihood and an applicability note instead of a bare
scalar — and the pipeline can refuse to read an inverted expert at face value.

Usage:
  python scripts/build_reliability_table.py
  python scripts/build_reliability_table.py --bins 6
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

SET_DIR = os.path.join(PROJECT_ROOT, "calibration", "set")
RAW_VALUES_PATH = os.path.join(SET_DIR, "raw_values.json")
MANIFEST_PATH = os.path.join(SET_DIR, "manifest.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_expert_report.json")
TABLE_PATH = os.path.join(PROJECT_ROOT, "calibration", "reliability_table.json")

MATCHED_CELLS = ("real_png", "fake_png", "real_jpeg_q95", "fake_jpeg_q95",
                 "real_jpeg_q85", "fake_jpeg_q85", "real_jpeg_q70", "fake_jpeg_q70")
MATCHED_TREATMENTS = ("png", "jpeg_q95", "jpeg_q85", "jpeg_q70")

# G7 A/B/C: fitting reads group A and nothing else.  The G2 experts used to be
# binned on all 392 matched samples — A+B+C, the very groups a later scheme is
# judged on — and their single weight was the png cell's separation applied to
# every format (jpeg's q70 cell is 0.569, not 0.972).  Both are fixed here;
# the previous table is kept as reliability_table_v1.json for replay.
FITTING_GROUP = "A"

# G2 report name -> expert source_name used at runtime.
# Both frequency versions share source_name "frequency_expert"; v1 is what the
# pipeline actually runs, so it owns the runtime key and v2 is stored apart.
SOURCE_MAP = {
    "frequency_v1": "frequency_expert",
    "frequency_v2": "frequency_expert_v2",
    "noise": "noise_expert",
    "jpeg": "jpeg_expert",
    "ela": "ela_expert",
}

SPLIT_PATH = os.path.join(SET_DIR, "g7_split.json")

# G7: external experts are calibrated on the **fitting group only**.  The G2
# experts were binned on every sample the manifest holds (all 700, which now
# includes B and C) — a table fitted on the groups a scheme is later judged on
# cannot judge it.  For anything new, the bins come from A and nowhere else.
EXTERNAL_EXPERTS = {
    "probe_dino_expert": {
        "report": os.path.join(PROJECT_ROOT, "calibration", "g7_external_probe_dino_A.json"),
        "group": "A",
        "treatments": ("png", "jpeg_q95", "jpeg_q85", "jpeg_q70"),
        "conditions": (
            "五格式全部适用（匹配格最低校正分离度 {worst:.3f}，q70 反而最强 {best:.3f}）；"
            "未见过的生成器上整体分离度 0.952。分数为整图 P(AI-generated)，不指出任何区域。"
        ),
    },
}


def applicability_label(aligned: bool, separation: float, drop: Optional[float]) -> str:
    """Rank an expert for whole-image generation detection (G2 §4.9)."""
    if separation < 0.55:
        return "disabled:no-signal"
    if not aligned:
        return "inverted:high-metric-means-real"
    if drop is not None and drop >= 0.15:
        return "shortcut-prone:compression-history"
    if separation >= 0.8:
        return "strong"
    return "weak:marginally-above-chance"


# Human-readable usage conditions keyed by applicability label (G2-c: these
# travel into every Evidence Bundle so the MLLM and auditors see *when* the
# measurement may be trusted, not just the number).
APPLICABILITY_CONDITIONS = {
    "disabled:no-signal": (
        "分离度≈随机，不得作为证据使用；仅保留用于回归对照。"
    ),
    "weak:marginally-above-chance": (
        "仅可作为辅助线索，不可单独支撑判定；必须与其他证据交叉验证。"
    ),
    "inverted:high-metric-means-real": (
        "方向与本任务语义相反：高 strength 统计上对应 Real。"
        "不得按原 support 标签解读；如需使用必须反转极性并重新校准。"
    ),
    "shortcut-prone:compression-history": (
        "分离度主要来自压缩历史差异。仅适用于未压缩或高质量来源（PNG、JPEG q≥85）；"
        "图像经过统一重压缩（q≤70 或社交传播）后失效。"
    ),
    "strong": (
        "可直接使用；仍需记录条件元数据并在结论中说明适用条件。"
    ),
}

# Per-expert condition overrides with the measured numbers.
EXPERT_CONDITION_OVERRIDES = {
    "frequency_expert": (
        "无有效信号（png 分离度 0.505，q70 0.541）；停用，不进入运行时工具箱。"
    ),
    "frequency_expert_v2": (
        "弱信号（png 0.556，q70 0.634）——唯一在压缩配平后仍略升的专家；"
        "仅作辅助线索，不得单独支撑判定。"
    ),
    "noise_expert": (
        "高 strength 统计上对应 Real（JPEG 压缩引入方差），不得按 AI-generated 解读；"
        "低 strength 侧（P(Fake)≈0.8）可作为弱证据使用。"
    ),
    "jpeg_expert": (
        "高 strength 统计上对应 Real（原生 JPEG 照片）；q70 配平后分离度仅 0.57。"
        "仅当来源无压缩历史时可反向参考。"
    ),
    "ela_expert": (
        "仅适用于未压缩或高质量来源（PNG / JPEG q≥85）；统一重压缩后（q70 校准）"
        "分离度降至随机（0.504）。"
    ),
}


def quantile_bins(values: List[float], labels: List[str], bins: int) -> List[dict]:
    """Equal-count bins with empirical P(Fake); outer bins are open-ended."""
    values_arr = np.asarray(values, dtype=np.float64)
    labels_arr = np.asarray(labels)
    edges = np.quantile(values_arr, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)

    output = []
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (values_arr >= low) & (values_arr < high)
        if index == len(edges) - 2:
            mask = (values_arr >= low)  # include the top edge in the last bin
        count = int(mask.sum())
        if count == 0:
            continue
        output.append({
            "lo": None if index == 0 else low,
            "hi": None if index == len(edges) - 2 else high,
            "n": count,
            "p_fake": float((labels_arr[mask] == "Fake").mean()),
        })
    return output


def _auroc(values: List[float], labels: List[str]) -> Optional[float]:
    """Rank AUROC with Fake as the positive class; None when a class is tiny."""
    positive = [v for v, y in zip(values, labels) if y == "Fake"]
    negative = [v for v, y in zip(values, labels) if y == "Real"]
    if len(positive) < 3 or len(negative) < 3:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(np.asarray(positive + negative, dtype=float))
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def external_entry(name: str, config: dict, splits: dict, bins: int) -> Optional[dict]:
    """
    One runtime entry for an external expert, fitted on its fitting group.

    The score's polarity is measured rather than assumed: if the AUROC on the
    fitting group comes out below 0.5, the entry is marked unaligned and the
    separation is stored corrected, exactly as the G2 entries do.
    """
    path = config["report"]
    if not os.path.exists(path):
        print(f"  {name:22s} skipped: {os.path.relpath(path, PROJECT_ROOT)} 不存在")
        return None
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)

    members = set(splits["groups"][config["group"]])
    rows = [row for row in report["rows"]
            if row["source_id"] in members and row["treatment"] in config["treatments"]]
    if len(rows) < 20:
        print(f"  {name:22s} skipped: 拟合组样本不足（{len(rows)}）")
        return None

    values = [float(row["score"]) for row in rows]
    labels = [row["label"] for row in rows]
    overall = _auroc(values, labels)
    aligned = bool(overall is not None and overall >= 0.5)
    separation = overall if aligned else (1.0 - overall if overall is not None else None)

    per_treatment = {}
    for treatment in config["treatments"]:
        subset = [row for row in rows if row["treatment"] == treatment]
        value = _auroc([float(r["score"]) for r in subset], [r["label"] for r in subset])
        per_treatment[treatment] = None if value is None else (
            value if aligned else 1.0 - value)

    measured = [v for v in per_treatment.values() if v is not None]
    drop = None
    if per_treatment.get("png") is not None and per_treatment.get("jpeg_q70") is not None:
        drop = float(per_treatment["jpeg_q70"] - per_treatment["png"])

    label = applicability_label(aligned, separation if separation is not None else 0.5, drop)
    conditions = config["conditions"].format(
        worst=min(measured) if measured else float("nan"),
        best=max(measured) if measured else float("nan"),
    )
    weights = (report.get("weights") or {})
    return {
        "g2_name": None,                       # not a G2 expert
        "semantics_aligned": aligned,
        "separation_polarity_corrected": separation,
        "confound_drop_png_to_q70": drop,
        "applicability": label,
        "applicability_conditions": conditions,
        "n_samples": len(values),
        "bins": quantile_bins(values, labels, bins),
        # Same shape as the G2 entries: the runtime weights a token by its own
        # format's measurement whenever the format is known.
        "per_cell_separation": {
            key: (None if value is None else round(float(value), 4))
            for key, value in per_treatment.items()},
        "provenance": {
            "fitted_on": config["group"],
            "mapping": "quantile_bins",
            "sources": len({row["source_id"] for row in rows}),
            "samples": len(values),
            "treatments": list(config["treatments"]),
            "split_hash": splits.get("hash"),
            "report": os.path.relpath(path, PROJECT_ROOT),
            "weights_sha256": weights.get("sha256"),
            "per_treatment_separation": {
                key: (None if value is None else round(float(value), 4))
                for key, value in per_treatment.items()},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bins", type=int, default=5)
    args = parser.parse_args()

    for path in (RAW_VALUES_PATH, MANIFEST_PATH, REPORT_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"missing input: {path}")

    with open(RAW_VALUES_PATH, encoding="utf-8") as handle:
        raw_values = json.load(handle)
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(REPORT_PATH, encoding="utf-8") as handle:
        report = json.load(handle)

    # cell -> (label, source_id) from the manifest
    sample_meta = {
        sample["sample_id"]: sample for sample in manifest["samples"]
    }

    # The fitting group is a declaration, not a default: without it the
    # builder would silently fit on everything the manifest holds.
    if not os.path.exists(SPLIT_PATH):
        raise SystemExit(
            f"the fitting split is required (fit on group {FITTING_GROUP} only): "
            f"missing {SPLIT_PATH}")
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        splits = json.load(handle)
    fitting_sources = set(splits["groups"][FITTING_GROUP])

    table: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema": 2,
        "source_raw_values": os.path.relpath(RAW_VALUES_PATH, PROJECT_ROOT),
        "cells_used": list(MATCHED_CELLS),
        "bins": args.bins,
        "fitting_group": FITTING_GROUP,
        "fitting_split_hash": splits.get("hash"),
        "experts": {},
    }

    for g2_name, entry in raw_values["experts"].items():
        source = SOURCE_MAP.get(g2_name)
        if source is None:
            continue
        report_entry = report.get("experts", {}).get(g2_name, {})

        values, labels, treatments = [], [], []
        for sample_id, raw in entry["raw"].items():
            meta = sample_meta.get(sample_id)
            if meta is None or meta["cell"] not in MATCHED_CELLS:
                continue
            if meta["source_id"] not in fitting_sources:
                continue                      # fitting reads group A only
            values.append(float(raw))
            labels.append(meta["label"])
            treatments.append(meta["treatment"])

        if len(values) < 20:
            continue

        # The separation is measured on the fitting samples, per format as well
        # as pooled: the single number the G2 report carried was its png cell,
        # and using it for a q70 image over-weights the expert by 0.4.
        overall = _auroc(values, labels)
        aligned = bool(overall is not None and overall >= 0.5)
        separation = None if overall is None else round(
            overall if aligned else 1.0 - overall, 4)
        per_cell: Dict[str, Optional[float]] = {}
        for treatment in MATCHED_TREATMENTS:
            subset = [(v, y) for v, y, t in zip(values, labels, treatments)
                      if t == treatment]
            auroc = _auroc([v for v, _ in subset], [y for _, y in subset])
            per_cell[treatment] = None if auroc is None else round(
                auroc if aligned else 1.0 - auroc, 4)
        drop = None
        if per_cell.get("png") is not None and per_cell.get("jpeg_q70") is not None:
            drop = round(per_cell["jpeg_q70"] - per_cell["png"], 4)

        label = applicability_label(
            aligned, separation if separation is not None else 0.5, drop
        )
        table["experts"][source] = {
            "g2_name": g2_name,
            "semantics_aligned": aligned,
            "separation_polarity_corrected": separation,
            # The weight to use when the format is known; `pooled` when it is
            # not.  Never a single cell's value applied to every format.
            "per_cell_separation": per_cell,
            "confound_drop_png_to_q70": drop,
            "applicability": label,
            "applicability_conditions": EXPERT_CONDITION_OVERRIDES.get(
                source, APPLICABILITY_CONDITIONS.get(label, "")
            ),
            "n_samples": len(values),
            "bins": quantile_bins(values, labels, args.bins),
            "provenance": {
                "fitted_on": FITTING_GROUP,
                "mapping": "quantile_bins",
                "sources": len({sample_meta[sid]["source_id"]
                                for sid in entry["raw"]
                                if sid in sample_meta
                                and sample_meta[sid]["source_id"] in fitting_sources}),
                "samples": len(values),
                "split_hash": splits.get("hash"),
                "report": os.path.relpath(REPORT_PATH, PROJECT_ROOT),
                "report_separation_png_cell": report_entry.get(
                    "separation_polarity_corrected"),
            },
        }
        print(f"  {source:22s} separation={separation} aligned={aligned} "
              f"drop={drop} per-cell={per_cell} "
              f"applicability={table['experts'][source]['applicability']}")

    # G7: external experts, fitted on their fitting group only.
    for name, config in EXTERNAL_EXPERTS.items():
        entry = external_entry(name, config, splits, args.bins)
        if entry is None:
            continue
        table["experts"][name] = entry
        print(f"  {name:22s} separation={entry['separation_polarity_corrected']:.4f} "
              f"aligned={entry['semantics_aligned']} "
              f"applicability={entry['applicability']} "
              f"（拟合组 {entry['provenance']['fitted_on']}，"
              f"{entry['provenance']['sources']} 源 / {entry['n_samples']} 样本）")

    with open(TABLE_PATH, "w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=2)
    print(f"\nReliability table: {TABLE_PATH}")
    print(f"Experts: {list(table['experts'])}")


if __name__ == "__main__":
    main()
