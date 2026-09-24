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

    table: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_raw_values": os.path.relpath(RAW_VALUES_PATH, PROJECT_ROOT),
        "cells_used": list(MATCHED_CELLS),
        "bins": args.bins,
        "experts": {},
    }

    for g2_name, entry in raw_values["experts"].items():
        source = SOURCE_MAP.get(g2_name)
        if source is None:
            continue
        report_entry = report.get("experts", {}).get(g2_name, {})

        values, labels = [], []
        for sample_id, raw in entry["raw"].items():
            meta = sample_meta.get(sample_id)
            if meta is None or meta["cell"] not in MATCHED_CELLS:
                continue
            values.append(float(raw))
            labels.append(meta["label"])

        if len(values) < 20:
            continue

        separation = report_entry.get("separation_polarity_corrected")
        aligned = bool(report_entry.get("semantics_aligned", True))
        drop = report_entry.get("confound_drop_png_to_q70", {}).get("drop")

        table["experts"][source] = {
            "g2_name": g2_name,
            "semantics_aligned": aligned,
            "separation_polarity_corrected": separation,
            "confound_drop_png_to_q70": drop,
            "applicability": applicability_label(
                aligned, separation if separation is not None else 0.5, drop
            ),
            "n_samples": len(values),
            "bins": quantile_bins(values, labels, args.bins),
        }
        print(f"  {source:22s} separation={separation} aligned={aligned} "
              f"drop={drop if drop is None else round(drop, 3)} "
              f"applicability={table['experts'][source]['applicability']}")

    with open(TABLE_PATH, "w", encoding="utf-8") as handle:
        json.dump(table, handle, ensure_ascii=False, indent=2)
    print(f"\nReliability table: {TABLE_PATH}")
    print(f"Experts: {list(table['experts'])}")


if __name__ == "__main__":
    main()
