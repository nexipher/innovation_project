#!/usr/bin/env python3
"""
G7: per-cell applicability for the external experts, in the G2 report's shape.

The G4 trajectory generator and the final_v2 validator both gate a tool by
"how well did it separate on *this* treatment", read from
`calibration/g2_expert_report.json`.  PROBE has no entry there — it was
measured later, on a different split — so without this file every probe policy
is refused with "no calibration entry" and every probe token is flagged by the
validator.

The numbers come from the fitting group **only** (A): the cell separations are
what decides whether a tool may carry a trajectory, and a threshold chosen on
the group a trajectory is later judged on is not a threshold, it is a leak.

Usage:
  python scripts/build_g7_applicability.py
  python scripts/build_g7_applicability.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

SPLIT_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_expert_report.json")

# Report name (what `tool_applicable` looks up) -> where its measurements are.
EXTERNAL_REPORTS = {
    "probe_dino": os.path.join(PROJECT_ROOT, "calibration", "g7_external_probe_dino_A.json"),
}


def auroc(values: List[float], labels: List[str]) -> Optional[float]:
    """Rank AUROC, Fake positive; None when a class is too small to rank."""
    positive = [v for v, y in zip(values, labels) if y == "Fake"]
    negative = [v for v, y in zip(values, labels) if y == "Real"]
    if len(positive) < 3 or len(negative) < 3:
        return None
    from scipy.stats import rankdata
    ranks = rankdata(np.asarray(positive + negative, dtype=float))
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def cells(report: dict, group: str, splits: dict) -> Dict[str, dict]:
    """Per-treatment polarity-corrected separation for one fitting group."""
    members = set(splits["groups"][group])
    rows = [row for row in report["rows"] if row["source_id"] in members]
    inverted = False
    overall = auroc([float(r["score"]) for r in rows], [r["label"] for r in rows])
    if overall is not None and overall < 0.5:
        inverted = True

    by_treatment: Dict[str, List[dict]] = {}
    for row in rows:
        by_treatment.setdefault(row["treatment"], []).append(row)

    table = {}
    for treatment, members_rows in sorted(by_treatment.items()):
        value = auroc([float(r["score"]) for r in members_rows],
                      [r["label"] for r in members_rows])
        if value is None:
            continue
        table[treatment] = {
            "n": len(members_rows),
            "n_real": sum(1 for r in members_rows if r["label"] == "Real"),
            "n_fake": sum(1 for r in members_rows if r["label"] == "Fake"),
            # The validator's gate reads the *corrected* separation, so the
            # polarity is settled here rather than assumed by each caller.
            "auroc_fake_vs_real": round(1.0 - value if inverted else value, 4),
        }
    return {"semantics_aligned": not inverted, "cell_metrics": table}


def build(dry_run: bool = False) -> Tuple[dict, list]:
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        splits = json.load(handle)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "external experts 的逐格适用性，供 G4 生成器与 final_v2 校验器使用",
        "fitted_on": "A",
        "split_hash": splits.get("hash"),
        "sources": {},
        "experts": {},
    }
    lines = []
    for name, path in EXTERNAL_REPORTS.items():
        if not os.path.exists(path):
            lines.append(f"  {name:14s} skipped: {os.path.relpath(path, PROJECT_ROOT)} 不存在")
            continue
        with open(path, encoding="utf-8") as handle:
            blob = json.load(handle)
        entry = cells(blob, "A", splits)
        report["experts"][name] = entry
        report["sources"][name] = os.path.relpath(path, PROJECT_ROOT)
        worst = min(entry["cell_metrics"].items(), key=lambda kv: kv[1]["auroc_fake_vs_real"])
        lines.append(f"  {name:14s} 格数 {len(entry['cell_metrics'])}  "
                     f"最低 {worst[0]} {worst[1]['auroc_fake_vs_real']:.4f}  "
                     f"aligned={entry['semantics_aligned']}")

    if not dry_run:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    return report, lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report, lines = build(dry_run=args.dry_run)
    for line in lines:
        print(line)
    if args.dry_run:
        print("（dry-run，未写文件）")
        return
    print(f"\nReport: {os.path.relpath(OUTPUT_PATH, PROJECT_ROOT)}")
    print(f"Experts: {list(report['experts'])}")


if __name__ == "__main__":
    main()
