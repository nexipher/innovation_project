#!/usr/bin/env python3
"""
G7 §4.17: model verdict → system label, as a matrix with the ids behind it.

The handoff's correction, made concrete: "68 overrides" and "61 overrides"
were both reported, and the second came from subtracting two aggregate counts
(123 model-Uncertains minus 62 system-Uncertains).  That subtraction assumes
transitions only run one way.  They do not: seven sessions moved the other
direction, one moved between labels, and nine produced no parseable verdict at
all while the system still assigned a label.

So this script emits the matrix and, for every cell, the trajectory ids.  No
count is derived by arithmetic on other counts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.g7_abstention_diagnosis import parse_verdicts
from utils.batch import load_trajectory_records

TRACES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_g7_b1")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_label_transitions.json")

LABELS = ("Uncertain", "Real", "Fake", "NONE")


def transitions(traces_dir: str = TRACES_DIR) -> dict:
    # Archived 2026-10-15: without this the empty read would produce an all-zero
    # matrix that looks like a report — the exact false zero this file exists
    # to prevent.
    from utils.batch import resolve_path

    traces_dir, from_archive = resolve_path(traces_dir)
    records = load_trajectory_records(traces_dir)
    if not records:
        raise SystemExit(f"没有轨迹记录：{traces_dir}")
    matrix: Counter = Counter()
    ids: Dict[tuple, List[str]] = defaultdict(list)
    for trajectory_id, record in sorted(records.items()):
        with open(os.path.join(PROJECT_ROOT, record["trace_path"]), encoding="utf-8") as handle:
            trace = json.load(handle)
        verdicts, _ = parse_verdicts(trace)
        model_last = verdicts[-1] if verdicts else "NONE"
        system = record.get("final_verdict") or "NONE"
        matrix[(model_last, system)] += 1
        ids[(model_last, system)].append(trajectory_id)

    def count(*cells) -> int:
        return sum(matrix.get(cell, 0) for cell in cells)

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "definition": ("行 = 模型最后一次**可解析**的 <verdict>；列 = 控制器采纳的 final_verdict。"
                       "任何计数都取自矩阵本身，不由其他汇总数相减得到"),
        "traces_dir": os.path.relpath(traces_dir, PROJECT_ROOT),
        "resolved_from_archive": bool(from_archive),
        "matrix": {f"{row}->{col}": matrix.get((row, col), 0)
                   for row in LABELS for col in LABELS},
        "ids": {f"{row}->{col}": ids[(row, col)]
                for row in LABELS for col in LABELS if ids[(row, col)]},
        "headline": {
            "uncertain_overridden_to_label": count(("Uncertain", "Real"), ("Uncertain", "Fake")),
            "label_overridden_to_uncertain": count(("Real", "Uncertain"), ("Fake", "Uncertain")),
            "label_overridden_to_other_label": count(("Real", "Fake"), ("Fake", "Real")),
            "no_parseable_verdict_but_labeled": count(("NONE", "Real"), ("NONE", "Fake")),
            "uncertain_kept": count(("Uncertain", "Uncertain")),
        },
        "note": ("此前报出的 68 与 61 是同一现象的两个数：68 来自逐会话口径（正确），"
                 "61 来自 123−62 的相减（错误，方向相反的转移与无法解析的会话都不在差里）。"
                 "后续引用一律以本矩阵为准"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=TRACES_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    report = transitions(args.traces)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    io.open(args.output, "w", encoding="utf-8").write(
        json.dumps(report, ensure_ascii=False, indent=1))

    print(f"{'model\\system':>14s} " + " ".join(f"{c:>10s}" for c in LABELS))
    for row in LABELS:
        cells = [report["matrix"][f"{row}->{col}"] for col in LABELS]
        if sum(cells) or row != "NONE":
            print(f"{row:>14s} " + " ".join(f"{c:10d}" for c in cells))
    print("\n" + json.dumps(report["headline"], ensure_ascii=False))
    print(f"\nReport: {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
