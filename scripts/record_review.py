#!/usr/bin/env python3
"""
G4-f: record review decisions.

Each sample gets one of four dispositions, the same vocabulary G0 used so the
two audits read alike:

  accept       训练可用（内容与证据都可信）
  format_only  格式可用但内容不可信（只用于格式学习，不用于结论）
  revise       需要修改后才能训练（进入待修清单，不进训练集）
  reject       严重错误 / 误导性样本（进拒绝集，永不训练）

Decisions are appended to `sft_data/review/dispositions.jsonl` — one JSON
object per line, never rewritten — and the latest decision per sample wins, so
a reviewer can change their mind without losing the history.

Usage:
  python scripts/record_review.py --sample f2_noise__ADM_x_png \
      --reviewer hj --decision accept --notes "证据与图像一致"
  python scripts/record_review.py --summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

DISPOSITIONS_PATH = os.path.join(PROJECT_ROOT, "sft_data", "review", "dispositions.jsonl")
FINAL_V2_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
BUCKET_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                "sft_honest_abstention.json")

DECISIONS = ("accept", "format_only", "revise", "reject")


def known_sample_ids(final_v2_dir: str = FINAL_V2_DIR) -> List[str]:
    identifiers = []
    for name in BUCKET_FILES:
        path = os.path.join(final_v2_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            identifiers.extend(sample["id"] for sample in json.load(handle))
    return identifiers


def load_dispositions(path: str = DISPOSITIONS_PATH) -> Dict[str, dict]:
    """Latest decision per sample; the file is a log, not a state."""
    latest: Dict[str, dict] = {}
    if not os.path.exists(path):
        return latest
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            latest[entry["sample_id"]] = entry
    return latest


def record(sample_id: str, reviewer: str, decision: str, notes: str = "",
           path: str = DISPOSITIONS_PATH,
           final_v2_dir: str = FINAL_V2_DIR) -> dict:
    """
    Append one decision, after checking it refers to a real sample.

    A decision attached to a typo'd id would silently leave the real sample
    undecided while looking reviewed, so the id is verified against the set.
    """
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}; expected one of {DECISIONS}")
    if not reviewer:
        raise ValueError("a reviewer name is required")
    if sample_id not in set(known_sample_ids(final_v2_dir)):
        raise ValueError(f"unknown sample id {sample_id!r}")

    entry = {
        "sample_id": sample_id,
        "reviewer": reviewer,
        "decision": decision,
        "notes": notes,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def summary(dispositions: Dict[str, dict],
            final_v2_dir: str = FINAL_V2_DIR) -> dict:
    samples = []
    for name in BUCKET_FILES:
        path = os.path.join(final_v2_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            samples.extend(json.load(handle))

    decided = {i: d for i, d in dispositions.items() if i in {s["id"] for s in samples}}
    by_decision = Counter(d["decision"] for d in decided.values())
    by_bucket = Counter()
    for sample in samples:
        entry = decided.get(sample["id"])
        by_bucket[(sample.get("type"), entry["decision"] if entry else "pending")] += 1
    reviewers = Counter(d["reviewer"] for d in decided.values())

    return {
        "samples": len(samples),
        "decided": len(decided),
        "pending": len(samples) - len(decided),
        "complete": len(decided) == len(samples),
        "by_decision": dict(by_decision),
        "by_bucket_and_decision": {f"{b}|{d}": n for (b, d), n in sorted(by_bucket.items())},
        "reviewers": dict(reviewers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample")
    parser.add_argument("--reviewer")
    parser.add_argument("--decision", choices=DECISIONS)
    parser.add_argument("--notes", default="")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--dispositions", default=DISPOSITIONS_PATH)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    args = parser.parse_args()

    if args.summary:
        report = summary(load_dispositions(args.dispositions), args.final_v2_dir)
        print(f"samples: {report['samples']}  decided: {report['decided']}  "
              f"pending: {report['pending']}  complete: {report['complete']}")
        print(f"by decision: {report['by_decision']}")
        for key, count in report["by_bucket_and_decision"].items():
            print(f"  {key:34s} {count}")
        if report["reviewers"]:
            print(f"reviewers: {report['reviewers']}")
        return

    if not (args.sample and args.reviewer and args.decision):
        parser.error("--sample, --reviewer and --decision are required (or use --summary)")
    entry = record(args.sample, args.reviewer, args.decision, args.notes,
                   args.dispositions, args.final_v2_dir)
    print(f"recorded: {entry['sample_id']} -> {entry['decision']} by {entry['reviewer']}")


if __name__ == "__main__":
    main()
