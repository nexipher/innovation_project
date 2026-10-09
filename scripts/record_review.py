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

The workstation keeps its decisions in the browser and exports them as JSON
(or the reviewer fills `worklist.csv`); `--import` feeds either file through
the same validation as the single-sample path, so nothing reaches the training
set that the CLI would have rejected.

Usage:
  python scripts/record_review.py --sample f2_noise__ADM_x_png \
      --reviewer hj --decision accept --notes "证据与图像一致"
  python scripts/record_review.py --import review_decisions.json --reviewer hj
  python scripts/record_review.py --import sft_data/review/worklist.csv --summary
  python scripts/record_review.py --summary
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

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


def parse_export(path: str) -> Tuple[str, Dict[str, dict]]:
    """
    Read a browser export or a filled worklist into {sample_id: entry}.

    The export is `{"reviewer": ..., "decisions": {id: {decision, notes,
    saved_at}}}`; the CSV is the worklist with `sample_id`, `decision`,
    `notes` and optionally `reviewer` columns.  Rows with no decision are
    skipped — an untouched spreadsheet is not a set of decisions.
    """
    with open(path, encoding="utf-8", newline="") as handle:
        if path.lower().endswith(".csv"):
            reviewer = ""
            entries: Dict[str, dict] = {}
            for row in csv.DictReader(handle):
                if not (row.get("decision") or "").strip():
                    continue
                sample_id = (row.get("sample_id") or "").strip()
                entries[sample_id] = {
                    "decision": row["decision"].strip(),
                    "notes": (row.get("notes") or "").strip(),
                    "saved_at": (row.get("saved_at") or "").strip(),
                    "reviewer": (row.get("reviewer") or "").strip(),
                }
                reviewer = reviewer or (row.get("reviewer") or "").strip()
            return reviewer, entries

        payload = json.load(handle)
    decisions = payload.get("decisions") if isinstance(payload, dict) else None
    if decisions is None:
        # A bare {id: {...}} map is accepted, but only when the entries really
        # are decisions — any other JSON object should stop the import rather
        # than silently contributing zero rows.
        decisions = {key: value for key, value in (payload or {}).items()
                     if isinstance(value, dict) and "decision" in value}
        if not decisions:
            raise ValueError(f"{path}: not a workstation export (no 'decisions' object)")
    if not isinstance(decisions, dict):
        raise ValueError(f"{path}: not a workstation export (no 'decisions' object)")
    entries = {}
    for sample_id, entry in decisions.items():
        if not isinstance(entry, dict) or not (entry.get("decision") or "").strip():
            continue
        entries[sample_id] = {
            "decision": entry["decision"].strip(),
            "notes": (entry.get("notes") or "").strip(),
            "saved_at": (entry.get("saved_at") or "").strip(),
            "reviewer": (entry.get("reviewer") or "").strip(),
        }
    return (payload.get("reviewer") or "").strip(), entries


def import_decisions(path: str, reviewer: Optional[str] = None,
                     dispositions_path: str = DISPOSITIONS_PATH,
                     final_v2_dir: str = FINAL_V2_DIR) -> dict:
    """
    Append every decided row of an export, skipping what cannot be imported.

    A typo'd id in a spreadsheet must not cost the reviewer the other 236
    rows, so unknown ids are collected and reported instead of raising; ids
    whose decision is already the latest with the same notes are counted as
    unchanged, which makes a re-import of the same file a no-op.
    """
    exported_reviewer, entries = parse_export(path)
    fallback_reviewer = reviewer or exported_reviewer
    latest = load_dispositions(dispositions_path)
    known = set(known_sample_ids(final_v2_dir))

    imported, unchanged, unknown, invalid, by_decision = 0, 0, [], [], Counter()
    for sample_id, entry in entries.items():
        if sample_id not in known:
            unknown.append(sample_id)
            continue
        if entry["decision"] not in DECISIONS:
            invalid.append((sample_id, entry["decision"]))
            continue
        previous = latest.get(sample_id)
        if previous and previous["decision"] == entry["decision"] \
                and (previous.get("notes") or "") == entry["notes"]:
            unchanged += 1
            continue
        who = entry["reviewer"] or fallback_reviewer
        if not who:
            raise ValueError("a reviewer name is required: pass --reviewer or "
                             "export from the page after filling it in")
        record(sample_id, who, entry["decision"], entry["notes"],
               dispositions_path, final_v2_dir)
        imported += 1
        by_decision[entry["decision"]] += 1

    return {"file": path, "rows": len(entries), "imported": imported,
            "unchanged": unchanged, "unknown": sorted(unknown),
            "invalid": invalid, "by_decision": dict(by_decision)}


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
    parser.add_argument("--import", "--import-file", dest="import_file", default=None,
                        help="browser export (JSON) or filled worklist.csv")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--dispositions", default=DISPOSITIONS_PATH)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    args = parser.parse_args()

    if args.import_file:
        report = import_decisions(args.import_file, args.reviewer,
                                  args.dispositions, args.final_v2_dir)
        print(f"imported {report['imported']} decisions from {report['file']} "
              f"({report['unchanged']} unchanged, {report['rows']} rows)")
        print(f"by decision: {report['by_decision']}")
        if report["unknown"]:
            print(f"skipped {len(report['unknown'])} unknown sample ids: "
                  f"{report['unknown'][:5]}")
        if report["invalid"]:
            print(f"skipped {len(report['invalid'])} rows with an unknown decision: "
                  f"{report['invalid'][:5]}")

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

    if args.import_file:
        return

    if not (args.sample and args.reviewer and args.decision):
        parser.error("--sample, --reviewer and --decision are required "
                     "(or use --import-file / --summary)")
    entry = record(args.sample, args.reviewer, args.decision, args.notes,
                   args.dispositions, args.final_v2_dir)
    print(f"recorded: {entry['sample_id']} -> {entry['decision']} by {entry['reviewer']}")


if __name__ == "__main__":
    main()
