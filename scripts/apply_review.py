#!/usr/bin/env python3
"""
G4-f → G4-g: turn review decisions into the training set.

Reads the recorded dispositions and writes `sft_data/train/final_v2_reviewed/`:

  sft_accepted.json    accept — trains the answer *and* the conclusion
  sft_format_only.json accept for format, content not trusted
  sft_to_revise.json   revise — not trained until fixed
  sft_rejected_reviewed.json  reject — never trained
  metadata.json        counts, composition, class weights

It refuses to run while any sample is undecided: a partially reviewed set would
train on samples nobody looked at, which is the failure mode the review exists
to prevent.  `--allow-partial` exists for a dry look and says so in the output.

The class weights are computed for the accepted set, because the composition is
skewed by construction (evidence only pays where the no-tool baseline is
confidently wrong, which is mostly Fake) and training on the raw mix would
teach a "tools mean Fake" prior.

The frozen sets are untouched: neither `final/` nor `final_v2/` is written to.

Usage:
  python scripts/apply_review.py
  python scripts/apply_review.py --allow-partial
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.record_review import BUCKET_FILES, load_dispositions

FINAL_V2_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2_reviewed")
FROZEN_DIRS = (os.path.join(PROJECT_ROOT, "sft_data", "train", "final"),
               FINAL_V2_DIR)

OUTPUT_FILES = {
    "accept": "sft_accepted.json",
    "format_only": "sft_format_only.json",
    "revise": "sft_to_revise.json",
    "reject": "sft_rejected_reviewed.json",
}


def assert_not_frozen(output_dir: str) -> None:
    resolved = os.path.realpath(output_dir)
    for frozen in FROZEN_DIRS:
        frozen = os.path.realpath(frozen)
        if resolved == frozen or resolved.startswith(frozen + os.sep):
            raise ValueError(f"refusing to write into a frozen set: {output_dir}")


def load_samples(final_v2_dir: str = FINAL_V2_DIR) -> List[dict]:
    samples = []
    for name in BUCKET_FILES:
        path = os.path.join(final_v2_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            samples.extend(json.load(handle))
    return samples


def class_weights(labels: collections.Counter) -> Dict[str, float]:
    """Weights that make one epoch see the classes equally."""
    if not labels:
        return {}
    largest = max(labels.values())
    return {label: round(largest / count, 4) for label, count in sorted(labels.items())}


def apply(dispositions: Dict[str, dict], final_v2_dir: str = FINAL_V2_DIR,
          output_dir: str = OUTPUT_DIR, allow_partial: bool = False) -> dict:
    assert_not_frozen(output_dir)
    samples = load_samples(final_v2_dir)

    undecided = [s["id"] for s in samples if s["id"] not in dispositions]
    if undecided and not allow_partial:
        raise SystemExit(
            f"{len(undecided)} samples are undecided, e.g. {undecided[:3]}; "
            f"finish the review or pass --allow-partial to write a provisional set")

    buckets: Dict[str, List[dict]] = {decision: [] for decision in OUTPUT_FILES}
    for sample in samples:
        entry = dispositions.get(sample["id"])
        if entry is None:
            continue
        bucket = dict(sample)
        bucket["review"] = {
            "reviewer": entry["reviewer"],
            "decision": entry["decision"],
            "notes": entry.get("notes", ""),
            "recorded_at": entry.get("recorded_at"),
        }
        buckets[entry["decision"]].append(bucket)

    os.makedirs(output_dir, exist_ok=True)
    for decision, filename in OUTPUT_FILES.items():
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as handle:
            json.dump(buckets[decision], handle, ensure_ascii=False, indent=2)

    accepted = buckets["accept"]
    labels = collections.Counter(s["ground_truth"] for s in accepted)
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provisional": bool(undecided),
        "undecided": len(undecided),
        "counts": {decision: len(rows) for decision, rows in buckets.items()},
        "accepted_labels": dict(labels),
        "accepted_training_weights": class_weights(labels),
        "accepted_buckets": dict(collections.Counter(s.get("type") for s in accepted)),
        "reviewers": dict(collections.Counter(
            entry["reviewer"] for entry in dispositions.values())),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--dispositions", default=None)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    dispositions = load_dispositions(args.dispositions) if args.dispositions \
        else load_dispositions()
    metadata = apply(dispositions, args.final_v2_dir, args.output_dir, args.allow_partial)

    print(f"counts: {metadata['counts']}")
    if metadata["provisional"]:
        print(f"PROVISIONAL: {metadata['undecided']} samples undecided and excluded")
    print(f"accepted labels: {metadata['accepted_labels']}")
    print(f"accepted training weights: {metadata['accepted_training_weights']}")
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
