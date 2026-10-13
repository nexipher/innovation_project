#!/usr/bin/env python3
"""
Review material straight from a batch's trajectories (no admission applied).

The b1 workflow went trajectories → admission → conversion → workstation.  A
single-arm acceptance batch has no no-tool baseline to pair against, so
admission would reject every trajectory for a reason that has nothing to do
with what this review is about.  This script renders the sessions into the
workstation's sample shape **without** claiming they were admitted: every
sample says `audit.admission = not_applied_review_only`, and the batch note
records the same thing.

What the reviewer gets is what the acceptance asks for: the raw assistant
turns beside the converted four-section answer (the workstation's fidelity
section), the evidence tokens, the ledger's findings, and the verdict with
its `verdict_status`.

Usage:
  python scripts/build_review_from_trajectories.py --batch g7_b2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.batch import batch_paths, load_trajectory_records
from utils.final_v2 import render_sample


def load_trace(record: dict) -> dict:
    path = record.get("trace_path")
    if not path:
        return {}
    try:
        with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _observation(record: dict) -> str:
    """
    The model's own visual account, read by the one implementation that knows
    both shapes it can take (an <observation> block, or the planning line).

    This script used to look only for the block, so every session of the
    v3 batch — whose account lives in the planning line — lost its words and
    the sample said nothing about the image.  Reading it a second way here
    would drift from the converter; calling it keeps one definition.
    """
    from scripts.build_final_v2 import extract_observation

    return extract_observation(record)


def build(batch_id: str, split: Dict) -> Dict:
    paths = batch_paths(batch_id)
    records = list(load_trajectory_records(paths["trajectories"]).values())
    if not records:
        raise SystemExit(f"{paths['trajectories']} 里没有轨迹")

    variants = {}
    if os.path.exists(paths["variants"]):
        with open(paths["variants"], encoding="utf-8") as handle:
            variants = {entry["variant_id"]: path
                        for path, entry in json.load(handle)["variants"].items()}

    samples = []
    for record in records:
        trace = load_trace(record)
        sample = render_sample(
            {**record, "image_path": variants.get(record.get("variant_id"), "")},
            "positive",
            observation_text=_observation({**record, "trace_path": record.get("trace_path")}),
            split_entry=(split or {}).get("sources", {}).get(record.get("source_id")),
            trace=trace)
        # Not an admitted sample: this is review material and says so.
        sample["source"] = "g4_review_only"
        sample["audit"] = {"review_status": "pending_human",
                           "review_method": "automated_validator",
                           "admission": "not_applied_review_only"}
        samples.append(sample)

    os.makedirs(paths["samples"], exist_ok=True)
    with open(os.path.join(paths["samples"], "sft_tool_positive.json"), "w",
              encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False, indent=2)
    note = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch": batch_id,
        "purpose": "v3 acceptance review (single arm; admission not applied)",
        "samples": len(samples),
        "review_files": "sft_tool_positive.json",
    }
    with open(os.path.join(paths["samples"], "review_only.json"), "w",
              encoding="utf-8") as handle:
        json.dump(note, handle, ensure_ascii=False, indent=2)
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="g7_b2")
    parser.add_argument("--no-workstation", action="store_true")
    args = parser.parse_args()

    split_path = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
    with open(split_path, encoding="utf-8") as handle:
        split = json.load(handle)
    note = build(args.batch, split)
    print(f"review samples: {note['samples']} -> "
          f"{os.path.relpath(batch_paths(args.batch)['samples'], PROJECT_ROOT)}")

    if not args.no_workstation:
        from scripts.build_review_workstation import build as build_workstation

        paths = batch_paths(args.batch)
        out = paths["review"]
        result = build_workstation(paths["samples"], out,
                                   trajectories_dir=paths["trajectories"],
                                   variants_manifest=paths["variants"],
                                   batch_id=args.batch)
        print(f"workstation: {result} -> {os.path.relpath(out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
