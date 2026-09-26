#!/usr/bin/env python3
"""
G4-d: assemble and validate `final_v2`.

Takes the trajectories G4-b produced, keeps the ones G4-c admitted, renders
each into the four-section answer, validates it against the record it came
from, and writes only what passes.

The old `final/` is frozen: it is the regression baseline, and this script
refuses to write anywhere inside it.

Output (`sft_data/train/final_v2/`):
  sft_tool_positive.json      tool trajectories whose risk fell and were right
  sft_no_tool_positive.json   baseline trajectories that were right
  sft_honest_abstention.json  abstentions kept as such
  sft_invalid.json            samples the validator rejected (never trained on)
  metadata.json               counts, provenance, validation summary

Usage:
  python scripts/build_final_v2.py --dry-run
  python scripts/build_final_v2.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.generate_trajectories_g4 import load_applicability
from utils.final_v2 import (
    RETIRED_EXPERTS,
    SCHEMA_VERSION,
    load_split,
    render_sample,
    validate_sample,
)

TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories")
ADMISSION_REPORT = os.path.join(PROJECT_ROOT, "sft_data", "admission_report.json")
VARIANTS_MANIFEST = os.path.join(PROJECT_ROOT, "sft_data", "variants", "manifest.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
FROZEN_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final")

FILES = {
    "positive": "sft_tool_positive.json",
    "no_tool_positive": "sft_no_tool_positive.json",
    "honest_abstention": "sft_honest_abstention.json",
}


def assert_not_frozen(output_dir: str) -> None:
    """The frozen baseline must survive this stage untouched."""
    resolved = os.path.realpath(output_dir)
    frozen = os.path.realpath(FROZEN_DIR)
    if resolved == frozen or resolved.startswith(frozen + os.sep):
        raise ValueError(f"refusing to write into the frozen baseline: {output_dir}")


def bucket_for(record: dict, category: str) -> str:
    if category == "positive":
        return "no_tool_positive" if not record.get("tools_served") else "positive"
    return "honest_abstention"


_PLANNING_ANOMALIES = re.compile(r"Visual Anomalies:\s*(.+)", re.I)


def extract_observation(record: dict) -> str:
    """
    The model's own first-turn observation, if the trace recorded one.

    Taken from `<planning>` — written before any evidence arrived, which is
    exactly what an observation section should contain.
    """
    path = record.get("trace_path")
    if not path:
        return ""
    try:
        with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as handle:
            trace = json.load(handle)
    except (OSError, ValueError):
        return ""
    for turn in trace.get("conversations", []):
        if turn.get("from") != "gpt":
            continue
        match = _PLANNING_ANOMALIES.search(turn.get("value", ""))
        if match:
            return f"模型视觉观察：{match.group(1).strip()}"
        break
    return ""


def load_trajectories(directory: str) -> Dict[str, dict]:
    records = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == "manifest.json":
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        records[record["trajectory_id"]] = record
    return records


def load_variant_paths(path: str = VARIANTS_MANIFEST) -> Dict[str, str]:
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {entry["variant_id"]: variant_path
            for variant_path, entry in manifest.get("variants", {}).items()}


def build(trajectory_dir: str = TRAJECTORIES_DIR,
          admission_path: str = ADMISSION_REPORT,
          output_dir: str = OUTPUT_DIR) -> dict:
    assert_not_frozen(output_dir)

    with open(admission_path, encoding="utf-8") as handle:
        admission = json.load(handle)
    admitted = [d for d in admission["decisions"]
                if d["category"] in ("positive", "honest_abstention")]

    trajectories = load_trajectories(trajectory_dir)
    variant_paths = load_variant_paths()
    split = load_split()
    applicability = load_applicability()

    buckets: Dict[str, List[dict]] = {name: [] for name in FILES}
    invalid: List[dict] = []

    for decision in admitted:
        record = trajectories.get(decision["trajectory_id"])
        if record is None:
            invalid.append({"trajectory_id": decision["trajectory_id"],
                            "problems": ["trajectory record missing"]})
            continue
        record = {**record,
                  "image_path": variant_paths.get(record["variant_id"], ""),
                  "resolution": (split or {}).get("sources", {})
                  .get(record["source_id"], {}).get("resolution")}
        split_entry = (split or {}).get("sources", {}).get(record["source_id"])
        sample = render_sample(record, decision["category"],
                               observation_text=extract_observation(record),
                               split_entry=split_entry)

        problems = validate_sample(sample, split=split, applicability=applicability)
        if problems:
            invalid.append({"id": sample["id"], "problems": problems})
            continue
        buckets[bucket_for(record, decision["category"])].append(sample)

    os.makedirs(output_dir, exist_ok=True)
    for bucket, samples in buckets.items():
        with open(os.path.join(output_dir, FILES[bucket]), "w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "sft_invalid.json"), "w", encoding="utf-8") as handle:
        json.dump(invalid, handle, ensure_ascii=False, indent=2)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "admission_report": os.path.relpath(admission_path, PROJECT_ROOT).replace(os.sep, "/"),
        "split_hash": (split or {}).get("hash"),
        "counts": {bucket: len(samples) for bucket, samples in buckets.items()},
        "invalid": len(invalid),
        "total": sum(len(samples) for samples in buckets.values()),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", default=TRAJECTORIES_DIR)
    parser.add_argument("--admission-report", default=ADMISSION_REPORT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    metadata = build(args.trajectories_dir, args.admission_report, args.output_dir)
    print(f"counts: {metadata['counts']}")
    print(f"rejected by the validator: {metadata['invalid']}")
    print(f"total: {metadata['total']}  ->  {args.output_dir}")


if __name__ == "__main__":
    main()
