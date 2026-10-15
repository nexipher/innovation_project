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
import collections
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.batch import LEGACY_BATCH, assert_not_frozen_batch, batch_paths
from scripts.generate_trajectories_g4 import load_applicability
from utils.final_v2 import (
    RETIRED_EXPERTS,
    SCHEMA_VERSION,
    audit_dialogue,
    conclusion_gate,
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


# The planning line that carries the model's own visual account.  The v3 batch
# wrote "Visual Anomalies:"; the next protocol asks for "Visual Observation:"
# (what the model sees) and then "Suspected Anomalies:" (what, if anything,
# looks off).  Both are read here, and neither is rewritten: the observation
# section carries the model's words, never a description the converter made up.
_PLANNING_ANOMALIES = re.compile(r"Visual (?:Observation|Anomalies):\s*(.+)", re.I)
# The no-tool training protocol asks for an explicit section; the tool protocols
# describe the same thing in their planning line.  Either way the model's own
# visual description is what the observation section is for.
_OBSERVATION_BLOCK = re.compile(r"<observation>\s*(.*?)\s*</observation>", re.S | re.I)


def load_trace(record: dict) -> Optional[dict]:
    """The session this trajectory came from, for reconstructing the turns."""
    path = record.get("trace_path")
    if not path:
        return None
    try:
        with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


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
        value = turn.get("value", "")
        block = _OBSERVATION_BLOCK.search(value)
        if block and block.group(1).strip():
            return f"模型视觉观察：{block.group(1).strip()}"
        match = _PLANNING_ANOMALIES.search(value)
        if match:
            return f"模型视觉观察：{match.group(1).strip()}"
        break
    return ""


def load_trajectories(directory: str) -> Dict[str, dict]:
    """Trajectory records by id (the directory also holds summaries)."""
    from utils.batch import load_trajectory_records

    return load_trajectory_records(directory)


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
          output_dir: str = OUTPUT_DIR,
          batch_id: str = LEGACY_BATCH,
          variants_manifest: Optional[str] = None,
          allow_frozen: bool = False) -> dict:
    assert_not_frozen(output_dir)

    with open(admission_path, encoding="utf-8") as handle:
        admission = json.load(handle)
    admitted = [d for d in admission["decisions"]
                if d["category"] in ("positive", "honest_abstention")]

    trajectories = load_trajectories(trajectory_dir)
    # The batch's own variants manifest: reading the public one left every
    # sample's image_path empty, and the review workstation then showed
    # "（未提供）" for the image the model actually saw — the paths were never
    # there to begin with, because the variant ids belong to this batch's
    # manifest and not to the 237 set's.
    variant_paths = load_variant_paths(
        variants_manifest or batch_paths(batch_id)["variants"])
    split = load_split()
    applicability = load_applicability()

    buckets: Dict[str, List[dict]] = {name: [] for name in FILES}
    invalid: List[dict] = []
    gated: List[dict] = []
    gate_reasons: collections.Counter = collections.Counter()
    legacy_without_check = 0

    for decision in admitted:
        record = trajectories.get(decision["trajectory_id"])
        if record is None:
            invalid.append({"trajectory_id": decision["trajectory_id"],
                            "problems": ["trajectory record missing"]})
            continue
        # §4.17 R4: the hard gate.  `failed` must not become a training sample,
        # and the same refusal applies to an adopted conclusion that does not
        # parse or that shows a call the controller never executed.  The
        # original trajectory record is left untouched; only the rendering is
        # refused, and the reason is written down.
        reason = conclusion_gate(record)
        if reason:
            gated.append({"trajectory_id": record["trajectory_id"],
                          "category": decision["category"], "reason": reason})
            gate_reasons[reason] += 1
            continue
        # §4.17 R6: the whole supervised dialogue, not just the conclusion.
        # Every assistant turn before the one the answer replaces survives
        # into the sample, so a fabrication in one of them is supervised — it
        # is refused here (and the trajectory kept) rather than rendered away.
        trace = load_trace(record)
        dialogue = audit_dialogue({**record, "_trace": trace})
        if dialogue["defects"]:
            gated.append({
                "trajectory_id": record["trajectory_id"],
                "category": decision["category"],
                "reason": "supervised_turn_fabrication",
                "turns": [finding["turn"] for finding in dialogue["defects"]],
                "kinds": sorted({finding["kind"] for finding in dialogue["defects"]}),
            })
            gate_reasons["supervised_turn_fabrication"] += 1
            continue
        if "evidence_check" not in record:
            legacy_without_check += 1
        record = {**record,
                  "image_path": variant_paths.get(record["variant_id"], ""),
                  "resolution": (split or {}).get("sources", {})
                  .get(record["source_id"], {}).get("resolution")}
        split_entry = (split or {}).get("sources", {}).get(record["source_id"])
        sample = render_sample(record, decision["category"],
                               observation_text=extract_observation(record),
                               split_entry=split_entry,
                               trace=trace)

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
    # Refused by the evidence gate: kept beside the samples so a reviewer can
    # see what was removed and why, without reading the trajectory again.
    with open(os.path.join(output_dir, "sft_gated.json"), "w", encoding="utf-8") as handle:
        json.dump(gated, handle, ensure_ascii=False, indent=2)

    # The batch's own version stamp: every sample's content digest, hashed once.
    # A review recorded against one fingerprint must not be applied to another,
    # and this is the value `record_review` stores with each decision.
    from utils.batch import write_manifest

    all_samples = [{**sample, "_bucket_file": name}
                   for name, samples in buckets.items() for sample in samples]
    write_manifest(batch_id, output_dir, samples=all_samples,
                   generation={"trajectories_dir": trajectory_dir,
                               "admission_report": admission_path,
                               "source": "g7 entry 4, step 2"})

    labels = collections.Counter(
        sample["ground_truth"] for samples in buckets.values() for sample in samples)
    # Training weights to sample the buckets class-balanced: evidence helps
    # mostly where the no-tool baseline is confidently wrong, which in this
    # dataset means Fake, so the raw composition is Fake-heavy (75/25) and
    # would otherwise teach a "tools mean Fake" prior.
    # Every admitted trajectory can be refused by the evidence gate, in which
    # case there is no label composition to balance (this used to raise on an
    # empty Counter instead of writing the empty, recorded result).
    weights = {}
    if labels:
        weights = {
            label: round(labels[max(labels, key=labels.get)]
                         / max(1, labels.get(label, 0)), 4)
            for label in ("Real", "Fake")
        }
    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "schema_version": SCHEMA_VERSION,
        "label_composition": dict(labels),
        "training_weights": weights,
        "per_bucket_labels": {
            bucket: dict(collections.Counter(s["ground_truth"] for s in samples))
            for bucket, samples in buckets.items()
        },
        "admission_report": os.path.relpath(admission_path, PROJECT_ROOT).replace(os.sep, "/"),
        "split_hash": (split or {}).get("hash"),
        "counts": {bucket: len(samples) for bucket, samples in buckets.items()},
        "invalid": len(invalid),
        "frozen_batch_used": bool(allow_frozen),
        "gated": len(gated),
        "gate_reasons": dict(gate_reasons),
        "fidelity": {
            "model_reasoning_verbatim": sum(
                1 for samples in buckets.values() for sample in samples
                if sample["fidelity"]["model_reasoning_verbatim"]),
            "bookkeeping_only": sum(
                1 for samples in buckets.values() for sample in samples
                if not sample["fidelity"]["model_reasoning_verbatim"]),
            "dialogue_candidates": sum(
                sample["fidelity"]["dialogue_audit"]["candidate_count"]
                for samples in buckets.values() for sample in samples),
        },
        "legacy_without_evidence_check": legacy_without_check,
        "total": sum(len(samples) for samples in buckets.values()),
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", default=None)
    parser.add_argument("--admission-report", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch", default=LEGACY_BATCH)
    parser.add_argument("--generation", required=True,
                        help="代次目录（如 g8）：新版批次必须显式指定，不猜默认值")
    parser.add_argument("--allow-frozen", action="store_true",
                        help="只用于分析：允许读取冻结批次，产物会标注 frozen_batch_used")
    args = parser.parse_args()
    # §4.17: the freeze is enforced here, not only written in the docs
    assert_not_frozen_batch(args.batch, "转换/出集", args.generation,
                            allow=args.allow_frozen)
    paths = batch_paths(args.batch, args.generation)
    if args.trajectories_dir is None:
        args.trajectories_dir = paths["trajectories"]
    if args.admission_report is None:
        args.admission_report = paths["admission"]
    if args.output_dir is None:
        args.output_dir = paths["samples"]

    metadata = build(args.trajectories_dir, args.admission_report, args.output_dir,
                     args.batch, allow_frozen=args.allow_frozen)
    print(f"counts: {metadata['counts']}")
    print(f"rejected by the validator: {metadata['invalid']}")
    print(f"total: {metadata['total']}  ->  {args.output_dir}")


if __name__ == "__main__":
    main()
