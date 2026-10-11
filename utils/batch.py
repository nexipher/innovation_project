"""
Batch identity for the review pipeline (G7 entry 4).

A review decision is only meaningful about *specific content*.  Until now a
decision was keyed by sample id alone, so regenerating a sample under the same
id — which is the normal case, the id names the variant and the policy — would
silently inherit the old verdict.  Two things are needed to stop that:

  content_sha256(sample)   a digest of what would actually be trained
  batch_fingerprint(samples)  the generation version: every sample's digest,
                              in id order, hashed once

A batch is a directory plus that fingerprint.  The historical 237 samples stay
where they are (`sft_data/train/final_v2/`) and are addressed as the legacy
batch; new work gets its own directory, so nothing is overwritten and no old
decision is inherited by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

from config import PROJECT_ROOT

# The reviewed-by-hand set from G4 (237 samples).  Kept in place, addressed by
# name, never regenerated in-place.
LEGACY_BATCH = "g4_trial"

BUCKET_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                "sft_honest_abstention.json")

# Fields that are about the *record*, not about the content being trained:
# the review block is what a decision adds, and the bucket filename says where
# the sample was filed.  Everything else participates in the digest, because
# everything else can end up in the rendered conversation.
VOLATILE_FIELDS = ("review", "_bucket_file", "review_source")


def content_sha256(sample: dict) -> str:
    """Digest of everything that could reach training for this sample."""
    payload = {key: value for key, value in sample.items()
               if key not in VOLATILE_FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def batch_fingerprint(samples: Iterable[dict]) -> str:
    """The generation version: content digests of the whole batch, in id order."""
    rows = sorted(f"{sample.get('id')}:{content_sha256(sample)}" for sample in samples)
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest[:16]


def is_trajectory_record(record) -> bool:
    """
    A session, not a summary that happens to sit in the same directory.

    The generator now writes its report beside the trajectories, and three
    loaders read that directory by filename alone — each of them parsed the
    report as a session and crashed.  The shape is the filter.
    """
    return isinstance(record, dict) and "trajectory_id" in record


def load_trajectory_records(directory: str) -> Dict[str, dict]:
    """trajectory_id -> record, for every real session in the directory.

    Subdirectories are not descended into: an archived arm keeps its own
    directory so a regenerated arm can live beside what it replaced.
    """
    records: Dict[str, dict] = {}
    if not os.path.isdir(directory):
        return records
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        if is_trajectory_record(record):
            records[record["trajectory_id"]] = record
    return records


def batch_paths(batch_id: str = LEGACY_BATCH) -> Dict[str, str]:
    """
    Where a batch's samples, reviews and output live.

    The legacy batch is addressed by its historical paths so the existing
    records keep working; every other batch gets a directory of its own.
    """
    if batch_id == LEGACY_BATCH:
        return {
            "samples": os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2"),
            "review": os.path.join(PROJECT_ROOT, "sft_data", "review"),
            "reviewed": os.path.join(PROJECT_ROOT, "sft_data", "train",
                                     "final_v2_reviewed"),
            "trajectories": os.path.join(PROJECT_ROOT, "sft_data", "trajectories"),
            "variants_dir": os.path.join(PROJECT_ROOT, "sft_data", "variants"),
            "variants": os.path.join(PROJECT_ROOT, "sft_data", "variants",
                                     "manifest.json"),
            "admission": os.path.join(PROJECT_ROOT, "sft_data", "admission_report.json"),
        }
    # Every stage of a new batch lives under its own name, generation included:
    # variants, trajectories and the admission report are what the samples are
    # built *from*, so sharing them would leave the new batch depending on the
    # old batch's files — and overwriting them would rewrite the 237's inputs.
    root = os.path.join(PROJECT_ROOT, "sft_data", "train", "batches", batch_id)
    return {
        "samples": root,
        "review": os.path.join(PROJECT_ROOT, "sft_data", "review", batch_id),
        "reviewed": root + "_reviewed",
        "trajectories": os.path.join(PROJECT_ROOT, "sft_data",
                                     f"trajectories_{batch_id}"),
        "variants_dir": os.path.join(PROJECT_ROOT, "sft_data",
                                     f"variants_{batch_id}"),
        "variants": os.path.join(PROJECT_ROOT, "sft_data",
                                 f"variants_{batch_id}", "manifest.json"),
        "admission": os.path.join(PROJECT_ROOT, "sft_data",
                                  f"admission_report_{batch_id}.json"),
    }


def isolated_paths(batch_id: str) -> List[str]:
    """
    Every path a batch writes to — used to prove two batches cannot collide.

    The isolation claim is only worth making if it is checkable, and the place
    to check it is here rather than in a paragraph of the plan.
    """
    return list(batch_paths(batch_id).values())


def load_samples(batch_id: str = LEGACY_BATCH,
                 samples_dir: Optional[str] = None) -> List[dict]:
    """Every bucket file of a batch, with the bucket name attached."""
    directory = samples_dir or batch_paths(batch_id)["samples"]
    samples: List[dict] = []
    for name in BUCKET_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for sample in json.load(handle):
                samples.append({**sample, "_bucket_file": name})
    return samples


def write_manifest(batch_id: str, directory: str, *, samples: List[dict],
                   generation: Optional[dict] = None) -> dict:
    """
    The batch's own record: what it contains, and the version it is.

    Written next to the samples so a later reader can tell whether the set on
    disk is the one the reviews were recorded against, without recomputing
    anything by hand.
    """
    manifest = {
        "batch_id": batch_id,
        "fingerprint": batch_fingerprint(samples),
        "samples": len(samples),
        "counts": _counts(samples),
        "generation": generation or {},
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def _counts(samples: List[dict]) -> dict:
    counts: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        bucket = sample.get("_bucket_file", "unfiled")
        counts.setdefault(bucket, {})
        label = sample.get("ground_truth", "?")
        counts[bucket][label] = counts[bucket].get(label, 0) + 1
    return counts
