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
from scripts.record_review import load_dispositions
from utils.batch import BUCKET_FILES, LEGACY_BATCH, batch_paths, content_sha256

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
    from utils.batch import load_samples as _load
    return _load(samples_dir=final_v2_dir)


def stale_decisions(samples: List[dict], dispositions: Dict[str, dict],
                    batch: str = LEGACY_BATCH) -> List[dict]:
    """
    Decisions that no longer describe the sample they are filed under.

    A sample regenerated under the same id — the normal case, since the id
    names the variant and the policy — must not inherit the old verdict.  The
    content digest recorded at review time is what makes that detectable; a
    digest from a different batch is stale for the same reason.
    """
    stale = []
    for sample in samples:
        entry = dispositions.get(sample["id"])
        if entry is None:
            continue
        recorded = entry.get("content_sha256")
        if recorded and recorded != content_sha256(sample):
            stale.append({"sample_id": sample["id"], "why": "内容已变（哈希不符）",
                          "reviewed_at": entry.get("recorded_at")})
        elif entry.get("batch") and entry["batch"] != batch:
            stale.append({"sample_id": sample["id"],
                          "why": f"决定来自另一个批次（{entry['batch']}，当前 {batch}）",
                          "reviewed_at": entry.get("recorded_at")})
    return stale


def class_weights(labels: collections.Counter) -> Dict[str, float]:
    """Weights that make one epoch see the classes equally."""
    if not labels:
        return {}
    largest = max(labels.values())
    return {label: round(largest / count, 4) for label, count in sorted(labels.items())}


def apply(dispositions: Dict[str, dict], final_v2_dir: str = FINAL_V2_DIR,
          output_dir: str = OUTPUT_DIR, allow_partial: bool = False,
          require_hashes: bool = False, batch: str = LEGACY_BATCH) -> dict:
    assert_not_frozen(output_dir)
    samples = load_samples(final_v2_dir)

    # G7 entry 4: a decision whose sample has changed is not a decision about
    # this sample.  It is dropped from the mapping — which makes the sample
    # undecided, so the set cannot be written without --allow-partial and the
    # reviewer is told exactly which ids to look at again.
    stale = stale_decisions(samples, dispositions, batch)
    stale_ids = {entry["sample_id"] for entry in stale}
    if stale:
        dispositions = {key: value for key, value in dispositions.items()
                        if key not in stale_ids}

    unversioned = [s["id"] for s in samples
                   if s["id"] in dispositions
                   and not dispositions[s["id"]].get("content_sha256")]
    if require_hashes and unversioned:
        raise SystemExit(
            f"{len(unversioned)} 条决定没有内容哈希（旧格式），本批次要求哈希："
            f"{unversioned[:3]}")

    undecided = [s["id"] for s in samples if s["id"] not in dispositions]
    if undecided and not allow_partial:
        raise SystemExit(
            f"{len(undecided)} samples are undecided, e.g. {undecided[:3]}; "
            f"finish the review or pass --allow-partial to write a provisional set"
            + (f"；其中 {len(stale_ids)} 条因内容变化需重审：{sorted(stale_ids)[:3]}"
               if stale else ""))

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
        "stale_decisions": stale,
        "unversioned_decisions": len(unversioned),
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
    """CLI entry: refuses to build a training set out of a frozen batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v2-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dispositions", default=None)
    parser.add_argument("--batch", default=LEGACY_BATCH)
    parser.add_argument("--require-hashes", action="store_true",
                        help="拒绝没有内容哈希的旧格式决定（新批次应当开启）")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--generation", required=True,
                        help="代次目录（如 g8）：新版批次必须显式指定，不猜默认值")
    parser.add_argument("--allow-frozen", action="store_true",
                        help="只用于分析：允许读取冻结批次，产物会标注 frozen_batch_used")
    args = parser.parse_args()

    # §4.17: the freeze is enforced here, not only written in the docs.  The
    # 237 集 (g4_trial) is frozen too, so a default run now refuses — that is
    # the intended behaviour; analysis passes --allow-frozen and is labelled.
    assert_not_frozen_batch(args.batch, "生成训练集", args.generation,
                            allow=args.allow_frozen)

    paths = batch_paths(args.batch, args.generation)
    if args.final_v2_dir is None:
        args.final_v2_dir = paths["samples"]
    if args.output_dir is None:
        args.output_dir = paths["reviewed"]
    if args.dispositions is None:
        args.dispositions = os.path.join(paths["review"], "dispositions.jsonl")
    dispositions = load_dispositions(args.dispositions)
    metadata = apply(dispositions, args.final_v2_dir, args.output_dir,
                     args.allow_partial, args.require_hashes, args.batch)
    if args.allow_frozen:
        metadata["frozen_batch_used"] = True
        with open(os.path.join(args.output_dir, "metadata.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(f"counts: {metadata['counts']}")
    if metadata["stale_decisions"]:
        print(f"STALE: {len(metadata['stale_decisions'])} 条决定因内容变化作废（需重审）："
              f"{[s['sample_id'] for s in metadata['stale_decisions']][:5]}")
    if metadata["unversioned_decisions"]:
        print(f"WARNING: {metadata['unversioned_decisions']} 条决定没有内容哈希（旧格式），"
              f"无法验证其对应内容")
    if metadata["provisional"]:
        print(f"PROVISIONAL: {metadata['undecided']} samples undecided and excluded")
    print(f"accepted labels: {metadata['accepted_labels']}")
    print(f"accepted training weights: {metadata['accepted_training_weights']}")
    print(f"-> {args.output_dir}")


if __name__ == "__main__":
    main()
