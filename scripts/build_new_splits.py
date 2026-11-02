"""
Assign dev/val/test partitions for AutoSplice q95 and GenImage q95 (plan §10.4).

Grouping rules:
  * AutoSplice: source unit is the base ID.  The authentic image, all 1-3
    forged variants, the shared mask and the caption stay in one partition.
  * GenImage: source unit is the original file stem; the q95 entry inherits the
    same partition.  Each generator is split independently.  The shared Real
    pool is split once; all eight generator groups use the same Real partition
    (group dependence is explicit, not hidden).
  * The SD15 exact-duplicate pair is fixed: keep the lexicographically first
    path, mark the other `excluded_duplicate` and never assign it to a split.

The split is deterministic (seed 20261102) and stratified by AutoSplice variant
count.  `test` is sealed for final evaluation; `val` is for post-processing /
threshold decisions; `dev` is for candidate comparison.

  python scripts/build_new_splits.py --dry-run
  python scripts/build_new_splits.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402

SEED = 20261102
AUTOSPLICE_INPUTS = os.path.join(PROJECT_ROOT, "dataset", "AutoSplice_q95",
                                 "inputs_manifest.json")
AUTOSPLICE_SPLIT = os.path.join(PROJECT_ROOT, "dataset", "AutoSplice_q95",
                                "split_manifest.json")
GENIMAGE_MANIFEST = os.path.join(PROJECT_ROOT, "dataset", "GenImage_q95",
                                 "manifest.json")
GENIMAGE_SPLIT = os.path.join(PROJECT_ROOT, "dataset", "GenImage_q95",
                              "split_manifest.json")
COMBINED_SPLIT = os.path.join(PROJECT_ROOT, "dataset", "splits_20261102.json")


def _assign(ids, seed: int) -> dict:
    """Return id -> partition with an exact 70/15/15 split."""
    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_dev = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    n_test = n - n_dev - n_val
    return ({source: "dev" for source in ids[:n_dev]} |
            {source: "val" for source in ids[n_dev:n_dev + n_val]} |
            {source: "test" for source in ids[n_dev + n_val:]})


def _counts(rows, key="split") -> dict:
    by_group = defaultdict(Counter)
    by_label = defaultdict(Counter)
    for row in rows:
        by_group[row["group"]][row.get(key, "unassigned")] += 1
        by_label[row["label"]][row.get(key, "unassigned")] += 1
    return {"by_group": {g: dict(c) for g, c in sorted(by_group.items())},
            "by_label": {l: dict(c) for l, c in sorted(by_label.items())}}


def _autosplice(dry_run: bool) -> dict:
    payload = json.load(open(AUTOSPLICE_INPUTS, encoding="utf-8"))
    rows = payload["inputs"]
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    variant_count = {s: sum(1 for r in rs if r["label"] == "Fake")
                     for s, rs in by_source.items()}
    strata = defaultdict(list)
    for source, count in variant_count.items():
        strata[count].append(source)
    mapping = {}
    for count in sorted(strata):
        mapping.update(_assign(strata[count], SEED + count))
    if set(mapping) != set(by_source):
        raise SystemExit("AutoSplice split not covering all sources")
    for row in rows:
        row["split"] = mapping[row["source_id"]]
    payload["split_seed"] = SEED
    payload["split_rules"] = {
        "unit": "base source_id",
        "authentic_and_variants_together": True,
        "mask_caption_together": True,
        "test_sealed": True,
    }
    split = {
        "schema": "autosplice_split/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "source_unit": "base source_id",
        "rules": payload["split_rules"],
        "counts": {"sources": dict(Counter(mapping.values())),
                   "sources_by_variant_count": {
                       str(k): dict(Counter(mapping[s] for s in v))
                       for k, v in sorted(strata.items())},
                   "inputs": _counts(rows)},
        "sources": {s: mapping[s] for s in sorted(mapping)},
        "note": ("Each base keeps authentic + all forged variants + mask/caption "
                 "in one partition. Test is sealed."),
    }
    if not dry_run:
        with open(AUTOSPLICE_INPUTS, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        with open(AUTOSPLICE_SPLIT, "w", encoding="utf-8") as handle:
            json.dump(split, handle, ensure_ascii=False, indent=2)
    return split


def _genimage(dry_run: bool) -> dict:
    payload = json.load(open(GENIMAGE_MANIFEST, encoding="utf-8"))
    rows = payload["inputs"]
    by_group = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)

    # Fixed duplicate removal: same source hash, lexicographically first wins.
    duplicate_of = {}
    for group, group_rows in by_group.items():
        by_hash = defaultdict(list)
        for row in group_rows:
            by_hash[row["source_sha256"]].append(row)
        for source_hash, same in by_hash.items():
            if len(same) <= 1:
                continue
            paths = sorted(r["source_path"] for r in same)
            canonical = paths[0]
            for row in same:
                if row["source_path"] != canonical:
                    duplicate_of[row["path"]] = canonical
    for row in rows:
        row["duplicate_of"] = duplicate_of.get(row["path"])
        row["split"] = ("excluded_duplicate" if row["path"] in duplicate_of
                        else "unassigned")

    mapping = {}
    group_sources = {}
    for group, group_rows in sorted(by_group.items()):
        ids = sorted({r["source_id"] for r in group_rows
                      if r["path"] not in duplicate_of})
        group_sources[group] = ids
        mapping.update(_assign(ids, SEED + sum(ord(c) for c in group)))
    for row in rows:
        if row["split"] != "excluded_duplicate":
            row["split"] = mapping[row["source_id"]]
    payload["split_seed"] = SEED
    payload["split_rules"] = {
        "unit": "original file / q95 derivative",
        "same_source_q95_together": True,
        "generator_splits_independent": True,
        "shared_real_pool": True,
        "shared_real_partition_across_generators": True,
        "duplicate_rule": "same source_sha256; lexicographically first canonical; others excluded",
        "test_sealed": True,
    }
    split = {
        "schema": "genimage_split/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "source_unit": "original file stem; q95 inherits partition",
        "rules": payload["split_rules"],
        "counts": {
            "inputs": _counts(rows),
            "sources_by_group": {
                g: dict(Counter(mapping[s] for s in ids))
                for g, ids in sorted(group_sources.items())},
        },
        "duplicate_exclusions": duplicate_of,
        "sources": {s: mapping[s] for s in sorted(mapping)},
        "note": ("Each generator is split by its own sources; Real is split once "
                 "and shared across the eight generator groups, so groups are not "
                 "independent. Test is sealed."),
    }
    if not dry_run:
        with open(GENIMAGE_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        with open(GENIMAGE_SPLIT, "w", encoding="utf-8") as handle:
            json.dump(split, handle, ensure_ascii=False, indent=2)
    return split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    auto = _autosplice(args.dry_run)
    gen = _genimage(args.dry_run)
    combined = {
        "schema": "new_splits/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "autosplice": {"split_manifest": os.path.relpath(AUTOSPLICE_SPLIT, PROJECT_ROOT),
                       "counts": auto["counts"]},
        "genimage": {"split_manifest": os.path.relpath(GENIMAGE_SPLIT, PROJECT_ROOT),
                     "counts": gen["counts"],
                     "duplicate_exclusions": gen["duplicate_exclusions"]},
        "mnw_div2k": {"split": "dataset/mnw_div2k/dev_final_split.json",
                      "note": "existing dev_current/dev_extension/final; no re-split"},
    }
    if not args.dry_run:
        with open(COMBINED_SPLIT, "w", encoding="utf-8") as handle:
            json.dump(combined, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"autosplice_sources": auto["counts"]["sources"],
                      "genimage_by_group": gen["counts"]["sources_by_group"],
                      "genimage_duplicates": gen["duplicate_exclusions"]},
                     ensure_ascii=False))
    if not args.dry_run:
        print(f"autosplice split: {os.path.relpath(AUTOSPLICE_SPLIT, PROJECT_ROOT)}")
        print(f"genimage split: {os.path.relpath(GENIMAGE_SPLIT, PROJECT_ROOT)}")
        print(f"combined: {os.path.relpath(COMBINED_SPLIT, PROJECT_ROOT)}")
    else:
        print("（dry-run）未写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
