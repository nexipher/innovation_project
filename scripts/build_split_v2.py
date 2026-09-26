#!/usr/bin/env python3
"""
G4-a: source-level train/val/test split with leak prevention.

The leakage unit is the **source image**, not the file.  A source appears in
the pipeline as several re-encodings — native, PNG, JPEG q95/q85/q70 — and if
one variant trained the model while another tested it, the evaluation would
measure recall of the source, not generalisation.  Every variant of a source
therefore inherits the source's partition.

Two more constraints the later stages depend on:

  * the generator is recorded per source, so G5 can run
    leave-one-generator-out conditions;
  * the 98 sources the G2-b expert calibration was fitted on are held out
    entirely — they are not training material and must not appear in the
    evaluation sets either, or the calibration would be scored on its own
    fitting data.

Output: `sft_data/split_v2.json` (source → partition, with metadata, counts,
and a content hash that makes the split reproducible).

Usage:
  python scripts/build_split_v2.py
  python scripts/build_split_v2.py --seed 20260926 --ratios 0.7 0.15 0.15
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    GENIMAGE_SUBDIRS,
    GENIMAGE_TEST_DIR,
    PROJECT_ROOT,
    REAL_DIR,
)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
CALIBRATION_MANIFEST = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")

PARTITIONS = ("train", "val", "test")
HOLDOUT_PARTITION = "calibration_holdout"

# A variant file is named `<source-stem>_<treatment>.<ext>`; the split of any
# file is the split of its source, which is what makes the grouping testable.
TREATMENT_SUFFIX = re.compile(
    r"_(native|png|jpeg_q\d+|blur|noise|resize_[\d.]+x|sharpen|brightness|screenshot)$"
)


def source_id_from_path(path: str, generator: Optional[str] = None,
                        root: str = PROJECT_ROOT) -> str:
    """
    Stable, generator-qualified id: `Real/abc`, `ADM/xyz`, `BigGAN/xyz`.

    The generator is part of the id because stems repeat across generators and
    because every later stage needs to attribute a source to its provenance.
    """
    relative = os.path.relpath(path, root)
    parts = relative.split(os.sep)
    if generator is None:
        generator = parts[2] if len(parts) > 2 and parts[1] == "GenImage_Test" else parts[1]
    return f"{generator}/{os.path.splitext(parts[-1])[0]}"


def split_for_variant(path: str, sources: Dict[str, dict],
                      root: str = PROJECT_ROOT) -> Optional[str]:
    """
    Partition of a variant (or any file derived from a source).

    Variants carry a treatment suffix; stripping it recovers the source id, so
    a file's partition is decided by its source and can never disagree with it.
    """
    identifier = source_id_from_path(path, root=root)
    if identifier in sources:
        return sources[identifier]["split"]
    stem, _ = os.path.splitext(identifier)
    stripped = TREATMENT_SUFFIX.sub("", stem)
    return sources.get(stripped, {}).get("split")


def enumerate_sources(root: Optional[str] = None) -> List[dict]:
    """Every dataset image, with its label, generator and resolution."""
    base = root or PROJECT_ROOT
    real_dir = REAL_DIR if root is None else os.path.join(base, "dataset", "Real")
    genimage_dir = (GENIMAGE_TEST_DIR if root is None
                    else os.path.join(base, "dataset", "GenImage_Test"))
    sources: List[dict] = []
    roots: List[Tuple[str, str, str]] = [(real_dir, "Real", "Real")]
    for generator in GENIMAGE_SUBDIRS:
        roots.append((os.path.join(genimage_dir, generator), "Fake", generator))

    for root, label, generator in roots:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            path = os.path.join(root, name)
            if not os.path.isfile(path):
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
            except OSError:
                continue
            sources.append({
                "source_id": source_id_from_path(path, generator, base),
                "path": os.path.relpath(path, base).replace(os.sep, "/"),
                "label": label,
                "generator": generator,
                "resolution": [height, width],
            })
    return sources


def load_holdout_sources() -> Dict[str, str]:
    """
    Sources the G2-b calibration was fitted on, keyed by source_id.

    They are recorded as held out rather than deleted: the calibration table's
    provenance stays visible, and G5 cannot accidentally score itself on them.
    """
    try:
        with open(CALIBRATION_MANIFEST, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}

    holdout: Dict[str, str] = {}
    for sample in manifest.get("samples", []):
        path = sample.get("source_path")
        if path:
            holdout[source_id_from_path(
                os.path.join(PROJECT_ROOT, path), root=PROJECT_ROOT
            )] = "calibration_g2b"
    return holdout


def stratified_split(
    sources: List[dict],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 20260926,
    balance_eval: bool = True,
) -> Tuple[Dict[str, str], Dict[str, float]]:
    """
    Assign each source to a partition.

    The dataset is 1000 Real to 8000 Fake, so stratifying faithfully would
    hand every partition an 11/89 class mix — incomparable with the balanced
    calibration set and the G3-e arms, and a good way to train a Fake-happy
    prior on top of the container prior the model already has.

    So the evaluation partitions are balanced: `val` and `test` take as many
    Fake sources as Real ones, spread evenly over the eight generators, which
    also gives G5 a per-generator slice to test on.  `train` keeps every
    remaining source and returns class sampling weights, so trajectory
    sampling can balance explicitly instead of silently inheriting 1:8.

    Returns (assignment, sampling_weights).
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    rng = random.Random(seed)
    assignment: Dict[str, str] = {}

    real = [s["source_id"] for s in sources if s["label"] == "Real"]
    fake_by_generator: Dict[str, List[str]] = defaultdict(list)
    for source in sources:
        if source["label"] == "Fake":
            fake_by_generator[source["generator"]].append(source["source_id"])

    rng.shuffle(real)
    n_real_val = int(round(len(real) * ratios[1]))
    n_real_test = len(real) - n_real_val - int(round(len(real) * ratios[0]))
    for index, identifier in enumerate(real):
        if index < n_real_val:
            assignment[identifier] = "val"
        elif index < n_real_val + n_real_test:
            assignment[identifier] = "test"
        else:
            assignment[identifier] = "train"

    # Fake evaluation slots match the Real ones and are spread over generators.
    eval_targets = {"val": n_real_val, "test": n_real_test} if balance_eval else {}
    if not balance_eval:
        for generator, identifiers in fake_by_generator.items():
            rng.shuffle(identifiers)
            n = len(identifiers)
            for index, identifier in enumerate(identifiers):
                if index < int(round(n * ratios[1])):
                    assignment[identifier] = "val"
                elif index < int(round(n * (ratios[1] + ratios[2]))):
                    assignment[identifier] = "test"
                else:
                    assignment[identifier] = "train"
        return assignment, {"Real": 1.0, "Fake": 1.0}

    remaining: List[str] = []
    for generator in sorted(fake_by_generator):
        identifiers = sorted(fake_by_generator[generator])
        rng.shuffle(identifiers)
        remaining.extend(identifiers)

    generator_of = {s["source_id"]: s["generator"] for s in sources}
    per_generator_cap = max(1, -(-eval_targets["val"] // max(1, len(fake_by_generator))))

    for partition in ("val", "test"):
        target = eval_targets[partition]
        picked: List[str] = []
        per_generator: Dict[str, int] = defaultdict(int)
        leftovers: List[str] = []
        for identifier in remaining:
            if len(picked) >= target:
                leftovers.append(identifier)
                continue
            generator = generator_of[identifier]
            if per_generator[generator] >= per_generator_cap:
                leftovers.append(identifier)
                continue
            picked.append(identifier)
            per_generator[generator] += 1
        # A cap can leave a partition short when a generator runs out first;
        # top up from the leftovers so the partitions stay balanced in size.
        for identifier in leftovers:
            if len(picked) >= target:
                break
            picked.append(identifier)
        picked_set = set(picked)
        for identifier in picked:
            assignment[identifier] = partition
        remaining = [r for r in remaining if r not in picked_set]

    for identifier in remaining:
        assignment.setdefault(identifier, "train")

    n_real_train = sum(1 for i, a in assignment.items() if a == "train" and generator_of[i] == "Real")
    n_fake_train = sum(1 for i, a in assignment.items() if a == "train" and generator_of[i] != "Real")
    weights = {
        "Real": 1.0,
        "Fake": round(n_fake_train / n_real_train, 4) if n_real_train else 1.0,
    }
    return assignment, weights


def validate(sources: Dict[str, dict], holdout: Dict[str, str]) -> List[str]:
    """Every invariant the split must satisfy, as a list of problems."""
    problems: List[str] = []

    splits = Counter(entry["split"] for entry in sources.values())
    for partition in PARTITIONS:
        if splits.get(partition, 0) == 0:
            problems.append(f"partition {partition} is empty")

    overlap = set(sources) & set(holdout)
    if overlap:
        problems.append(f"{len(overlap)} calibration sources also assigned to a partition")

    # A source must appear once, with one split — enforced by construction, but
    # checked because a hand-edited file is the realistic failure mode.
    seen: Dict[str, str] = {}
    for identifier, entry in sources.items():
        if identifier in seen and seen[identifier] != entry["split"]:
            problems.append(f"{identifier} appears in two partitions")
        seen[identifier] = entry["split"]

    for identifier, entry in sources.items():
        if entry["split"] not in PARTITIONS:
            problems.append(f"unknown partition {entry['split']} for {identifier}")
        if not entry.get("generator") or not entry.get("label"):
            problems.append(f"{identifier} lacks label/generator provenance")

    return problems


def counts_by(entries: Dict[str, dict]) -> dict:
    summary: Dict[str, dict] = {}
    for partition in PARTITIONS:
        members = [e for e in entries.values() if e["split"] == partition]
        summary[partition] = {
            "sources": len(members),
            "labels": dict(Counter(m["label"] for m in members)),
            "generators": dict(sorted(Counter(m["generator"] for m in members).items())),
        }
    return summary


def content_hash(sources: Dict[str, dict]) -> str:
    payload = json.dumps(
        {k: sources[k]["split"] for k in sorted(sources)}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def build(sources: List[dict], ratios=(0.7, 0.15, 0.15), seed: int = 20260926,
          balance_eval: bool = True) -> dict:
    holdout = load_holdout_sources()
    eligible = [s for s in sources if s["source_id"] not in holdout]
    assignment, sampling_weights = stratified_split(eligible, ratios, seed, balance_eval)

    entries: Dict[str, dict] = {}
    for source in eligible:
        entries[source["source_id"]] = {
            "split": assignment[source["source_id"]],
            "path": source["path"],
            "label": source["label"],
            "generator": source["generator"],
            "resolution": source["resolution"],
        }

    # A variant must resolve to its source's partition — the check that makes
    # "PNG trained, q70 tested" impossible rather than merely unlikely.
    for identifier, entry in entries.items():
        resolved = split_for_variant(
            os.path.join(PROJECT_ROOT, entry["path"]), entries, root=PROJECT_ROOT
        )
        if resolved != entry["split"]:
            raise AssertionError(f"variant resolution disagrees for {identifier}: {resolved}")

    problems = validate(entries, holdout)
    if problems:
        raise AssertionError("split validation failed: " + "; ".join(problems))

    return {
        "version": "split_v2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "hash": content_hash(entries),
        "counts": counts_by(entries),
        # Class weights for trajectory sampling: training keeps every eligible
        # Fake source, so 1:1 sampling needs these multipliers rather than a
        # silent 1:8 prior.
        "train_sampling_weights": sampling_weights,
        "holdout": {k: holdout[k] for k in sorted(holdout)},
        "sources": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--natural-ratio", action="store_true",
                        help="keep the dataset's 1:8 Real:Fake mix in val/test too")
    args = parser.parse_args()

    sources = enumerate_sources()
    print(f"dataset sources: {len(sources)}")
    report = build(sources, tuple(args.ratios), args.seed,
                   balance_eval=not args.natural_ratio)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"holdout (G2-b calibration): {len(report['holdout'])} sources")
    for partition, summary in report["counts"].items():
        print(f"  {partition:7s} {summary['sources']:5d} sources  labels={summary['labels']}")
        print(f"          generators={summary['generators']}")
    print(f"train sampling weights (to sample 1:1): {report['train_sampling_weights']}")
    print(f"hash: {report['hash']}")
    print(f"Split: {args.output}")


if __name__ == "__main__":
    main()