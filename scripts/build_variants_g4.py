#!/usr/bin/env python3
"""
G4-b (part 1): format variants for the training sources.

A trajectory is generated per (source, treatment, tool policy), so the
treatments have to exist for the sources that will be trained on — the
calibration set only built them for its own 98 sources.  The encodings are
taken from `build_calibration_set.build_cells`, not re-implemented: the
reliability table describes *those* transforms, and a second implementation
would quietly invalidate it.

Sources come from `sft_data/split_v2.json`, so every variant inherits its
source's partition and no re-encoding can cross a split boundary.  Sampling is
class-balanced by default, using the weights the split recorded, so a run does
not silently inherit the dataset's 1:8 Real:Fake ratio.

Output: `sft_data/variants/` plus `sft_data/variants/manifest.json` mapping each
variant to its source, generator, label, partition and treatment.

Usage:
  python scripts/build_variants_g4.py --sources 100            # balanced draw
  python scripts/build_variants_g4.py --sources 100 --natural-ratio
  python scripts/build_variants_g4.py --sources 40 --partition val
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.build_calibration_set import build_cells

SPLIT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
VARIANTS_DIR = os.path.join(PROJECT_ROOT, "sft_data", "variants")
MANIFEST_PATH = os.path.join(VARIANTS_DIR, "manifest.json")


def variant_stem(source_id: str) -> str:
    """`ADM/xyz` -> `ADM_xyz`; the generator stays visible in every filename."""
    return source_id.replace("/", "_")


def select_sources(split: dict, partition: str, count: int,
                   balanced: bool = True, seed: int = 20260926) -> List[str]:
    """
    Draw sources from one partition, balanced across class by default.

    Balanced means equal Real and Fake counts; the draw is deterministic for a
    seed.  With `balanced=False` the natural ratio is kept and `count` is read
    as a Real budget, so the caller still controls the scale.
    """
    members = [identifier for identifier, entry in split["sources"].items()
               if entry["split"] == partition]
    real = sorted(i for i in members if split["sources"][i]["generator"] == "Real")
    fake = sorted(i for i in members if split["sources"][i]["generator"] != "Real")

    rng = random.Random(seed)
    rng.shuffle(real)
    rng.shuffle(fake)

    if balanced:
        per_class = max(1, count // 2)
        chosen = real[:per_class] + fake[:per_class]
    else:
        chosen = real[:count] + fake[:int(round(count * len(fake) / max(1, len(real))))]
    rng.shuffle(chosen)
    return chosen


def build_variants(source_ids: List[str], split: dict,
                   out_dir: str = VARIANTS_DIR,
                   skip_existing: bool = True,
                   root: Optional[str] = None) -> Dict[str, dict]:
    """
    Write the format cells for each source and return the variant manifest.

    A source whose variants are all present is skipped, which makes the build
    resumable without re-encoding thousands of files.
    """
    base = root or PROJECT_ROOT
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict[str, dict] = {}
    written = skipped = failed = 0

    for source_id in source_ids:
        entry = split["sources"][source_id]
        stem = variant_stem(source_id)
        expected = [os.path.join(out_dir, f"{stem}_png.png")] + [
            os.path.join(out_dir, f"{stem}_jpeg_q{q}.jpg") for q in (95, 85, 70)
        ]
        if skip_existing and all(os.path.exists(p) for p in expected):
            skipped += 1
        else:
            image = cv2.imread(os.path.join(base, entry["path"]))
            if image is None:
                failed += 1
                continue
            build_cells(image, stem, out_dir=out_dir)
            written += 1

        for path in expected:
            if not os.path.exists(path):
                continue
            treatment = ("png" if path.endswith("_png.png")
                         else f"jpeg_q{os.path.splitext(path)[0].rsplit('_q', 1)[-1]}")
            relative = os.path.relpath(path, base).replace(os.sep, "/")
            manifest[relative] = {
                "variant_id": os.path.splitext(os.path.basename(path))[0],
                "source_id": source_id,
                "label": entry["label"],
                "generator": entry["generator"],
                "split": entry["split"],
                "resolution": entry["resolution"],
                "treatment": treatment,
                "container": "png" if treatment == "png" else "jpg",
                "quality": None if treatment == "png" else int(treatment.split("_q")[1]),
            }

    print(f"sources: {len(source_ids)}  written: {written}  skipped: {skipped}  failed: {failed}")
    return manifest


def counts_by(manifest: Dict[str, dict]) -> dict:
    labels = Counter(entry["label"] for entry in manifest.values())
    treatments = Counter(entry["treatment"] for entry in manifest.values())
    generators = Counter(entry["generator"] for entry in manifest.values())
    return {
        "variants": len(manifest),
        "sources": len({entry["source_id"] for entry in manifest.values()}),
        "labels": dict(labels),
        "treatments": dict(sorted(treatments.items())),
        "generators": dict(sorted(generators.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=100,
                        help="number of sources to draw (balanced Real/Fake by default)")
    parser.add_argument("--partition", default="train")
    parser.add_argument("--natural-ratio", action="store_true",
                        help="keep the dataset's class ratio instead of balancing")
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--output-dir", default=VARIANTS_DIR)
    args = parser.parse_args()

    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)

    source_ids = select_sources(split, args.partition, args.sources,
                               balanced=not args.natural_ratio, seed=args.seed)
    manifest = build_variants(source_ids, split, out_dir=args.output_dir)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_hash": split["hash"],
        "partition": args.partition,
        "seed": args.seed,
        "balanced": not args.natural_ratio,
        "counts": counts_by(manifest),
        "variants": manifest,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"counts: {report['counts']}")
    print(f"Manifest: {os.path.join(args.output_dir, 'manifest.json')}")


if __name__ == "__main__":
    main()
