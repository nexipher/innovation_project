#!/usr/bin/env python3
"""
G7 entry 1, supplement: group D — a fresh set for one TruFor re-check.

Why this exists: the fitting group produced one unexplained result — the q85
cell's AUROC was 0.342 with a source-grouped CI of (0.2094, 0.4823), entirely
below 0.5, while the other four formats straddled it.  One observation is not
a finding, and the plan forbids flipping the sign after the fact and calling
it a gain.  The way to settle it is to measure again on images nothing has
ever seen.

Where the images come from: `split_v2`'s **val** partition.  Val is disjoint
from the 98 calibration sources (they are `split_v2`'s holdout), from A/B/C,
and from the test partition, and — being a validation rather than a training
partition — these files can never enter the SFT training set later.

Treatments are the calibration set's own encodings (`build_cells`), so the
numbers land in the same units as the fitting group's: 64 sources x 5 formats
= 320 images, n=64 per format against the fitting group's 62.

Usage:
  python scripts/build_g7_group_d.py --dry-run     # selection only, no files written
  python scripts/build_g7_group_d.py               # write images + manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.build_calibration_set import build_cells, resolution_bucket
from utils.image_utils import ImageUtils

SET_DIR = os.path.join(PROJECT_ROOT, "calibration", "set")
IMAGES_DIR = os.path.join(SET_DIR, "images_d")
MANIFEST_PATH = os.path.join(SET_DIR, "manifest_d.json")
SPLIT_PATH = os.path.join(SET_DIR, "g7_split_d.json")
SPLIT_V2_PATH = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
CALIBRATION_MANIFEST = os.path.join(SET_DIR, "manifest.json")
GROUP = "D"
DEFAULT_SEED = 20261011
SOURCES_PER_CLASS = 32
PER_GENERATOR = 4


def load_val_sources() -> Dict[str, dict]:
    """`split_v2`'s validation sources, keyed by their split key."""
    with open(SPLIT_V2_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    return {key: entry for key, entry in split["sources"].items()
            if entry["split"] == "val"}


def previous_paths() -> set:
    """
    Every image this project has already measured or trained on.

    The calibration manifest covers A/B/C (its 98 sources are `split_v2`'s
    holdout); the variant manifest covers the sources the G4 trial built
    training material from.  A fresh group must touch none of them.
    """
    used = set()
    with open(CALIBRATION_MANIFEST, encoding="utf-8") as handle:
        used |= {s["source_path"] for s in json.load(handle)["samples"]}
    # The 237 set's variant manifest was archived on 2026-10-15.  Skipping the
    # exclusion silently would let a "fresh" group reuse sources that are not
    # fresh at all, so the mapping is followed and a miss is reported.
    from utils.batch import resolve_path

    variants = resolve_path(os.path.join(PROJECT_ROOT, "sft_data", "variants",
                                         "manifest.json"))[0]
    if os.path.exists(variants):
        with open(variants, encoding="utf-8") as handle:
            blob = json.load(handle)
        for entry in blob.get("variants", []):
            for key in ("source_path", "path"):
                if isinstance(entry, dict) and entry.get(key):
                    used.add(entry[key])
            source = entry.get("source") if isinstance(entry, dict) else None
            if isinstance(source, dict) and source.get("path"):
                used.add(source["path"])
    return used


def select(val_sources: Dict[str, dict], seed: int) -> List[dict]:
    """32 Real + 4 per generator (32 Fake), seeded and reported."""
    rng = np.random.default_rng(seed)
    real = sorted((key, entry) for key, entry in val_sources.items()
                  if entry["label"] == "Real")
    chosen: List[dict] = []

    picked = sorted(rng.choice(len(real), size=SOURCES_PER_CLASS, replace=False).tolist())
    for index, position in enumerate(picked):
        key, entry = real[position]
        chosen.append({"key": key, "label": "Real", "generator": "Real",
                       "source_path": entry["path"], "source_id": f"d_real_{index:04d}"})

    generators = sorted({entry["generator"] for entry in val_sources.values()
                         if entry["label"] == "Fake"})
    for generator in generators:
        pool = sorted((key, entry) for key, entry in val_sources.items()
                      if entry["label"] == "Fake" and entry["generator"] == generator)
        take = min(PER_GENERATOR, len(pool))
        positions = sorted(rng.choice(len(pool), size=take, replace=False).tolist())
        for position in positions:
            key, entry = pool[position]
            index = sum(1 for s in chosen if s["generator"] == generator)
            chosen.append({"key": key, "label": "Fake", "generator": generator,
                           "source_path": entry["path"],
                           "source_id": f"d_fake_{generator.lower()}_{index:04d}"})
    return chosen


def check_disjoint(chosen: List[dict], val_sources: Dict[str, dict]) -> List[str]:
    problems = []
    used = previous_paths()
    for source in chosen:
        if source["key"] not in val_sources:
            problems.append(f"{source['source_id']} 不在 val 分区")
        if source["source_path"] in used:
            problems.append(f"{source['source_id']} 与已用过的图重叠：{source['source_path']}")
    paths = [s["source_path"] for s in chosen]
    if len(set(paths)) != len(paths):
        problems.append("选中集合内有重复源图")
    return problems


def fingerprint(source_ids: List[str]) -> str:
    canonical = json.dumps(sorted(source_ids), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build(chosen: List[dict], dry_run: bool = False) -> dict:
    samples: List[Dict[str, Any]] = []
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for source in chosen:
        image = ImageUtils.load_image(source["source_path"])
        if image is None:
            raise SystemExit(f"读不到源图：{source['source_path']}")
        height, width = image.shape[:2]
        common = {
            "source_id": source["source_id"],
            "source_path": os.path.relpath(source["source_path"], PROJECT_ROOT).replace(os.sep, "/"),
            "label": source["label"], "generator": source["generator"],
            "resolution": [height, width], "resolution_bucket": resolution_bucket(height, width),
            "group": GROUP,
        }
        if dry_run:
            continue

        # native: the source's own container, re-encoded the calibration way
        extension = os.path.splitext(source["source_path"])[1].lower()
        native_path = os.path.join(IMAGES_DIR, f"{source['source_id']}_native{extension}")
        import cv2
        if extension in (".jpg", ".jpeg"):
            ok = bool(cv2.imwrite(native_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95]))
            container = "jpg"
        else:
            ok = bool(cv2.imwrite(native_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 6]))
            container = "png"
        if ok:
            samples.append({**common, "sample_id": f"{source['source_id']}_native",
                            "cell": f"{source['label'].lower()}_native",
                            "treatment": "native", "container": container,
                            "quality": None, "perturbation": None,
                            "path": os.path.relpath(native_path, PROJECT_ROOT).replace(os.sep, "/")})

        for cell in build_cells(image, source["source_id"], out_dir=IMAGES_DIR):
            samples.append({**common, "sample_id": f"{source['source_id']}_{cell['treatment']}",
                            "cell": f"{source['label'].lower()}_{cell['treatment']}",
                            "treatment": cell["treatment"], "container": cell["container"],
                            "quality": cell["quality"], "perturbation": None, "path": cell["path"]})

    if dry_run:
        return {"samples": [], "dry_run": True}

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "group": GROUP,
        "seed": DEFAULT_SEED,
        "partition": "val（split_v2）",
        "sources_per_class": SOURCES_PER_CLASS,
        "cells": sorted({s["cell"] for s in samples}),
        "counts": {
            "samples": len(samples),
            "sources": len({s["source_id"] for s in samples}),
            "by_label": {label: sum(1 for s in samples if s["label"] == label)
                         for label in ("Real", "Fake")},
        },
        "note": ("TruFor 复测专用（G7 入口 1 补充）。D 组不参与任何拟合、选择或准入决策；"
                 "它只用于检验 A 组 q85 格的异常是否可复现。"),
        "samples": samples,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    source_ids = sorted({s["source_id"] for s in samples})
    split = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": DEFAULT_SEED,
        "hash": fingerprint(source_ids),
        "counts": {GROUP: len(source_ids)}, "samples": {GROUP: len(samples)},
        "groups": {GROUP: source_ids},
        "note": ("D 组来自 split_v2 的 val 分区，与 98 个校准源（holdout）、A/B/C 三组、"
                 "最终 test 分区互斥；仅用于 TruFor 独立复测，不参与任何决策"),
    }
    with open(SPLIT_PATH, "w", encoding="utf-8") as handle:
        json.dump(split, handle, ensure_ascii=False, indent=1)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    val_sources = load_val_sources()
    print(f"val 分区源图：{len(val_sources)}（Real "
          f"{sum(1 for e in val_sources.values() if e['label'] == 'Real')} / Fake "
          f"{sum(1 for e in val_sources.values() if e['label'] == 'Fake')}）")
    chosen = select(val_sources, args.seed)
    problems = check_disjoint(chosen, val_sources)
    if problems:
        raise SystemExit("D 组选择不合法：" + "；".join(problems))
    print(f"选中 {len(chosen)} 源（Real {sum(1 for s in chosen if s['label'] == 'Real')} / "
          f"Fake {sum(1 for s in chosen if s['label'] == 'Fake')}）"
          f"，hash {fingerprint(sorted(s['source_id'] for s in chosen))}")

    manifest = build(chosen, dry_run=args.dry_run)
    if args.dry_run:
        for source in chosen[:5]:
            print(f"  {source['source_id']:22s} {source['label']:4s} "
                  f"{source['generator']:12s} {source['source_path']}")
        print("  …（dry-run，未写任何文件）")
        return
    print(f"\n清单：{os.path.relpath(MANIFEST_PATH, PROJECT_ROOT)}"
          f"（{manifest['counts']['samples']} 样本 / {manifest['counts']['sources']} 源）")
    print(f"划分：{os.path.relpath(SPLIT_PATH, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
