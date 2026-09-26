#!/usr/bin/env python3
"""
G2-a: build a format-balanced, resolution-bucketed calibration set.

Phase 2 discovered a class/format confound: native Real images are JPEG and
native Fake images are PNG, so any expert that measures compression traces
can separate the two classes by container alone. This script derives
format-matched cells so expert discrimination can be attributed:

  cells (per class):
    native        as-is container (Real=JPEG, Fake=PNG) — the confounded setup
    png           lossless PNG container (Real keeps pixel-level JPEG history)
    jpeg_q95/85/70  single re-encode at the given quality

  perturbation cells (subset, PNG container):
    blur, noise, resize_0.5x, resize_2x, sharpen, brightness, screenshot

Outputs:
  calibration/set/images/*            derived images (gitignored)
  calibration/set/manifest.json       sample table committed to the repo

Usage:
  python scripts/build_calibration_set.py                 # default 50/class
  python scripts/build_calibration_set.py --num-per-class 20 --perturbation-sources 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GENIMAGE_SUBDIRS, GENIMAGE_TEST_DIR, PROJECT_ROOT, REAL_DIR
from utils.image_utils import ImageUtils

SET_DIR = os.path.join(PROJECT_ROOT, "calibration", "set")
IMAGES_DIR = os.path.join(SET_DIR, "images")
MANIFEST_PATH = os.path.join(SET_DIR, "manifest.json")

JPEG_QUALITIES = (95, 85, 70)
PERTURBATIONS = ("blur", "noise", "resize_0.5x", "resize_2x", "sharpen", "brightness", "screenshot")
RESOLUTION_BUCKETS = ((0, 256, "small"), (257, 512, "mid"), (513, 10_000, "large"))


def resolution_bucket(height: int, width: int) -> str:
    longest = max(height, width)
    for low, high, name in RESOLUTION_BUCKETS:
        if low <= longest <= high:
            return name
    return "large"


def _write_png(path: str, image: np.ndarray) -> bool:
    return bool(cv2.imwrite(path, image, [cv2.IMWRITE_PNG_COMPRESSION, 6]))


def _write_jpeg(path: str, image: np.ndarray, quality: int) -> bool:
    return bool(cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality]))


# ---------------------------------------------------------------------------
# Treatments
# ---------------------------------------------------------------------------

def build_cells(image: np.ndarray, sample_id: str,
                out_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Derive the format cells for one source image.

    `out_dir` lets other stages (G4-b builds variants for training sources)
    reuse the exact treatment encodings the calibration set was built with —
    the two must not drift, or the reliability table would describe a different
    transform than the pipeline applies.
    """
    directory = out_dir or IMAGES_DIR
    os.makedirs(directory, exist_ok=True)
    outputs = []

    png_path = os.path.join(directory, f"{sample_id}_png.png")
    if _write_png(png_path, image):
        outputs.append({"treatment": "png", "container": "png", "quality": None,
                        "path": os.path.relpath(png_path, PROJECT_ROOT).replace(os.sep, "/")})

    for quality in JPEG_QUALITIES:
        jpeg_path = os.path.join(directory, f"{sample_id}_jpeg_q{quality}.jpg")
        if _write_jpeg(jpeg_path, image, quality):
            outputs.append({"treatment": f"jpeg_q{quality}", "container": "jpg",
                            "quality": quality,
                            "path": os.path.relpath(jpeg_path, PROJECT_ROOT).replace(os.sep, "/")})

    return outputs


def build_perturbations(image: np.ndarray, sample_id: str) -> List[Dict[str, Any]]:
    """Derive post-processing perturbations (PNG container) for stability tests."""
    outputs = []
    height, width = image.shape[:2]

    def emit(name: str, perturbed: np.ndarray, container: str = "png") -> None:
        path = os.path.join(IMAGES_DIR, f"{sample_id}_{name}.{container}")
        if container == "png":
            ok = _write_png(path, perturbed)
        else:
            ok = _write_jpeg(path, perturbed, 70)
        if ok:
            outputs.append({"treatment": name, "container": container, "quality": None,
                            "path": os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")})

    emit("blur", cv2.GaussianBlur(image, (0, 0), sigmaX=1.5))
    noise = np.clip(
        image.astype(np.int16) + np.random.default_rng(0).normal(0, 8, image.shape).astype(np.int16),
        0, 255,
    ).astype(np.uint8)
    emit("noise", noise)
    emit("resize_0.5x", cv2.resize(image, (max(8, width // 2), max(8, height // 2)), interpolation=cv2.INTER_AREA))
    emit("resize_2x", cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_CUBIC))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    emit("sharpen", cv2.filter2D(image, -1, kernel))
    emit("brightness", cv2.convertScaleAbs(image, alpha=1.15, beta=18))
    # Screenshot-style re-encode: JPEG q70 then lossless PNG.
    tmp_jpeg = os.path.join(IMAGES_DIR, f"{sample_id}_screenshot_tmp.jpg")
    if _write_jpeg(tmp_jpeg, image, 70):
        reloaded = cv2.imread(tmp_jpeg, cv2.IMREAD_COLOR)
        os.remove(tmp_jpeg)
        if reloaded is not None:
            emit("screenshot", reloaded)

    return outputs


# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------

def _list_images(directory: str) -> List[str]:
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith((".jpg", ".jpeg", ".png"))
    )


def select_real_sources(count: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    paths = _list_images(REAL_DIR)
    if len(paths) < count:
        raise SystemExit(f"Real pool too small: {len(paths)} < {count}")
    chosen = sorted(rng.choice(len(paths), size=count, replace=False).tolist())
    return [
        {"label": "Real", "generator": "Real", "source_path": paths[i],
         "source_id": f"real_{index:04d}"}
        for index, i in enumerate(chosen)
    ]


def select_fake_sources(count: int, rng: np.random.Generator) -> List[Dict[str, Any]]:
    generators = list(GENIMAGE_SUBDIRS)
    per_generator = max(1, count // len(generators))
    sources: List[Dict[str, Any]] = []

    for generator in generators:
        directory = os.path.join(GENIMAGE_TEST_DIR, generator)
        if not os.path.isdir(directory):
            continue
        paths = _list_images(directory)
        take = min(per_generator, len(paths))
        chosen = sorted(rng.choice(len(paths), size=take, replace=False).tolist())
        for i in chosen:
            sources.append({
                "label": "Fake", "generator": generator,
                "source_path": paths[i],
                "source_id": f"fake_{generator.lower()}_{len(sources):04d}",
            })

    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-per-class", type=int, default=50,
                        help="Number of Real and Fake source images")
    parser.add_argument("--perturbation-sources", type=int, default=15,
                        help="Per-class sources that also get perturbation variants")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-perturbations", action="store_true")
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Selecting sources: {args.num_per_class} Real + {args.num_per_class} Fake ...")
    sources = select_real_sources(args.num_per_class, rng) + select_fake_sources(args.num_per_class, rng)
    print(f"  {len(sources)} source images selected")

    samples: List[Dict[str, Any]] = []
    failures = 0

    for index, source in enumerate(sources, 1):
        image = ImageUtils.load_image(source["source_path"])
        height, width = image.shape[:2]
        bucket = resolution_bucket(height, width)
        common = {
            "source_id": source["source_id"],
            "source_path": os.path.relpath(source["source_path"], PROJECT_ROOT).replace(os.sep, "/"),
            "label": source["label"],
            "generator": source["generator"],
            "resolution": [height, width],
            "resolution_bucket": bucket,
        }

        # native container: copy the source as-is
        native_ext = os.path.splitext(source["source_path"])[1].lower()
        native_path = os.path.join(IMAGES_DIR, f"{source['source_id']}_native{native_ext}")
        if native_ext in (".jpg", ".jpeg"):
            ok = _write_jpeg(native_path, image, 95)
        else:
            ok = _write_png(native_path, image)
        if ok:
            samples.append({
                **common, "sample_id": f"{source['source_id']}_native",
                "cell": f"{source['label'].lower()}_native",
                "treatment": "native", "container": "jpg" if native_ext in (".jpg", ".jpeg") else "png",
                "quality": None, "perturbation": None,
                "path": os.path.relpath(native_path, PROJECT_ROOT).replace(os.sep, "/"),
            })
        else:
            failures += 1

        for cell in build_cells(image, source["source_id"]):
            samples.append({
                **common, "sample_id": f"{source['source_id']}_{cell['treatment']}",
                "cell": f"{source['label'].lower()}_{cell['treatment']}",
                "treatment": cell["treatment"], "container": cell["container"],
                "quality": cell["quality"], "perturbation": None, "path": cell["path"],
            })

        if index % 20 == 0:
            print(f"  [{index}/{len(sources)}] sources processed")

    if not args.skip_perturbations:
        print(f"Building perturbations for {args.perturbation_sources} sources per class ...")
        perturbation_sources = (
            [s for s in sources if s["label"] == "Real"][:args.perturbation_sources]
            + [s for s in sources if s["label"] == "Fake"][:args.perturbation_sources]
        )
        for source in perturbation_sources:
            image = ImageUtils.load_image(source["source_path"])
            height, width = image.shape[:2]
            common = {
                "source_id": source["source_id"],
                "source_path": os.path.relpath(source["source_path"], PROJECT_ROOT).replace(os.sep, "/"),
                "label": source["label"],
                "generator": source["generator"],
                "resolution": [height, width],
                "resolution_bucket": resolution_bucket(height, width),
            }
            for cell in build_perturbations(image, source["source_id"]):
                samples.append({
                    **common, "sample_id": f"{source['source_id']}_{cell['treatment']}",
                    "cell": f"{source['label'].lower()}_{cell['treatment']}",
                    "treatment": cell["treatment"], "container": cell["container"],
                    "quality": cell["quality"], "perturbation": cell["treatment"],
                    "path": cell["path"],
                })

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "num_per_class": args.num_per_class,
        "cells": sorted({s["cell"] for s in samples}),
        "counts": {
            "samples": len(samples),
            "by_cell": {cell: sum(1 for s in samples if s["cell"] == cell)
                        for cell in sorted({s["cell"] for s in samples})},
            "by_label": {label: sum(1 for s in samples if s["label"] == label)
                         for label in ("Real", "Fake")},
            "failures": failures,
        },
        "samples": samples,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\nManifest: {MANIFEST_PATH}")
    print(f"Samples:  {len(samples)}  (failures: {failures})")
    print("Cells:")
    for cell, count in manifest["counts"]["by_cell"].items():
        print(f"  {cell:26s} {count}")


if __name__ == "__main__":
    main()
