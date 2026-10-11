#!/usr/bin/env python3
"""
G7 entry 1, steps 2–3: run an external expert for real and measure it.

The order the review set for landing an external expert is: verify the
weights, run a few Real/Fake images and check the values and directions,
measure cold-start and inference time (plus memory), and only then decide
between one process per image and a resident worker.

This script does steps 2 and 3 in one reproducible command and writes what it
saw to `calibration/g7_smoke_<expert>.json`: the per-image scores and map
statistics, the timing summary, and — when CUDA is available — the peak VRAM.
The raw products themselves are written by the adapter, under its results
directory, so a later re-analysis does not need the model again.

Usage:
  python scripts/g7_smoke_external.py --dry-run                 # CPU, mock expert
  python scripts/g7_smoke_external.py --expert trufor --limit 4 # GPU/CPU, real
  python scripts/g7_smoke_external.py --expert probe_dino --limit 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "calibration")

EXPERTS = {
    "trufor": ("experts.trufor", "TruForExpert", "MockTruForExpert"),
    "probe_dino": ("experts.probe_dino", "ProbeDinoExpert", "MockProbeExpert"),
}


def pick_images(limit: int, treatment: str = "png") -> List[dict]:
    """
    A few Real and a few Fake images in one format.

    The smoke run is about plumbing and speed, not about separation, so the
    sample is deliberately tiny — the calibration run later uses the whole
    source-grouped splits.
    """
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    picked: List[dict] = []
    for label in ("Real", "Fake"):
        wanted = [s for s in manifest["samples"]
                  if s["label"] == label and s["treatment"] == treatment]
        picked.extend(wanted[:max(1, limit // 2)])
    return picked


def load_expert(name: str, dry_run: bool, device: str):
    module_name, real_name, mock_name = EXPERTS[name]
    module = __import__(module_name, fromlist=[real_name])
    if dry_run:
        return getattr(module, mock_name)(), mock_name
    return getattr(module, real_name)(device=device), real_name


def run(expert, images: List[dict], image_size: Optional[int] = None) -> dict:
    records = []
    for entry in images:
        path = os.path.join(PROJECT_ROOT, entry["path"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            records.append({"sample_id": entry["sample_id"], "error": "unreadable"})
            continue
        if image_size and max(image.shape[:2]) > image_size:
            scale = image_size / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        started = time.perf_counter()
        result = expert.analyze(image)
        elapsed = time.perf_counter() - started
        artifacts = expert.render_artifacts(image)
        records.append({
            "sample_id": entry["sample_id"],
            "label": entry["label"],
            "treatment": entry["treatment"],
            "shape": list(image.shape[:2]),
            "score": round(float(result.raw_metric), 4),
            "strength": round(float(result.strength), 4),
            "support": result.support,
            "region": result.region,
            "elapsed_s": round(elapsed, 2),
            "artifacts": {name: list(array.shape) for name, array in artifacts.items()},
            "metadata": {k: v for k, v in result.metadata.items()
                         if k in ("preprocess", "high_response_threshold",
                                  "weights_sha256", "stats", "tiles")},
        })
    return {"records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=sorted(EXPERTS), default="trufor")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--treatment", default="png")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=None,
                        help="downscale longer side to this before the expert")
    parser.add_argument("--dry-run", action="store_true",
                        help="use the Mock expert (CPU plumbing only)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if _cuda() else "cpu")
    expert, expert_name = load_expert(args.expert, args.dry_run, device)
    reason = expert.unavailable_reason()
    if reason:
        raise SystemExit(f"专家不可用：{reason}")

    provenance = expert.provenance()
    print(f"专家 {expert_name}（{args.expert}）｜ device={device}")
    print(f"权重：{provenance.describe()}")

    images = pick_images(args.limit, args.treatment)
    print(f"样本：{[i['sample_id'] for i in images]}")

    started = time.perf_counter()
    report = run(expert, images, args.image_size)
    wall = time.perf_counter() - started

    timing = expert.timing_summary() if hasattr(expert, "timing_summary") else {}
    report.update({
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expert": args.expert,
        "expert_name": expert_name,
        "dry_run": args.dry_run,
        "device": device,
        "weights": provenance.as_dict(),
        "timing": timing,
        "wall_s": round(wall, 2),
        "peak_vram_gb": _peak_vram() if device.startswith("cuda") else None,
    })

    output = args.output or os.path.join(
        OUTPUT_DIR, f"g7_smoke_{args.expert}{'_dry_run' if args.dry_run else ''}.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    print(f"\n{'样本':34s} {'标签':5s} {'score':>7s} {'耗时':>7s}  产物")
    for record in report["records"]:
        if "error" in record:
            print(f"{record['sample_id'][:34]:34s} {record.get('label','?'):5s} {record['error']}")
            continue
        print(f"{record['sample_id'][:34]:34s} {record['label']:5s} "
              f"{record['score']:7.4f} {record['elapsed_s']:6.2f}s  "
              f"{','.join(record['artifacts']) or '—'}")
    print(f"\n计时：{timing}")
    if report["peak_vram_gb"]:
        print(f"峰值显存：{report['peak_vram_gb']:.2f} GB")
    print(f"\nReport: {os.path.relpath(output, PROJECT_ROOT)}")


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _peak_vram() -> Optional[float]:
    try:
        import torch
        return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        return None


if __name__ == "__main__":
    main()
