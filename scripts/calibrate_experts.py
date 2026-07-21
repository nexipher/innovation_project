#!/usr/bin/env python3
"""
Expert calibration script — Phase 2.2.

Runs all three forensic experts on a benchmark set (Real + GenImage images)
using FULL-IMAGE analysis (not bbox crop), collects raw_metric distributions,
and grid-searches optimal sigmoid parameters and strength thresholds.

Output:
  - calibration/calibration_report.json   — full results with recommended params
  - Printed summary with suggested config.py updates.

Usage:
  python scripts/calibrate_experts.py --num-real 100 --num-fake 30
  python scripts/calibrate_experts.py --quick   # 20 real + 10 fake, fast test
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REAL_DIR, GENIMAGE_TEST_DIR, GENIMAGE_SUBDIRS,
    STRENGTH_THRESHOLD_LOW, STRENGTH_THRESHOLD_HIGH,
    FREQ_SIGMOID_MIDPOINT, FREQ_SIGMOID_STEEPNESS,
    NOISE_SIGMOID_MIDPOINT, NOISE_SIGMOID_STEEPNESS,
    JPEG_SIGMOID_MIDPOINT, JPEG_SIGMOID_STEEPNESS,
)
from utils.image_utils import ImageUtils
from experts.frequency import FrequencyExpert
from experts.noise import NoiseExpert
from experts.jpeg import JPEGExpert


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def load_image_paths(real_dir: str, gen_dir: str, subdirs: List[str],
                     num_real: int, num_fake_per_class: int) -> Dict[str, List[str]]:
    """Collect image paths for Real + each GenImage subdir."""
    paths = {"Real": [], "Fake": []}

    # Real images
    real_files = sorted([
        f for f in os.listdir(real_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:num_real]
    paths["Real"] = [os.path.join(real_dir, f) for f in real_files]

    # Fake images — sample equally from each generator
    for sub in subdirs:
        sub_path = os.path.join(gen_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        files = sorted([
            f for f in os.listdir(sub_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:num_fake_per_class]
        for f in files:
            paths["Fake"].append(os.path.join(sub_path, f))

    return paths


def run_expert_on_image(expert, img: np.ndarray) -> float:
    """Run an expert on the FULL image and return raw_metric."""
    result = expert.analyze(img)
    return result.raw_metric


def collect_metrics(image_paths: Dict[str, List[str]]) -> Dict[str, dict]:
    """
    Run all three experts on all images.
    Returns: {"freq": {"Real": [...], "Fake": [...]}, "noise": {...}, "jpeg": {...}}
    """
    experts = {
        "freq": FrequencyExpert(),
        "noise": NoiseExpert(),
        "jpeg": JPEGExpert(),
    }

    metrics = {name: {"Real": [], "Fake": []} for name in experts}

    total = sum(len(v) for v in image_paths.values())
    count = 0

    for label, paths in image_paths.items():
        for path in paths:
            count += 1
            try:
                img = ImageUtils.load_image(path)
            except Exception as e:
                print(f"  SKIP {path}: {e}")
                continue

            for name, expert in experts.items():
                raw = run_expert_on_image(expert, img)
                metrics[name][label].append(raw)

            if count % 20 == 0:
                print(f"  [{count}/{total}] processed...")

    return metrics


# ---------------------------------------------------------------------------
# Sigmoid calibration
# ---------------------------------------------------------------------------

def sigmoid(x: float, midpoint: float, steepness: float) -> float:
    return 1.0 / (1.0 + np.exp(-steepness * (x - midpoint)))


def youden_index(tpr: float, fpr: float) -> float:
    return tpr - fpr


def calibrate_sigmoid(real_vals: List[float], fake_vals: List[float],
                      midpoints=None, steepnesses=None) -> dict:
    """
    Grid-search optimal sigmoid midpoint and steepness.
    Maximises Youden index (TPR - FPR) on Real vs Fake binary classification.

    For anomaly detection: "Fake" is the positive class (anomaly = high strength).
    """
    if midpoints is None:
        # Scan around the data range
        all_vals = real_vals + fake_vals
        lo, hi = np.percentile(all_vals, [5, 95])
        midpoints = np.linspace(lo, hi, 30)

    if steepnesses is None:
        steepnesses = np.logspace(-1, 2, 20)  # 0.1 ~ 100

    real_arr = np.array(real_vals)
    fake_arr = np.array(fake_vals)

    best = {"midpoint": None, "steepness": None, "youden": -1.0,
            "tpr": 0.0, "fpr": 0.0, "threshold": 0.5}

    for mp in midpoints:
        for st in steepnesses:
            # Compute strengths for all samples
            real_strengths = sigmoid(real_arr, mp, st)
            fake_strengths = sigmoid(fake_arr, mp, st)

            # Find optimal decision threshold (max Youden)
            combined = np.concatenate([real_strengths, fake_strengths])
            labels = np.concatenate([
                np.zeros(len(real_strengths)),
                np.ones(len(fake_strengths)),
            ])

            # Scan thresholds
            for thr in np.linspace(0.05, 0.95, 50):
                preds = (combined >= thr).astype(int)
                tp = np.sum((preds == 1) & (labels == 1))
                fp = np.sum((preds == 1) & (labels == 0))
                tn = np.sum((preds == 0) & (labels == 0))
                fn = np.sum((preds == 0) & (labels == 1))

                tpr = tp / max(tp + fn, 1)
                fpr = fp / max(fp + tn, 1)
                yj = youden_index(tpr, fpr)

                if yj > best["youden"]:
                    best["midpoint"] = float(mp)
                    best["steepness"] = float(st)
                    best["youden"] = float(yj)
                    best["tpr"] = float(tpr)
                    best["fpr"] = float(fpr)
                    best["threshold"] = float(thr)
                    best["real_median_strength"] = float(np.median(real_strengths))
                    best["fake_median_strength"] = float(np.median(fake_strengths))

    return best


def calibrate_threeway_thresholds(real_vals, fake_vals, freq_params, noise_params, jpeg_params):
    """
    Find optimal STRENGTH_THRESHOLD_LOW and STRENGTH_THRESHOLD_HIGH
    for 3-way classification (Real / Uncertain / Fake).
    """
    # This is a coarse grid search — the real calibration happens with SFT data
    all_strengths = []
    all_labels = []  # 0=Real, 1=Fake

    for v in real_vals:
        all_strengths.append(sigmoid(v, freq_params["midpoint"], freq_params["steepness"]))
        all_labels.append(0)
    for v in fake_vals:
        all_strengths.append(sigmoid(v, freq_params["midpoint"], freq_params["steepness"]))
        all_labels.append(1)

    all_strengths = np.array(all_strengths)
    all_labels = np.array(all_labels)

    best = {"low": 0.3, "high": 0.7, "accuracy": 0.0}

    for lo in np.linspace(0.1, 0.4, 15):
        for hi in np.linspace(0.6, 0.9, 15):
            if lo >= hi:
                continue
            correct = 0
            for s, l in zip(all_strengths, all_labels):
                if s < lo:
                    pred = 0  # Real
                elif s > hi:
                    pred = 1  # Fake
                else:
                    pred = -1  # Uncertain — count as half-correct for now
                    correct += 0.5
                    continue
                if pred == l:
                    correct += 1

            acc = correct / len(all_labels)
            if acc > best["accuracy"]:
                best["low"] = float(lo)
                best["high"] = float(hi)
                best["accuracy"] = float(acc)

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate forensic expert parameters")
    parser.add_argument("--num-real", type=int, default=50, help="Number of Real images")
    parser.add_argument("--num-fake", type=int, default=20, help="Fake images per generator class")
    parser.add_argument("--quick", action="store_true", help="Quick run: 20 real + 8 fake")
    parser.add_argument("--output", type=str, default="calibration/calibration_report.json")
    args = parser.parse_args()

    if args.quick:
        args.num_real = 20
        args.num_fake = 8

    print("=" * 60)
    print("Expert Calibration — Phase 2.2")
    print(f"  Real images:  {args.num_real}")
    print(f"  Fake/class:   {args.num_fake}")
    print(f"  Total Fake:   ~{args.num_fake * len(GENIMAGE_SUBDIRS)}")
    print("=" * 60)

    # 1. Collect images
    print("\n[1/4] Loading image paths...")
    paths = load_image_paths(
        REAL_DIR, GENIMAGE_TEST_DIR, GENIMAGE_SUBDIRS,
        args.num_real, args.num_fake,
    )
    print(f"  Real:  {len(paths['Real'])} images")
    print(f"  Fake:  {len(paths['Fake'])} images")
    print(f"  Total: {len(paths['Real']) + len(paths['Fake'])} images")

    # 2. Run experts
    print("\n[2/4] Running experts on full images...")
    t0 = time.time()
    metrics = collect_metrics(paths)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # 3. Calibrate each expert
    print("\n[3/4] Calibrating sigmoid parameters...")
    report = {
        "config": {
            "num_real": len(paths["Real"]),
            "num_fake": len(paths["Fake"]),
        },
        "distributions": {},
        "calibration": {},
        "recommended_config": {},
    }

    for name in ("freq", "noise", "jpeg"):
        real_vals = metrics[name]["Real"]
        fake_vals = metrics[name]["Fake"]

        if not real_vals or not fake_vals:
            print(f"  {name}: SKIP (missing data)")
            continue

        real_arr = np.array(real_vals)
        fake_arr = np.array(fake_vals)

        # Statistics
        dist = {
            "real_mean": float(np.mean(real_arr)),
            "real_median": float(np.median(real_arr)),
            "real_std": float(np.std(real_arr)),
            "fake_mean": float(np.mean(fake_arr)),
            "fake_median": float(np.median(fake_arr)),
            "fake_std": float(np.std(fake_arr)),
            "real_p5": float(np.percentile(real_arr, 5)),
            "real_p95": float(np.percentile(real_arr, 95)),
            "fake_p5": float(np.percentile(fake_arr, 5)),
            "fake_p95": float(np.percentile(fake_arr, 95)),
            "separation": float(
                abs(np.mean(fake_arr) - np.mean(real_arr))
                / max(np.std(real_arr) + np.std(fake_arr), 1e-8)
            ),
        }
        report["distributions"][name] = dist

        # Calibrate
        cal = calibrate_sigmoid(real_vals, fake_vals)
        report["calibration"][name] = cal

        # Current vs recommended
        cur_mp = {"freq": FREQ_SIGMOID_MIDPOINT, "noise": NOISE_SIGMOID_MIDPOINT,
                   "jpeg": JPEG_SIGMOID_MIDPOINT}[name]
        cur_st = {"freq": FREQ_SIGMOID_STEEPNESS, "noise": NOISE_SIGMOID_STEEPNESS,
                   "jpeg": JPEG_SIGMOID_STEEPNESS}[name]

        report["recommended_config"][f"{name}_sigmoid_midpoint"] = cal["midpoint"]
        report["recommended_config"][f"{name}_sigmoid_steepness"] = cal["steepness"]

        print(f"\n  [{name}]")
        print(f"    Real median raw:  {dist['real_median']:.4f}")
        print(f"    Fake median raw:  {dist['fake_median']:.4f}")
        print(f"    Separation (d'):  {dist['separation']:.3f}")
        print(f"    Current sigmoid:  mp={cur_mp:.3f}, st={cur_st:.1f}")
        print(f"    Optimal  sigmoid: mp={cal['midpoint']:.4f}, st={cal['steepness']:.1f}")
        print(f"    Youden index:     {cal['youden']:.3f}")
        print(f"    TPR: {cal['tpr']:.3f}  FPR: {cal['fpr']:.3f}  (thr={cal['threshold']:.2f})")
        print(f"    Real median str:  {cal['real_median_strength']:.3f}")
        print(f"    Fake median str:  {cal['fake_median_strength']:.3f}")

    # 4. Three-way thresholds
    print("\n[4/4] Calibrating 3-way thresholds...")
    thr_cal = calibrate_threeway_thresholds(
        metrics["freq"]["Real"], metrics["freq"]["Fake"],
        report["calibration"].get("freq", {}),
        report["calibration"].get("noise", {}),
        report["calibration"].get("jpeg", {}),
    )
    report["recommended_config"]["strength_threshold_low"] = thr_cal["low"]
    report["recommended_config"]["strength_threshold_high"] = thr_cal["high"]
    print(f"  Current: LOW={STRENGTH_THRESHOLD_LOW}, HIGH={STRENGTH_THRESHOLD_HIGH}")
    print(f"  Optimal: LOW={thr_cal['low']:.3f}, HIGH={thr_cal['high']:.3f}")
    print(f"  Accuracy: {thr_cal['accuracy']:.3f}")

    # Save report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {args.output}")

    # Print suggested config.py updates
    print("\n" + "=" * 60)
    print("SUGGESTED config.py UPDATES:")
    print("=" * 60)
    for key, val in report["recommended_config"].items():
        cur = {
            "freq_sigmoid_midpoint": FREQ_SIGMOID_MIDPOINT,
            "freq_sigmoid_steepness": FREQ_SIGMOID_STEEPNESS,
            "noise_sigmoid_midpoint": NOISE_SIGMOID_MIDPOINT,
            "noise_sigmoid_steepness": NOISE_SIGMOID_STEEPNESS,
            "jpeg_sigmoid_midpoint": JPEG_SIGMOID_MIDPOINT,
            "jpeg_sigmoid_steepness": JPEG_SIGMOID_STEEPNESS,
            "strength_threshold_low": STRENGTH_THRESHOLD_LOW,
            "strength_threshold_high": STRENGTH_THRESHOLD_HIGH,
        }.get(key)
        arrow = "→" if cur and abs(cur - val) > 1e-6 else "="
        print(f"  {key.upper():40s} = {val:.6f}  ({arrow} was {cur})")


if __name__ == "__main__":
    main()
