#!/usr/bin/env python3
"""
G2-b: measure expert discrimination, stability, applicability and overlap.

Reads calibration/set/manifest.json, runs every expert on the FULL image of
every calibration sample (intrinsic tool quality; the runtime pipeline is
measured separately in G2-d), and writes:

  calibration/g2_expert_report.json   metrics, reliability intervals, failure
                                      conditions, perturbation stability and
                                      error-overlap matrices
  calibration/set/artifacts/          rendered visual artifacts (subset)

Key comparisons:
  native cell        Real-JPEG vs Fake-PNG  -> the confounded reference
  png cell           Real-PNG  vs Fake-PNG  -> format-matched
  jpeg_q95/85/70     Real-JPEG vs Fake-JPEG -> format+quality-matched
The gap between the confounded and matched AUROCs quantifies how much of an
expert's apparent skill is container-format detection.

Usage:
  python scripts/evaluate_experts_g2.py
  python scripts/evaluate_experts_g2.py --experts noise jpeg ela --artifacts-per-class 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.image_utils import ImageUtils

SET_DIR = os.path.join(PROJECT_ROOT, "calibration", "set")
MANIFEST_PATH = os.path.join(SET_DIR, "manifest.json")
ARTIFACT_DIR = os.path.join(SET_DIR, "artifacts")
REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_expert_report.json")

MATCHED_PAIRS = [
    ("native", "real_native", "fake_native"),
    ("png", "real_png", "fake_png"),
    ("jpeg_q95", "real_jpeg_q95", "fake_jpeg_q95"),
    ("jpeg_q85", "real_jpeg_q85", "fake_jpeg_q85"),
    ("jpeg_q70", "real_jpeg_q70", "fake_jpeg_q70"),
]
PERTURBATION_CELLS = ("blur", "noise", "resize_0.5x", "resize_2x", "sharpen", "brightness", "screenshot")
STRENGTH_BINS = ((0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0))


def build_experts(names: List[str]) -> Dict[str, Any]:
    from experts.ela import ELAExpert
    from experts.frequency import FrequencyExpert
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.jpeg import JPEGExpert
    from experts.noise import NoiseExpert

    registry = {
        "frequency_v1": FrequencyExpert,
        "frequency_v2": FrequencyExpertV2,
        "noise": NoiseExpert,
        "jpeg": JPEGExpert,
        "ela": ELAExpert,
    }
    return {name: registry[name]() for name in names}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auroc(positive: List[float], negative: List[float]) -> Optional[float]:
    """Rank-based AUROC with tie handling (positive class ranks high)."""
    if len(positive) < 3 or len(negative) < 3:
        return None
    scores = np.concatenate([positive, negative])
    ranks = rankdata(scores)
    n_pos, n_neg = len(positive), len(negative)
    rank_sum = ranks[:n_pos].sum()
    return float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_threshold(positive: List[float], negative: List[float]) -> Tuple[float, float, float]:
    """Youden-optimal threshold on the pooled raw metrics.

    Returns (threshold, tpr, fpr).
    """
    if not positive or not negative:
        return 0.0, 0.0, 0.0
    candidates = np.unique(np.concatenate([positive, negative]))
    best = (float(np.median(candidates)), 0.0, 1.0)
    best_youden = -1.0
    pos_arr, neg_arr = np.asarray(positive), np.asarray(negative)
    for threshold in candidates:
        tpr = float((pos_arr >= threshold).mean())
        fpr = float((neg_arr >= threshold).mean())
        youden = tpr - fpr
        if youden > best_youden:
            best_youden = youden
            best = (float(threshold), tpr, fpr)
    return best


def f1_at(positive: List[float], negative: List[float], threshold: float) -> float:
    pos_arr, neg_arr = np.asarray(positive), np.asarray(negative)
    tp = int((pos_arr >= threshold).sum())
    fp = int((neg_arr >= threshold).sum())
    fn = len(positive) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def phi_coefficient(errors_a: List[bool], errors_b: List[bool]) -> Optional[float]:
    """Matthews-style phi between two error indicator vectors."""
    a = np.asarray(errors_a, dtype=bool)
    b = np.asarray(errors_b, dtype=bool)
    if a.size != b.size or a.size == 0:
        return None
    n11 = int((a & b).sum())
    n00 = int((~a & ~b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    denominator = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    if denominator == 0:
        return None
    return float((n11 * n00 - n10 * n01) / denominator)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_experts(samples: List[dict], experts: Dict[str, Any],
                cache_path: Optional[str] = None, use_cache: bool = True) -> Dict[str, dict]:
    """Run every expert on every sample; cache per-sample values for re-runs."""
    if use_cache and cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("samples") == len(samples):
            print(f"  using cached raw values: {cache_path}")
            return {
                name: {
                    "raw": {k: float(v) for k, v in entry["raw"].items()},
                    "strength": {k: float(v) for k, v in entry["strength"].items()},
                    "latency_ms": {k: float(v) for k, v in entry["latency_ms"].items()},
                    "meta": entry["meta"],
                    "failures": entry.get("failures", []),
                }
                for name, entry in cached["experts"].items()
                if name in experts
            }

    results: Dict[str, dict] = {
        name: {"raw": {}, "strength": {}, "latency_ms": {}, "failures": [], "meta": {}}
        for name in experts
    }
    total = len(samples)

    for index, sample in enumerate(samples, 1):
        image = ImageUtils.load_image(os.path.join(PROJECT_ROOT, sample["path"]))
        if image is None or image.size == 0:
            for name in experts:
                results[name]["failures"].append(sample["sample_id"])
            continue
        for name, expert in experts.items():
            started = time.perf_counter()
            try:
                expert_result = expert.analyze(image)
            except Exception:
                results[name]["failures"].append(sample["sample_id"])
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results[name]["raw"][sample["sample_id"]] = float(expert_result.raw_metric)
            results[name]["strength"][sample["sample_id"]] = float(expert_result.strength)
            results[name]["latency_ms"][sample["sample_id"]] = elapsed_ms
            results[name]["meta"][sample["sample_id"]] = {
                "label": sample["label"], "cell": sample["cell"],
                "source_id": sample["source_id"],
                "generator": sample["generator"], "bucket": sample["resolution_bucket"],
                "treatment": sample["treatment"],
            }
        if index % 100 == 0:
            print(f"  [{index}/{total}] samples evaluated")

    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump({
                "samples": len(samples),
                "experts": {
                    name: {
                        "raw": results[name]["raw"],
                        "strength": results[name]["strength"],
                        "latency_ms": results[name]["latency_ms"],
                        "meta": results[name]["meta"],
                        "failures": results[name]["failures"],
                    } for name in experts
                },
            }, handle, ensure_ascii=False)
        print(f"  raw values cached: {cache_path}")
    return results


def values_for_cell(results: dict, cell: str) -> List[float]:
    return [
        results["raw"][sid] for sid, meta in results["meta"].items()
        if meta["cell"] == cell and sid in results["raw"]
    ]


def evaluate_expert(name: str, results: dict, samples: List[dict]) -> dict:
    report: Dict[str, Any] = {"cell_metrics": {}, "raw_metric_stats": {}}
    if not results["raw"]:
        return {"error": "no samples analysed"}

    # --- matched and confounded cell comparisons -----------------------
    thresholds_by_cell: Dict[str, float] = {}
    for label, real_cell, fake_cell in MATCHED_PAIRS:
        real_values = values_for_cell(results, real_cell)
        fake_values = values_for_cell(results, fake_cell)
        auc = auroc(fake_values, real_values)
        threshold, tpr, fpr = best_threshold(fake_values, real_values)
        thresholds_by_cell[label] = threshold
        report["cell_metrics"][label] = {
            "real_cell": real_cell, "fake_cell": fake_cell,
            "n_real": len(real_values), "n_fake": len(fake_values),
            "auroc_fake_vs_real": auc,
            "best_threshold": threshold, "tpr": tpr, "fpr": fpr,
            "f1": f1_at(fake_values, real_values, threshold),
            "real_median": float(np.median(real_values)) if real_values else None,
            "fake_median": float(np.median(fake_values)) if fake_values else None,
        }
        report["raw_metric_stats"][label] = {
            "real_p10": float(np.percentile(real_values, 10)) if real_values else None,
            "real_p90": float(np.percentile(real_values, 90)) if real_values else None,
            "fake_p10": float(np.percentile(fake_values, 10)) if fake_values else None,
            "fake_p90": float(np.percentile(fake_values, 90)) if fake_values else None,
        }

    native = report["cell_metrics"].get("native", {}).get("auroc_fake_vs_real")
    png = report["cell_metrics"].get("png", {}).get("auroc_fake_vs_real")
    q70 = report["cell_metrics"].get("jpeg_q70", {}).get("auroc_fake_vs_real")
    if native is not None and png is not None:
        report["format_confound_gap"] = {
            "native_auroc": native, "png_matched_auroc": png,
            "gap": native - png,
            "interpretation": "native vs png only differ in container, both keep the JPEG-history confound",
        }
    if png is not None and q70 is not None:
        # The meaningful confound measure: PNG cells keep Real's JPEG history
        # while Fake has none; the q70 cell gives both a fresh compression
        # history, simulating real-world delivery. A large positive drop means
        # the expert's skill was largely compression-history detection.
        report["confound_drop_png_to_q70"] = {
            "png_auroc": png, "jpeg_q70_auroc": q70, "drop": png - q70,
            "interpretation": "large positive drop = skill is compression-history detection",
        }
    if png is not None:
        report["separation_polarity_corrected"] = float(max(png, 1.0 - png))
        report["semantics_aligned"] = bool(png > 0.5)
        report["semantics_note"] = (
            "aligned: high metric empirically means Fake (matches the expert's claim); "
            "inverted: high metric empirically means Real — support labels are misleading "
            "for this task"
        )

    # --- perturbation stability ---------------------------------------
    png_threshold = thresholds_by_cell.get("png", 0.0)
    perturbation_report = {}
    for treatment in PERTURBATION_CELLS:
        real_values = values_for_cell(results, f"real_{treatment}")
        fake_values = values_for_cell(results, f"fake_{treatment}")
        if len(real_values) < 3 or len(fake_values) < 3:
            continue
        # delta against the same source's lossless png value
        deltas = []
        for sid, meta in results["meta"].items():
            if meta["cell"] != f"{meta['label'].lower()}_{treatment}":
                continue
            baseline_sid = f"{meta['source_id']}_png"
            if baseline_sid in results["raw"] and sid in results["raw"]:
                deltas.append(results["raw"][sid] - results["raw"][baseline_sid])
        crossings = 0
        deltas_arr = np.asarray(deltas) if deltas else np.array([0.0])
        if deltas:
            # how often the perturbation flips the raw metric across the matched threshold
            for sid, meta in results["meta"].items():
                if meta["cell"] != f"{meta['label'].lower()}_{treatment}":
                    continue
                baseline_sid = f"{meta['source_id']}_png"
                if baseline_sid not in results["raw"] or sid not in results["raw"]:
                    continue
                before = results["raw"][baseline_sid] >= png_threshold
                after = results["raw"][sid] >= png_threshold
                if before != after:
                    crossings += 1
        perturbation_report[treatment] = {
            "auroc": auroc(fake_values, real_values),
            "n_real": len(real_values), "n_fake": len(fake_values),
            "delta_median": float(np.median(deltas_arr)),
            "delta_p90_abs": float(np.percentile(np.abs(deltas_arr), 90)),
            "threshold_crossing_rate": crossings / max(len(deltas), 1),
        }
    report["perturbation_stability"] = perturbation_report

    # --- reliability intervals (pooled matched cells) ------------------
    pooled = []
    for label in ("png", "jpeg_q95", "jpeg_q85", "jpeg_q70"):
        real_cell = f"real_{label}" if label != "png" else "real_png"
        fake_cell = f"fake_{label}" if label != "png" else "fake_png"
        for sid, meta in results["meta"].items():
            if meta["cell"] in (real_cell, fake_cell) and sid in results["strength"]:
                pooled.append((results["strength"][sid], meta["label"]))
    intervals = []
    for low, high in STRENGTH_BINS:
        bucket = [label for strength, label in pooled if low <= strength < high]
        if not bucket:
            intervals.append({"bin": f"[{low},{high})", "n": 0, "precision_fake": None})
            continue
        intervals.append({
            "bin": f"[{low},{high})", "n": len(bucket),
            "precision_fake": sum(1 for label in bucket if label == "Fake") / len(bucket),
        })
    report["reliability_intervals"] = intervals

    # --- latency -------------------------------------------------------
    latencies = list(results["latency_ms"].values())
    if latencies:
        report["latency_ms"] = {
            "median": float(np.median(latencies)),
            "p90": float(np.percentile(latencies, 90)),
        }
    report["failures"] = {"count": len(results["failures"]),
                          "sample_ids": results["failures"][:10]}
    return report


def error_overlap(name_results: Dict[str, dict], samples: List[dict]) -> dict:
    """Pairwise phi between expert error indicators on the PNG matched cell."""
    cell_labels = {}
    for sid, meta in next(iter(name_results.values()))["meta"].items():
        if meta["cell"] in ("real_png", "fake_png"):
            cell_labels[sid] = meta["label"]

    # per-expert best PNG threshold and error set
    error_sets = {}
    for name, results in name_results.items():
        reals = [results["raw"][sid] for sid in cell_labels
                 if cell_labels[sid] == "Real" and sid in results["raw"]]
        fakes = [results["raw"][sid] for sid in cell_labels
                 if cell_labels[sid] == "Fake" and sid in results["raw"]]
        if len(reals) < 3 or len(fakes) < 3:
            continue
        threshold, _, _ = best_threshold(fakes, reals)
        errors = []
        for sid in cell_labels:
            if sid not in results["raw"]:
                errors.append(False)
                continue
            predicted_fake = results["raw"][sid] >= threshold
            truth_fake = cell_labels[sid] == "Fake"
            errors.append(predicted_fake != truth_fake)
        error_sets[name] = errors

    matrix = {}
    names = list(error_sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            matrix[f"{a}_vs_{b}"] = phi_coefficient(error_sets[a], error_sets[b])
    return matrix


def render_artifact_subset(samples: List[dict], experts: Dict[str, Any], per_class: int) -> int:
    """Render artifacts for a stratified subset (png cell) of both classes."""
    subset = (
        [s for s in samples if s["cell"] == "real_png"][:per_class]
        + [s for s in samples if s["cell"] == "fake_png"][:per_class]
    )
    count = 0
    for sample in subset:
        image = ImageUtils.load_image(os.path.join(PROJECT_ROOT, sample["path"]))
        if image is None:
            continue
        for name, expert in experts.items():
            try:
                artifacts = expert.render_artifacts(image)
            except Exception:
                continue
            for artifact_name, artifact in artifacts.items():
                out_dir = os.path.join(ARTIFACT_DIR, name)
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"{sample['sample_id']}_{artifact_name}.png")
                if cv2.imwrite(path, artifact):
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", nargs="*",
                        default=["frequency_v1", "frequency_v2", "noise", "jpeg", "ela"])
    parser.add_argument("--artifacts-per-class", type=int, default=25)
    parser.add_argument("--skip-artifacts", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute raw values instead of reusing the cache")
    args = parser.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = manifest["samples"]
    print(f"Manifest: {len(samples)} samples, {len(manifest['cells'])} cells")

    experts = build_experts(args.experts)
    print(f"Experts: {list(experts)}")

    print("\nRunning experts on full images ...")
    started = time.time()
    cache_path = os.path.join(SET_DIR, "raw_values.json")
    results = run_experts(samples, experts, cache_path=cache_path,
                          use_cache=not args.no_cache)
    print(f"  done in {time.time() - started:.1f}s")

    print("\nComputing metrics ...")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": os.path.relpath(MANIFEST_PATH, PROJECT_ROOT),
        "samples": len(samples),
        "experts": {
            name: evaluate_expert(name, results[name], samples)
            for name in experts
        },
        "error_overlap_png_cell": error_overlap(results, samples),
    }

    if not args.skip_artifacts:
        print("Rendering artifact subset ...")
        rendered = render_artifact_subset(samples, experts, args.artifacts_per_class)
        report["artifacts"] = {
            "dir": os.path.relpath(ARTIFACT_DIR, PROJECT_ROOT),
            "rendered": rendered,
        }
        print(f"  {rendered} artifacts")

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\nReport: {REPORT_PATH}")

    # ---- console summary ---------------------------------------------
    print("\n=== AUROC summary (Fake vs Real; >0.5 aligned, <0.5 inverted) ===")
    print(f"{'expert':14s} {'png':>6s} {'q70':>6s} {'sep':>6s} {'aligned':>8s} {'drop':>7s}")
    for name, expert_report in report["experts"].items():
        cells = expert_report.get("cell_metrics", {})
        png = cells.get("png", {}).get("auroc_fake_vs_real")
        q70 = cells.get("jpeg_q70", {}).get("auroc_fake_vs_real")
        separation = expert_report.get("separation_polarity_corrected")
        aligned = expert_report.get("semantics_aligned")
        drop = expert_report.get("confound_drop_png_to_q70", {}).get("drop")
        print(f"{name:14s} {png if png is None else f'{png:.3f}':>6s} "
              f"{q70 if q70 is None else f'{q70:.3f}':>6s} "
              f"{separation if separation is None else f'{separation:.3f}':>6s} "
              f"{str(aligned):>8s} "
              f"{drop if drop is None else f'{drop:+.3f}':>7s}")

    print("\n=== Perturbation AUROC (png cell baseline) ===")
    for name, expert_report in report["experts"].items():
        stability = expert_report.get("perturbation_stability", {})
        parts = []
        for treatment, values in stability.items():
            auc = values.get("auroc")
            parts.append(f"{treatment}={auc:.2f}" if auc is not None else f"{treatment}=-")
        print(f"  {name:14s} " + "  ".join(parts))


if __name__ == "__main__":
    main()
