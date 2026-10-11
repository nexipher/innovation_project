#!/usr/bin/env python3
"""
G7-1: evaluate an external expert on one source group, the G2-b way.

What the review asked for, in one command:

  * per-format discrimination with **source-grouped bootstrap CIs**, so a
    number is reported with the uncertainty of the sources it came from;
  * **conditional** admission — a format may be usable while another is not;
  * a **correlation check** against the experts already in the toolkit, so a
    new expert cannot quietly double-count the same evidence;
  * raw scores kept (the adapter writes them), so thresholds and calibrations
    can be re-derived without re-running anything.

Groups come from `calibration/set/g7_split.json`: A fits, B selects, C
validates, and the project's own test partition is reserved for the final
comparison.  This script never mixes them — it reads one group and says which.

Usage:
  python scripts/evaluate_external_expert.py --expert trufor --group A --dry-run
  python scripts/evaluate_external_expert.py --expert trufor --group A
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.g7_smoke_external import EXPERTS, load_expert

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")
RAW_VALUES_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "raw_values.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "calibration")

BOOTSTRAP_ROUNDS = 2000
TREATMENTS = ("native", "png", "jpeg_q95", "jpeg_q85", "jpeg_q70")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_group(group: str, split_path: str = SPLIT_PATH,
               manifest_path: str = MANIFEST_PATH) -> Tuple[List[dict], dict]:
    """The manifest samples whose source belongs to `group`, and the split."""
    with open(split_path, encoding="utf-8") as handle:
        split = json.load(handle)
    if group not in ("A", "B", "C"):
        raise SystemExit(f"unknown group {group!r}; the split has A (fit), B (select), C (validate)")
    members = set(split["groups"][group])
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = [s for s in manifest["samples"] if s["source_id"] in members]
    return samples, split


def existing_scores(sample_ids: List[str],
                    path: str = RAW_VALUES_PATH) -> Dict[str, Dict[str, float]]:
    """Noise/jpeg/frequency_v2 raw metrics for the same images, if cached."""
    try:
        with open(path, encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, ValueError):
        return {}
    wanted = set(sample_ids)
    return {expert: {sid: value for sid, value in blob.get("raw", {}).items()
                     if sid in wanted and value is not None}
            for expert, blob in cached.get("experts", {}).items()}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def auroc(pairs: List[Tuple[float, str]]) -> Optional[float]:
    """AUROC with Fake as the positive class; None when a class is too small."""
    positive = [v for v, gt in pairs if gt == "Fake"]
    negative = [v for v, gt in pairs if gt == "Real"]
    if len(positive) < 3 or len(negative) < 3:
        return None
    scores = np.concatenate([positive, negative])
    ranks = rankdata(scores)
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def bootstrap_ci(rows: List[dict], statistic, rounds: int = BOOTSTRAP_ROUNDS,
                 seed: int = 7) -> Optional[Tuple[float, float]]:
    """
    Resample **sources**, not images.

    Every treatment of one photograph shares its content, so treating them as
    independent would understate the interval — the review's point about
    source-grouped CIs.
    """
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    sources = sorted(by_source)
    if len(sources) < 4:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        picked = [rng.choice(sources) for _ in sources]
        sample = [row for source in picked for row in by_source[source]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if len(values) < rounds // 2:
        return None
    values.sort()
    return (round(values[int(0.025 * len(values))], 4),
            round(values[int(0.975 * len(values)) - 1], 4))


def orientation_corrected(rows: List[dict]) -> dict:
    """
    Their score's direction on the *fitting* group, reported rather than assumed.

    Returns the raw AUROC and the polarity the calibration would use; a value
    near 0.5 means the score carries no direction in this group at all.
    """
    scores = [(row["score"], row["label"]) for row in rows]
    value = auroc(scores)
    if value is None:
        return {"auroc": None, "polarity": None}
    polarity = 1 if value >= 0.5 else -1
    return {"auroc": round(value, 4), "polarity": polarity,
            "separated": round(max(value, 1 - value), 4)}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(expert, samples: List[dict], image_size: Optional[int] = None,
             progress: bool = True) -> List[dict]:
    rows: List[dict] = []
    for index, entry in enumerate(samples, 1):
        path = os.path.join(PROJECT_ROOT, entry["path"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            continue
        if image_size and max(image.shape[:2]) > image_size:
            scale = image_size / max(image.shape[:2])
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        result = expert.analyze(image)
        row = {
            "sample_id": entry["sample_id"], "source_id": entry["source_id"],
            "label": entry["label"], "treatment": entry["treatment"],
            "cell": entry["cell"], "score": float(result.raw_metric),
        }
        stats = (result.metadata or {}).get("stats") or {}
        row.update({f"map_{k}": v for k, v in stats.items() if isinstance(v, (int, float))})
        rows.append(row)
        if progress and index % 25 == 0:
            print(f"    [{index}/{len(samples)}]")
    return rows


def _round(value):
    """None-safe rounding for optional statistics."""
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 4)


def summarise(rows: List[dict], group: str) -> dict:
    report: Dict[str, object] = {"group": group, "samples": len(rows)}
    by_treatment: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_treatment[row["treatment"]].append(row)

    per_treatment = {}
    for treatment in TREATMENTS:
        members = by_treatment.get(treatment, [])
        if not members:
            continue
        pairs = [(row["score"], row["label"]) for row in members]
        entry = {
            "n": len(members),
            "auroc": _round(auroc(pairs)),
            "ci": bootstrap_ci(members, lambda rs: auroc([(r["score"], r["label"]) for r in rs])),
        }
        map_pairs = [(row.get("map_anomaly_peak"), row["label"]) for row in members
                     if isinstance(row.get("map_anomaly_peak"), (int, float))]
        if len(map_pairs) >= 6:
            entry["map_peak_auroc"] = _round(auroc(map_pairs))
        per_treatment[treatment] = entry
    report["per_treatment"] = per_treatment
    report["overall"] = orientation_corrected(rows)
    report["overall_ci"] = bootstrap_ci(
        rows, lambda rs: auroc([(r["score"], r["label"]) for r in rs]))
    return report


def correlation_report(rows: List[dict], cached: Dict[str, Dict[str, float]]) -> dict:
    """Spearman correlation with the experts already in the toolkit."""
    out: Dict[str, dict] = {}
    ours = [(row["sample_id"], row["score"]) for row in rows]
    for expert, scores in cached.items():
        pairs = [(score, scores[sid]) for sid, score in ours if sid in scores]
        if len(pairs) < 10:
            continue
        left, right = zip(*pairs)
        rho, p = spearmanr(left, right)
        if not np.isfinite(rho):
            # A constant score has no correlation to report; saying "nan" in a
            # JSON file is not a number, it is a parser error waiting to happen.
            out[expert] = {"n": len(pairs), "spearman": None,
                           "note": "候选专家分数在本组内恒定，无法计算相关"}
            continue
        out[expert] = {"n": len(pairs), "spearman": round(float(rho), 3),
                       "p": float(f"{p:.2e}")}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", choices=sorted(EXPERTS), required=True)
    parser.add_argument("--group", choices=("A", "B", "C"), default="A",
                        help="A fits, B selects, C validates; the test partition is not here")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    samples, split = load_group(args.group)
    if args.limit:
        samples = samples[:args.limit]
    print(f"专家 {args.expert} ｜ 组 {args.group}（{split['counts'][args.group]} 源）"
          f"｜ 样本 {len(samples)}")

    device = args.device or ("cuda" if _cuda() else "cpu")
    expert, expert_name = load_expert(args.expert, args.dry_run, device)
    reason = expert.unavailable_reason()
    if reason:
        raise SystemExit(f"专家不可用：{reason}")

    rows = evaluate(expert, samples, args.image_size)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expert": args.expert, "expert_name": expert_name,
        "dry_run": args.dry_run, "device": device,
        "split_hash": split["hash"],
        "weights": expert.provenance().as_dict(),
        "timing": expert.timing_summary() if hasattr(expert, "timing_summary") else {},
        "summary": summarise(rows, args.group),
        "correlation_with_existing": correlation_report(
            rows, existing_scores([row["sample_id"] for row in rows])),
        "rows": rows,
    }
    output = args.output or os.path.join(
        OUTPUT_DIR,
        f"g7_external_{args.expert}_{args.group}{'_dry_run' if args.dry_run else ''}.json")
    with open(output, "w", encoding="utf-8") as handle:
        # allow_nan=False turns a slipped-through NaN into a failure here
        # rather than a file no strict JSON reader will accept later.
        json.dump(_finite_tree(report), handle, ensure_ascii=False, indent=1,
                  allow_nan=False)

    print(f"\n{'格式':10s} {'n':>4s} {'AUROC':>7s} {'95% CI(按源)':>18s} {'map峰值AUROC':>12s}")
    for treatment, entry in report["summary"]["per_treatment"].items():
        ci = entry.get("ci")
        print(f"{treatment:10s} {entry['n']:4d} "
              f"{(entry['auroc'] if entry['auroc'] is not None else float('nan')):7.3f} "
              f"{str(ci):>18s} {str(entry.get('map_peak_auroc', '—')):>12s}")
    overall = report["summary"]["overall"]
    print(f"\n整体：AUROC {overall['auroc']}  polarity {overall['polarity']} "
          f"分离度 {overall.get('separated')}  CI {report['summary']['overall_ci']}")
    if report["correlation_with_existing"]:
        print("与现有专家的 Spearman 相关：")
        for name, entry in report["correlation_with_existing"].items():
            rho = entry["spearman"]
            shown = f"{rho:+.3f}" if rho is not None else "—（分数恒定）"
            print(f"  {name:18s} ρ={shown}  (n={entry['n']})")
    print(f"\nReport: {os.path.relpath(output, PROJECT_ROOT)}")


def _finite_tree(node):
    """Recursively replace non-finite floats with None, and keep the rest."""
    if isinstance(node, dict):
        return {key: _finite_tree(value) for key, value in node.items()}
    if isinstance(node, (list, tuple)):
        return [_finite_tree(value) for value in node]
    if isinstance(node, float) and not np.isfinite(node):
        return None
    return node


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
