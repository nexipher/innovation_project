#!/usr/bin/env python3
"""
G2-d analysis: is the expert-evidence gain (or loss) statistically real?

The four conditions run on the same 120 samples, so the comparison is paired:
each condition is tested against the RGB baseline with McNemar's test, and
AUROC differences get a paired bootstrap interval.  Unpaired summaries would
hide that the arms differ on exactly the samples where the model changed its
mind.

Outputs a console table plus calibration/g2_gain_analysis.json.

Usage:
  python scripts/analyze_g2_gain.py
  python scripts/analyze_g2_gain.py --bootstrap 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.qwen_gain_baseline import REPORT_PATH, _auroc, _pseudo_probability

ANALYSIS_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_gain_analysis.json")
BASELINE = "rgb"


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def align_records(conditions: Dict[str, List[dict]]) -> Tuple[List[str], Dict[str, dict]]:
    """
    Index every condition's records by sample_id, keeping only the samples
    present in all of them (a paired test is only valid on shared samples).
    """
    shared: Optional[set] = None
    by_condition: Dict[str, dict] = {}
    for name, records in conditions.items():
        by_condition[name] = {r["sample_id"]: r for r in records}
        keys = set(by_condition[name])
        shared = keys if shared is None else (shared & keys)
    if not shared:
        raise SystemExit("no samples shared across all conditions")
    order = sorted(shared)
    return order, by_condition


def mcnemar(baseline_correct: List[bool], other_correct: List[bool]) -> dict:
    """
    McNemar's test on paired correctness (continuity-corrected).

    b: baseline right, other wrong; c: baseline wrong, other right.
    """
    b = sum(1 for base, other in zip(baseline_correct, other_correct)
            if base and not other)
    c = sum(1 for base, other in zip(baseline_correct, other_correct)
            if not base and other)
    if b + c == 0:
        return {"b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    return {
        "b": b, "c": c,
        "statistic": float(statistic),
        "p_value": float(chi2.sf(statistic, df=1)),
    }


def paired_bootstrap_auroc(
    records_by_condition: Dict[str, dict],
    order: List[str],
    baseline: str = BASELINE,
    iterations: int = 2000,
    seed: int = 42,
) -> Dict[str, dict]:
    """Paired bootstrap over samples: 95% CI for each AUROC and for Δ vs baseline."""
    rng = np.random.default_rng(seed)
    names = list(records_by_condition)
    base_records = [records_by_condition[baseline][sid] for sid in order]
    n = len(order)

    draws = {name: [] for name in names}
    deltas = {name: [] for name in names}
    for _ in range(iterations):
        index = rng.integers(0, n, n)
        sample = [base_records[i] for i in index]
        base_auc = _auroc(sample)
        for name in names:
            arm = [records_by_condition[name][order[i]] for i in index]
            auc = _auroc(arm)
            if auc is None:
                continue
            draws[name].append(auc)
            if base_auc is not None and name != baseline:
                deltas[name].append(auc - base_auc)

    summary: Dict[str, dict] = {}
    for name in names:
        point = _auroc([records_by_condition[name][sid] for sid in order])
        values = np.array(draws[name])
        entry = {
            "auroc": point,
            "auroc_ci": [float(np.percentile(values, 2.5)),
                         float(np.percentile(values, 97.5))] if len(values) else None,
        }
        if name != baseline and deltas[name]:
            delta_values = np.array(deltas[name])
            entry["delta_auroc"] = (point - _auroc(base_records)
                                    if point is not None else None)
            entry["delta_ci"] = [float(np.percentile(delta_values, 2.5)),
                                 float(np.percentile(delta_values, 97.5))]
            # Two-sided bootstrap p: how often the sign flips.
            entry["delta_p"] = float(
                2 * min((delta_values <= 0).mean(), (delta_values >= 0).mean())
            )
        summary[name] = entry
    return summary


def class_breakdown(records: List[dict]) -> dict:
    """Where an arm sits on the Real/Fake trade-off — accuracy alone hides it."""
    real = [r for r in records if r["gt"] == "Real"]
    fake = [r for r in records if r["gt"] == "Fake"]
    return {
        "real_recall": sum(1 for r in real if r["verdict"] == "Real") / len(real),
        "fake_recall": sum(1 for r in fake if r["verdict"] == "Fake") / len(fake),
        "predicted_real_rate": sum(1 for r in records if r["verdict"] == "Real")
        / len(records),
        "uncertain_rate": sum(1 for r in records if r["verdict"] == "Uncertain")
        / len(records),
    }


def analyze(report: dict, iterations: int = 2000) -> dict:
    conditions = {name: entry["records"]
                  for name, entry in report["conditions"].items() if entry["records"]}
    order, by_condition = align_records(conditions)

    baseline_correct = [by_condition[BASELINE][sid]["verdict"]
                        == by_condition[BASELINE][sid]["gt"] for sid in order]
    bootstrap = paired_bootstrap_auroc(by_condition, order,
                                       iterations=iterations)

    arms = {}
    for name in conditions:
        records = [by_condition[name][sid] for sid in order]
        correct = [r["verdict"] == r["gt"] for r in records]
        arms[name] = {
            "n": len(records),
            "accuracy": sum(correct) / len(correct),
            "auroc": bootstrap[name]["auroc"],
            "auroc_ci": bootstrap[name]["auroc_ci"],
            "delta_auroc": bootstrap[name].get("delta_auroc"),
            "delta_ci": bootstrap[name].get("delta_ci"),
            "delta_p": bootstrap[name].get("delta_p"),
            "mcnemar": mcnemar(baseline_correct, correct) if name != BASELINE else None,
            **class_breakdown(records),
        }
    return {
        "baseline": BASELINE,
        "paired_samples": len(order),
        "bootstrap_iterations": iterations,
        "arms": arms,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(analysis: dict) -> None:
    arms = analysis["arms"]
    print(f"\n配对样本 n={analysis['paired_samples']}, "
          f"bootstrap={analysis['bootstrap_iterations']}（基线: {analysis['baseline']}）\n")

    header = (f"{'condition':8s} {'acc':>6s} {'auroc':>6s} {'auroc 95%CI':>16s} "
              f"{'Δauroc':>7s} {'Δ 95%CI':>16s} {'p':>7s} "
              f"{'Real召回':>8s} {'Fake召回':>8s} {'判Real率':>8s}")
    print(header)
    print("-" * len(header))
    for name, arm in arms.items():
        ci = arm["auroc_ci"]
        ci_text = f"[{ci[0]:.3f},{ci[1]:.3f}]" if ci else "-"
        if arm["delta_ci"]:
            delta_text = f"[{arm['delta_ci'][0]:+.3f},{arm['delta_ci'][1]:+.3f}]"
        else:
            delta_text = "-"
        delta = f"{arm['delta_auroc']:+.3f}" if arm["delta_auroc"] is not None else "-"
        p_value = f"{arm['delta_p']:.3f}" if arm["delta_p"] is not None else "-"
        print(f"{name:8s} {arm['accuracy']:6.3f} {arm['auroc']:6.3f} {ci_text:>16s} "
              f"{delta:>7s} {delta_text:>16s} {p_value:>7s} "
              f"{arm['real_recall']:8.3f} {arm['fake_recall']:8.3f} "
              f"{arm['predicted_real_rate']:8.3f}")

    print("\nMcNemar（vs 基线，配对正确率）:")
    for name, arm in arms.items():
        if arm["mcnemar"] is None:
            continue
        m = arm["mcnemar"]
        print(f"  {name:8s} 基线对/该臂错 b={m['b']:3d}  基线错/该臂对 c={m['c']:3d}  "
              f"χ²={m['statistic']:6.2f}  p={m['p_value']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        report = json.load(handle)
    analysis = analyze(report, iterations=args.bootstrap)
    analysis["source_report"] = os.path.relpath(args.report, PROJECT_ROOT)
    analysis["generated_at"] = report.get("generated_at")

    print_table(analysis)
    with open(ANALYSIS_PATH, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
    print(f"\nAnalysis: {ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
