#!/usr/bin/env python3
"""
G4-c: admission — which trajectories may teach the model anything.

A trajectory that happens to end on the right label is not training material.
Two gates decide, and both are recorded per trajectory so a reviewer can see
why a sample was kept or dropped.

Layer 1 — conditional.  A tool may only appear where the calibration measured
it as separating (the same 0.65 bar the generator gated on), the retired
experts (frequency v1, ELA) may not appear at all, and the weak frequency v2
may never be the *sole* basis of a conclusion — a trajectory that calls only
frequency cannot be a positive sample no matter how it ends.

Layer 2 — gain, measured against the no-tool run of the *same* variant:
Brier and NLL for the true label, the posterior it assigns to the truth, and
whether the run traded honest abstention for confident error.  Only a
conditional trajectory whose risk actually fell becomes a positive sample; a
run that abstains where the baseline was confidently wrong is kept separately
as an honest abstention.

Admission reads the training partition only.  Choosing trajectories with the
validation or test partition in view would fit the selection to the metric the
model is later scored on, which is the leak this project keeps re-learning to
avoid.

Output: `sft_data/admission_report.json` (per-trajectory decisions + aggregate).

Usage:
  python scripts/admit_trajectories_g4.py
  python scripts/admit_trajectories_g4.py --trajectories-dir sft_data/trajectories
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.generate_trajectories_g4 import (
    APPLICABILITY_THRESHOLD,
    load_applicability,
    tool_applicable,
)

TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories")
REPORT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "admission_report.json")

# Experts that were retired (G2-e) and must not appear in any training input.
RETIRED_TOOLS = ("freq_v1", "ela")
# A trajectory may only be a positive sample if at least one of its tools was
# measured as separating; frequency alone never is.
CORROBORATION_ONLY = ("freq",)

# How much the Brier score must improve before the gain counts.  A hair of
# improvement is noise, not a lesson.
RISK_EPSILON = 0.01
CONFIDENT = 0.8


def load_trajectories(directory: str = TRAJECTORIES_DIR) -> List[dict]:
    records = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == "manifest.json":
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        record["_file"] = name
        records.append(record)
    return records


def assert_train_only(records: List[dict]) -> None:
    """
    Refuse to select trajectories using anything but the training partition.

    Called before any decision is made, so a misplaced file fails loudly
    rather than quietly shaping the training set.
    """
    offenders = sorted({r.get("split") for r in records} - {"train"})
    if offenders:
        raise ValueError(
            f"admission may only read the training partition; found {offenders}"
        )


def pair_by_variant(records: List[dict]) -> Dict[str, Dict[str, dict]]:
    """variant_id -> {policy: record}, so each tool run has its baseline."""
    paired: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for record in records:
        paired[record["variant_id"]][record["policy"]] = record
    return paired


def _brier(record: dict) -> Optional[float]:
    return (record.get("scores") or {}).get("posterior_brier")


def _p_truth(record: dict) -> Optional[float]:
    error = (record.get("scores") or {}).get("posterior_error")
    return None if error is None else 1.0 - error


def _correct(record: dict) -> bool:
    return record.get("final_verdict") == record.get("ground_truth")


def _confident_error(record: dict) -> bool:
    confidence = record.get("confidence") or 0.0
    return (record.get("final_verdict") in ("Real", "Fake")
            and not _correct(record) and confidence >= CONFIDENT)


def admit(tool_record: dict, baseline_record: Optional[dict],
          applicability: Dict[str, Dict[str, float]],
          threshold: float = APPLICABILITY_THRESHOLD) -> dict:
    """
    Decide one trajectory's fate, with the reasoning that produced it.

    Returns a decision dict: category (positive / honest_abstention / reject),
    reasons, and the measured deltas.
    """
    reasons: List[str] = []
    treatment = tool_record["treatment"]
    tools = tool_record.get("tools_served") or []

    retired = [t for t in tools if t in RETIRED_TOOLS]
    if retired:
        reasons.append(f"retired expert in trajectory: {retired}")

    # A tool below the bar is blocking when it is meant to carry weight; the
    # corroboration-only expert may appear even though its own separation is
    # weak — that is what "may only corroborate" means — but it can never be
    # the reason a sample is admitted.
    conditional_ok = True
    carrying = []
    for tool in tools:
        ok, why = tool_applicable(tool, treatment, applicability, threshold)
        if ok and tool not in CORROBORATION_ONLY:
            carrying.append(tool)
        elif not ok and tool not in CORROBORATION_ONLY:
            conditional_ok = False
            reasons.append(why)
        elif not ok:
            reasons.append(f"corroboration only, below the bar: {why}")

    positive_allowed = bool(carrying)
    if not positive_allowed and tools:
        reasons.append("no tool in this trajectory passed the bar on its own")

    deltas: Dict[str, Optional[float]] = {
        "brier_delta": None, "nll_delta": None, "p_truth_delta": None,
        "cost_delta": None,
    }
    risk_reduced = False
    introduced_confident_error = _confident_error(tool_record)
    if introduced_confident_error:
        reasons.append("introduced a confident error")

    if baseline_record is not None:
        tool_brier, base_brier = _brier(tool_record), _brier(baseline_record)
        if tool_brier is not None and base_brier is not None:
            deltas["brier_delta"] = round(tool_brier - base_brier, 4)
            risk_reduced = deltas["brier_delta"] <= -RISK_EPSILON
        tool_nll = (tool_record.get("scores") or {}).get("posterior_nll")
        base_nll = (baseline_record.get("scores") or {}).get("posterior_nll")
        if tool_nll is not None and base_nll is not None:
            deltas["nll_delta"] = round(tool_nll - base_nll, 4)
            risk_reduced = risk_reduced or deltas["nll_delta"] <= -RISK_EPSILON
        tool_p, base_p = _p_truth(tool_record), _p_truth(baseline_record)
        if tool_p is not None and base_p is not None:
            deltas["p_truth_delta"] = round(tool_p - base_p, 4)
        deltas["cost_delta"] = round(
            tool_record["counters"]["weighted_cost"]
            - baseline_record["counters"]["weighted_cost"], 4)
    else:
        reasons.append("no no-tool baseline for this variant")

    abstained = tool_record.get("final_verdict") == "Uncertain"
    baseline_confidently_wrong = (
        baseline_record is not None
        and _confident_error(baseline_record)
    )
    honest_abstention = abstained and (baseline_confidently_wrong
                                       or (baseline_record is not None
                                           and baseline_record.get("final_verdict") == "Uncertain"))

    # A tool-free trajectory has no tool gain to measure — comparing it with
    # its own baseline yields zero by construction, which silently discarded
    # every correct no-tool answer.  What makes it training material is that it
    # reached the right label from the image alone.
    no_tools = not tools
    if no_tools:
        if (not introduced_confident_error and _correct(tool_record)
                and tool_record.get("final_verdict") in ("Real", "Fake")):
            category = "positive"
            reasons.append("no-tool conclusion, correct and confident")
        elif abstained:
            category = "honest_abstention"
        else:
            category = "reject"
            if not _correct(tool_record):
                reasons.append("no-tool conclusion is wrong")
    elif retired or introduced_confident_error or not conditional_ok:
        category = "reject"
    elif positive_allowed and risk_reduced and _correct(tool_record):
        category = "positive"
    elif honest_abstention:
        category = "honest_abstention"
    else:
        category = "reject"
        if not risk_reduced:
            reasons.append("risk did not fall against the no-tool run")
        if risk_reduced and not _correct(tool_record):
            reasons.append("risk fell but the conclusion is still wrong")

    return {
        "trajectory_id": tool_record["trajectory_id"],
        "policy": tool_record["policy"],
        "uses_tools": not no_tools,
        "tools_served": tools,
        "variant_id": tool_record["variant_id"],
        "source_id": tool_record["source_id"],
        "treatment": treatment,
        "generator": tool_record["generator"],
        "ground_truth": tool_record["ground_truth"],
        "verdict": tool_record.get("final_verdict"),
        "category": category,
        "conditional_ok": conditional_ok,
        "risk_reduced": risk_reduced,
        "cost_delta": deltas["cost_delta"],
        "reasons": reasons,
        **{k: v for k, v in deltas.items() if k != "cost_delta"},
    }


def run(records: List[dict], applicability: Dict[str, Dict[str, float]],
        threshold: float = APPLICABILITY_THRESHOLD) -> dict:
    assert_train_only(records)
    paired = pair_by_variant(records)

    decisions = []
    for variant_id, by_policy in sorted(paired.items()):
        baseline = by_policy.get("no-tool")
        for policy, record in sorted(by_policy.items()):
            decisions.append(admit(record, baseline, applicability, threshold))

    categories = Counter(d["category"] for d in decisions)
    admitted = [d for d in decisions if d["category"] == "positive"]
    rejected = [d for d in decisions if d["category"] == "reject"]

    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else None

    no_tool_positives = [d for d in admitted if not d.get("uses_tools", True)]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "applicability_threshold": threshold,
        "no_tool_positive_labels": dict(
            Counter(d["ground_truth"] for d in no_tool_positives)),
        "risk_epsilon": RISK_EPSILON,
        "trajectories": len(decisions),
        "variants": len(paired),
        "categories": dict(categories),
        "per_policy": {
            policy: {
                "n": sum(1 for d in decisions if d["policy"] == policy),
                "positive": sum(1 for d in decisions
                                if d["policy"] == policy and d["category"] == "positive"),
                "mean_cost_delta": mean([d["cost_delta"] for d in decisions
                                         if d["policy"] == policy]),
            }
            for policy in sorted({d["policy"] for d in decisions})
        },
        "admitted_mean_cost_delta": mean([d["cost_delta"] for d in admitted]),
        "rejected_mean_cost_delta": mean([d["cost_delta"] for d in rejected]),
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories-dir", default=TRAJECTORIES_DIR)
    parser.add_argument("--output", default=REPORT_PATH)
    parser.add_argument("--threshold", type=float, default=APPLICABILITY_THRESHOLD)
    args = parser.parse_args()

    records = load_trajectories(args.trajectories_dir)
    if not records:
        raise SystemExit(f"no trajectories in {args.trajectories_dir}")
    print(f"trajectories: {len(records)}")

    report = run(records, load_applicability(), args.threshold)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"variants: {report['variants']}  categories: {report['categories']}")
    for policy, summary in report["per_policy"].items():
        print(f"  {policy:16s} n={summary['n']:3d} positive={summary['positive']:3d} "
              f"mean Δcost={summary['mean_cost_delta']}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
