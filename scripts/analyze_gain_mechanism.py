#!/usr/bin/env python3
"""
G2-d mechanism analysis: why the evidence arms lost.

Companion to `scripts/analyze_g2_gain.py`, which owns the inferential
statistics (bootstrap AUROC intervals, delta-AUROC, per-class recall).  This
script answers the mechanism question that decided G2-e:

  Does the model follow the direction the token claims, and does the token
  claim agree with the calibration table?

It reconstructs each session's injection arm from the trace conversation,
then measures:

  * per-cell accuracy and arm-vs-arm McNemar (the committed script only
    compares against the RGB baseline);
  * the Fake rate conditioned on the token's `support` claim;
  * the mean calibrated P(Fake) behind each claim — a large gap means the
    Bundle contradicts itself, telling the model two opposite things;
  * whether the verdict in the visual-only arm tracks the raw expert metric
    that G2-b showed to be inverted.

Output: console tables plus `calibration/g2_gain_mechanism.json`.

Usage:
  python scripts/analyze_gain_mechanism.py
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.qwen_gain_baseline import compute_metrics

REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_gain_report.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_gain_mechanism.json")
SESSIONS_DIR = os.path.join(PROJECT_ROOT, "traces", "sft_sessions")

ARMS = ("rgb", "text", "image", "both")
TRACE_PATTERN = re.compile(r"forensic_sft_session_(\d{8}_\d{6})_(.+)\.json$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_report(path: str = REPORT_PATH) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def classify_arm(conversation: List[dict]) -> str:
    """
    Which injection mode produced this trace, judged from what the model saw.

    The trace does not record the mode, so it is reconstructed: no evidence
    turn means the RGB arm; a JSON turn means the text was shown, and the
    attached artifacts then say whether the visuals were too.
    """
    evidence_turns = [
        turn for turn in conversation
        if turn.get("from") == "user" and "<image>" not in turn.get("value", "")
    ]
    if not evidence_turns:
        return "rgb"
    first = evidence_turns[0]["value"].lstrip()
    if first.startswith("{"):
        return "both" if evidence_turns[0].get("image_paths") else "text"
    return "image"


def load_traces(sessions_dir: str = SESSIONS_DIR) -> Dict[Tuple[str, str], dict]:
    """
    Map (arm, sample_id) -> the latest trace for it.

    The sample_id is the calibration image stem baked into the session id, so
    a trace can be matched to a report record directly.  Latest wins: a smoke
    run may have recorded the same sample before the full run did.
    """
    traces: Dict[Tuple[str, str], Tuple[str, dict]] = {}
    for path in glob.glob(os.path.join(sessions_dir, "*.json")):
        match = TRACE_PATTERN.search(os.path.basename(path))
        if not match:
            continue
        timestamp, sample_id = match.groups()
        try:
            with open(path, encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, ValueError):
            continue
        if "calibration/set/images" not in trace.get("image_path", ""):
            continue
        key = (classify_arm(trace.get("conversations", [])), sample_id)
        if key not in traces or timestamp > traces[key][0]:
            traces[key] = (timestamp, trace)
    return {key: trace for key, (_, trace) in traces.items()}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def per_cell_table(records: Dict[str, Dict[str, dict]]) -> Dict[str, Dict[str, str]]:
    cells = sorted({r["cell"] for r in records["rgb"].values()})
    table: Dict[str, Dict[str, str]] = {}
    for cell in cells:
        row = {}
        for arm in ARMS:
            subset = [r for r in records[arm].values() if r["cell"] == cell]
            hits = sum(1 for r in subset if r["verdict"] == r["gt"])
            row[arm] = f"{hits}/{len(subset)}"
        table[cell] = row
    return table


def mcnemar_vs_baseline(records: Dict[str, Dict[str, dict]]) -> Dict[str, dict]:
    """Paired comparison against the RGB arm (same samples, same order)."""
    baseline = records["rgb"]
    out: Dict[str, dict] = {}
    for arm in ARMS[1:]:
        baseline_wins = new_wins = 0
        for sample_id, base in baseline.items():
            other = records[arm].get(sample_id)
            if other is None:
                continue
            base_ok = base["verdict"] == base["gt"]
            other_ok = other["verdict"] == other["gt"]
            if base_ok and not other_ok:
                baseline_wins += 1
            elif other_ok and not base_ok:
                new_wins += 1
        discordant = baseline_wins + new_wins
        p_value = (binomtest(min(baseline_wins, new_wins), discordant, 0.5).pvalue * 2
                   if discordant else 1.0)
        out[arm] = {
            "baseline_only_correct": baseline_wins,
            "arm_only_correct": new_wins,
            "p_value": round(float(min(p_value, 1.0)), 5),
            "significant": bool(p_value < 0.05),
        }
    return out


def evidence_following(traces: Dict[Tuple[str, str], dict]) -> Dict[str, dict]:
    """
    Does a verdict track the direction the token claimed?

    `support` is the expert's own claim; `calibrated_likelihood.Fake` is the
    calibration table's probability.  An instructive model should raise its
    Fake rate when either points at Fake.
    """
    per_arm: Dict[str, dict] = {}
    for arm in ("text", "both"):
        by_support: Dict[str, Counter] = defaultdict(Counter)
        pairs: List[Tuple[float, int]] = []  # (P(Fake) from table, verdict is Fake)
        for (trace_arm, _), trace in traces.items():
            if trace_arm != arm:
                continue
            verdict = (trace.get("final_verdict") or {}).get("verdict")
            chain = trace.get("evidence_chain") or []
            if not chain or verdict is None:
                continue
            # A session may hold several tokens; use the strongest claim.
            claims = [t for t in chain if t.get("support") in ("Real", "AI-generated")]
            claim = max(claims, key=lambda t: t.get("strength", 0.0), default=chain[0])
            likelihood = (claim.get("calibrated_likelihood") or {}).get("Fake")
            if likelihood is not None:
                pairs.append((likelihood, int(verdict == "Fake")))
            if claim.get("support"):
                by_support[claim["support"]][verdict] += 1

        entry = {
            "by_support": {
                support: {"n": sum(counts.values()), "fake_rate": _fake_rate(counts)}
                for support, counts in by_support.items()
            },
            "support_vs_calibration": _support_versus_calibration(traces, arm),
        }
        if len(pairs) >= 10:
            probabilities = np.array([p for p, _ in pairs])
            labels = np.array([y for _, y in pairs])
            entry["likelihood_correlation"] = {
                "n": len(pairs),
                "mean_p_fake_when_verdict_fake": float(probabilities[labels == 1].mean()) if labels.any() else None,
                "mean_p_fake_when_verdict_real": float(probabilities[labels == 0].mean()) if (~labels.astype(bool)).any() else None,
            }
        per_arm[arm] = entry
    return per_arm


def _support_versus_calibration(traces: Dict[Tuple[str, str], dict], arm: str) -> dict:
    """
    Do the two direction fields in the Bundle agree?

    `support` is the expert's own claim; `calibrated_likelihood` is what the
    calibration table measured after correcting the inversion.  If they
    disagree, the Bundle tells the model two opposite things and whichever
    field it happens to trust decides the verdict.
    """
    by_support: Dict[str, List[float]] = defaultdict(list)
    for (trace_arm, _), trace in traces.items():
        if trace_arm != arm:
            continue
        for token in trace.get("evidence_chain") or []:
            likelihood = (token.get("calibrated_likelihood") or {}).get("Fake")
            if token.get("support") and likelihood is not None:
                by_support[token["support"]].append(float(likelihood))
    return {
        support: {"n": len(values), "mean_calibrated_p_fake": round(float(np.mean(values)), 3)}
        for support, values in sorted(by_support.items())
    }


def _fake_rate(counts: Counter) -> float:
    total = sum(counts.values())
    return round(counts["Fake"] / total, 3) if total else 0.0


def visual_channel_leakage(traces: Dict[Tuple[str, str], dict]) -> Dict[str, dict]:
    """
    In the visual-only arm the model sees artifacts, never numbers.

    If its Fake calls rise with an expert's raw metric — and G2-b showed high
    metric means Real for noise/jpeg/ELA — then the inverted semantics leak
    through the image channel instead of the text channel.
    """
    per_source: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"fake": [], "not_fake": []})
    for (arm, _), trace in traces.items():
        if arm != "image":
            continue
        verdict = (trace.get("final_verdict") or {}).get("verdict")
        if verdict is None or verdict == "Uncertain":
            continue
        for token in trace.get("evidence_chain") or []:
            source = token.get("source")
            metric = token.get("raw_metric")
            if source is None or metric is None:
                continue
            bucket = "fake" if verdict == "Fake" else "not_fake"
            per_source[source][bucket].append(float(metric))

    out: Dict[str, dict] = {}
    for source, buckets in sorted(per_source.items()):
        fake, other = buckets["fake"], buckets["not_fake"]
        if not fake or not other:
            out[source] = {"n_fake": len(fake), "n_not_fake": len(other),
                           "mean_metric_fake": round(float(np.mean(fake)), 4) if fake else None,
                           "mean_metric_not_fake": round(float(np.mean(other)), 4) if other else None,
                           "direction": "insufficient"}
            continue
        delta = float(np.mean(fake) - np.mean(other))
        out[source] = {
            "n_fake": len(fake),
            "n_not_fake": len(other),
            "mean_metric_fake": round(float(np.mean(fake)), 4),
            "mean_metric_not_fake": round(float(np.mean(other)), 4),
            "delta": round(delta, 4),
            "direction": "fake_calls_have_higher_metric" if delta > 0
                         else "fake_calls_have_lower_metric",
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    report = load_report()
    traces = load_traces()
    print(f"Report: {report['mode']} / per_cell {report['per_cell']}")
    print(f"Traces matched: {len(traces)}")

    records: Dict[str, Dict[str, dict]] = {
        arm: {r["sample_id"]: r for r in report["conditions"].get(arm, {}).get("records", [])}
        for arm in ARMS
    }
    for arm in ARMS:
        print(f"  {arm:6s} records={len(records[arm]):3d}  traces={sum(1 for (a, _) in traces if a == arm):3d}")

    analysis = {
        "metrics": {arm: compute_metrics(list(records[arm].values())) for arm in ARMS},
        "verdict_distribution": {
            arm: dict(Counter(r["verdict"] for r in records[arm].values()))
            for arm in ARMS
        },
        "per_cell": per_cell_table(records),
        "mcnemar_vs_rgb": mcnemar_vs_baseline(records),
        "evidence_following": evidence_following(traces),
        "visual_channel_leakage": visual_channel_leakage(traces),
    }

    _print_tables(analysis)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(analysis, handle, ensure_ascii=False, indent=2)
    print(f"\nAnalysis: {OUTPUT_PATH}")


def _print_tables(analysis: dict) -> None:
    print(f"\n{'arm':6s} {'n':>4s} {'acc':>6s} {'f1':>6s} {'auroc':>6s} {'ece':>6s} "
          f"{'unc':>6s} {'turns':>6s} {'calls':>6s} {'sec':>6s}")
    for arm, metrics in analysis["metrics"].items():
        print(f"{arm:6s} {metrics['n']:4d} {metrics['accuracy']:6.3f} {metrics['f1_fake']:6.3f} "
              f"{metrics['auroc'] if metrics['auroc'] is not None else float('nan'):6.3f} "
              f"{metrics['ece'] if metrics['ece'] is not None else float('nan'):6.3f} "
              f"{metrics['uncertain_rate']:6.3f} {metrics['avg_model_turns']:6.1f} "
              f"{metrics['avg_expert_calls']:6.1f}")

    print("\nverdict distribution")
    for arm, counts in analysis["verdict_distribution"].items():
        print(f"  {arm:6s} " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    print("\nper-cell accuracy")
    cells = list(analysis["per_cell"])
    print(f"  {'cell':16s}" + "".join(f"{arm:>8s}" for arm in ARMS))
    for cell in cells:
        row = analysis["per_cell"][cell]
        print(f"  {cell:16s}" + "".join(f"{row[arm]:>8s}" for arm in ARMS))

    print("\nMcNemar vs rgb")
    for arm, entry in analysis["mcnemar_vs_rgb"].items():
        verdict = "worse" if entry["baseline_only_correct"] > entry["arm_only_correct"] else "better"
        print(f"  rgb→{arm:6s} baseline-only={entry['baseline_only_correct']:3d} "
              f"arm-only={entry['arm_only_correct']:3d} p={entry['p_value']:.4f} "
              f"({verdict}{'*, significant' if entry['significant'] else ''})")

    print("\nevidence following (text / both)")
    for arm, entry in analysis["evidence_following"].items():
        for support, stats in entry.get("by_support", {}).items():
            print(f"  {arm:6s} token says {support:13s} n={stats['n']:3d} P(verdict Fake)={stats['fake_rate']:.3f}")
        corr = entry.get("likelihood_correlation")
        if corr:
            print(f"  {arm:6s} mean table P(Fake): verdict=Fake {corr['mean_p_fake_when_verdict_fake']} "
                  f"vs verdict=Real {corr['mean_p_fake_when_verdict_real']}")
        for support, stats in entry.get("support_vs_calibration", {}).items():
            print(f"  {arm:6s} support={support:13s} → mean calibrated P(Fake)={stats['mean_calibrated_p_fake']} "
                  f"(n={stats['n']})")

    print("\nvisual-channel leakage (image arm): raw metric vs Fake call")
    for source, entry in analysis["visual_channel_leakage"].items():
        print(f"  {source:18s} {entry['direction']:34s} "
              f"metric(fake)={entry['mean_metric_fake']} metric(other)={entry['mean_metric_not_fake']} "
              f"n={entry['n_fake']}/{entry['n_not_fake']}")


if __name__ == "__main__":
    main()
