#!/usr/bin/env python3
"""
Offline halting replay: v1 as recorded vs v2 on the same evidence (G3-c).

The v1/v2 comparison a policy change needs.  Historical traces recorded what
v1 concluded and why; this script re-reads each trace's final evidence chain,
rectifies it the way the pipeline now does (G3-b), and asks what halting
policy v2 would have produced from exactly that evidence.

What this can and cannot say:

  * it CAN compare labels, confidence calibration (ECE / Brier) and the
    override behaviour on the *same* evidence, and it can classify each old
    stop as "v2 would have agreed / abstained / overridden the model";
  * it CANNOT measure how many calls v2 would have saved, because that
    depends on what the model would do after a different halting decision —
    an offline replay has no model in it.  Call and turn counts are reported
    as recorded under v1, not as counterfactuals, and the reason mix reflects
    a session that already ended (no further calls are priced in).

Output: console tables plus `calibration/g3_halting_replay.json`.

Usage:
  python scripts/replay_halting_g3.py
  python scripts/replay_halting_g3.py --limit 200
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from state_machine.evidence_rectifier import EvidenceRectifier
from state_machine.halting_v2 import HaltingPolicyV2
from utils.parser import Parser
from utils.reliability import ReliabilityTable

SESSIONS_DIR = os.path.join(PROJECT_ROOT, "traces", "sft_sessions")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g3_halting_replay.json")
TRACE_PATTERN = re.compile(r"forensic_sft_session_(\d{8}_\d{6})_(.+)\.json$")


def load_traces(limit: Optional[int] = None,
                sessions_dir: str = SESSIONS_DIR) -> List[dict]:
    """Recorded Qwen sessions that carry both evidence and a ground truth."""
    traces = []
    for path in sorted(glob.glob(os.path.join(sessions_dir, "*.json"))):
        if not TRACE_PATTERN.search(os.path.basename(path)):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, ValueError):
            continue
        if trace.get("metadata", {}).get("mock_mode") != "qwen_real":
            continue
        if not trace.get("ground_truth") or not trace.get("evidence_chain"):
            continue
        trace["_path"] = path
        traces.append(trace)
    return traces[:limit] if limit else traces


def expert_weights(table: Optional[ReliabilityTable], source_names) -> Dict[str, float]:
    if table is None:
        return {name: 0.0 for name in source_names}
    separations = {}
    for name in source_names:
        entry = table.expert_entry(name) or {}
        separations[name] = entry.get("separation_polarity_corrected")
    return HaltingPolicyV2.expert_weights(separations)


def model_candidate(trace: dict) -> Optional[str]:
    """The verdict the model itself proposed in its closing turn."""
    for turn in reversed(trace.get("conversations", [])):
        if turn.get("from") != "gpt":
            continue
        verdict = Parser.parse_verdict(turn.get("value", ""))
        if verdict and "verdict" in verdict:
            return verdict["verdict"]
    return None


def ece(probabilities: List[float], outcomes: List[int], bins: int = 10) -> Optional[float]:
    if len(probabilities) < 10:
        return None
    probabilities = np.array(probabilities)
    outcomes = np.array(outcomes, dtype=float)
    total = len(probabilities)
    error = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probabilities >= low) & (probabilities < high)
        if index == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        error += (count / total) * abs(outcomes[mask].mean() - probabilities[mask].mean())
    return float(error)


def brier(probabilities: List[float], outcomes: List[int]) -> Optional[float]:
    if not probabilities:
        return None
    probabilities = np.array(probabilities)
    outcomes = np.array(outcomes, dtype=float)
    return float(np.mean((probabilities - outcomes) ** 2))


def replay(traces: List[dict], table: Optional[ReliabilityTable]) -> dict:
    rows = []
    for trace in traces:
        chain = [EvidenceRectifier.rectify(copy.deepcopy(t)) for t in trace["evidence_chain"]]
        sources = {t.get("source") for t in chain if t.get("source")}
        weights = expert_weights(table, sources)
        posterior = HaltingPolicyV2.posterior(chain)
        conflict = HaltingPolicyV2.conflict_score(chain)
        candidate = model_candidate(trace)

        # Every expert that produced evidence was called under v1 too.
        decision = HaltingPolicyV2.decide(
            model_turns=trace["metadata"].get("model_turn_count", 1),
            expert_calls=trace["metadata"].get("expert_call_count", len(chain)),
            evidence_chain=chain,
            candidate_verdict={"verdict": candidate} if candidate else None,
            available_experts=weights,
            called_experts=sources,
            # Neutral: the replay asks what the *evidence* implies, not what a
            # stalled conversation would do.  A stall parameter of 2 here would
            # make model_stalled dominate the reason mix by construction and
            # say nothing about the policy.
            turns_without_new_evidence=0,
        )

        recorded = (trace.get("final_verdict") or {}).get("verdict")
        ground_truth = trace["ground_truth"]
        rows.append({
            "sample": os.path.basename(trace["_path"]),
            "ground_truth": ground_truth,
            "recorded_verdict": recorded,
            "recorded_reason": trace["metadata"].get("halting_reason"),
            "model_candidate": candidate,
            "v2_verdict": decision.verdict,
            "v2_reason": decision.primary_reason,
            "v2_posterior": round(posterior, 4),
            "v2_confidence": decision.confidence,
            "conflict_score": round(conflict, 4),
            "v2_overrides_model": (
                candidate is not None and decision.verdict not in (None, candidate)
                and HaltingPolicyV2.CANDIDATE_OVERRIDDEN in decision.all_reasons
            ),
            "recorded_calls": trace["metadata"].get("expert_call_count", 0),
            "recorded_turns": trace["metadata"].get("model_turn_count", 0),
        })
    return {"rows": rows}


def summarise(rows: List[dict]) -> dict:
    def outcomes():
        """
        The policy's P(Fake) against what was true, over every replayed
        session — abstentions included, since an honest posterior is what
        makes abstention a decision rather than a dodge.
        """
        probabilities = [r["v2_posterior"] for r in rows]
        labels = [int(r["ground_truth"] == "Fake") for r in rows]
        return probabilities, labels

    def accuracy(key):
        scored = [r for r in rows if r[key] in ("Real", "Fake")]
        hits = sum(1 for r in scored if r[key] == r["ground_truth"])
        return (hits / len(scored) if scored else None), len(scored)

    v1_accuracy, v1_n = accuracy("recorded_verdict")
    v2_accuracy, v2_n = accuracy("v2_verdict")
    probabilities, labels = outcomes()

    return {
        "sessions": len(rows),
        "v1": {
            "labelled": v1_n,
            "accuracy": round(v1_accuracy, 4) if v1_accuracy is not None else None,
            "abstained": sum(1 for r in rows if r["recorded_verdict"] == "Uncertain"),
            "reasons": dict(Counter(r["recorded_reason"] for r in rows)),
            "mean_calls": round(float(np.mean([r["recorded_calls"] for r in rows])), 3),
            "mean_turns": round(float(np.mean([r["recorded_turns"] for r in rows])), 3),
        },
        "v2": {
            "labelled": v2_n,
            "accuracy": round(v2_accuracy, 4) if v2_accuracy is not None else None,
            "abstained": sum(1 for r in rows if r["v2_verdict"] == "Uncertain"),
            "reasons": dict(Counter(r["v2_reason"] for r in rows)),
            "ece": ece(probabilities, labels),
            "brier": brier(probabilities, labels),
        },
        "override": {
            "model_candidates": sum(1 for r in rows if r["model_candidate"]),
            "v2_overrides": sum(1 for r in rows if r["v2_overrides_model"]),
            "candidate_would_be_accepted": sum(
                1 for r in rows if r["model_candidate"] and r["v2_verdict"] == r["model_candidate"]
            ),
        },
        "conflict": {
            "sessions_with_conflict": sum(1 for r in rows if r["conflict_score"] > 0.5),
            "v1_labelled_those": sum(
                1 for r in rows
                if r["conflict_score"] > 0.5 and r["recorded_verdict"] in ("Real", "Fake")
            ),
            "v2_abstained_those": sum(
                1 for r in rows
                if r["conflict_score"] > 0.5 and r["v2_verdict"] == "Uncertain"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    table = ReliabilityTable.load()
    if table is None:
        raise SystemExit("no reliability table — v2 needs measured expert weights")

    traces = load_traces(args.limit)
    if not traces:
        raise SystemExit("no recorded Qwen sessions found")

    replay_result = replay(traces, table)
    rows = replay_result["rows"]
    summary = summarise(rows)
    summary["rows"] = rows

    print(f"Sessions replayed: {summary['sessions']}")
    v1, v2 = summary["v1"], summary["v2"]
    print(f"\n{'':6s} {'labelled':>9s} {'accuracy':>9s} {'abstained':>10s} {'ECE':>7s} {'Brier':>7s}")
    print(f"v1     {v1['labelled']:9d} {v1['accuracy'] if v1['accuracy'] is not None else float('nan'):9.3f} "
          f"{v1['abstained']:10d} {'-':>7s} {'-':>7s}")
    print(f"v2     {v2['labelled']:9d} {v2['accuracy'] if v2['accuracy'] is not None else float('nan'):9.3f} "
          f"{v2['abstained']:10d} "
          f"{(v2['ece'] if v2['ece'] is not None else float('nan')):7.3f} "
          f"{(v2['brier'] if v2['brier'] is not None else float('nan')):7.3f}")

    print("\nv1 reasons:", v1["reasons"])
    print("v2 reasons:", v2["reasons"])
    print("v1 cost:   mean calls %.2f, mean turns %.2f" % (v1["mean_calls"], v1["mean_turns"]))
    print("\noverride:", summary["override"])
    print("conflict:", summary["conflict"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"\nReplay: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
