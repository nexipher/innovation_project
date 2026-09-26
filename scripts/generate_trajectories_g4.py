#!/usr/bin/env python3
"""
G4-b: candidate trajectories — one session per (variant, tool policy).

A training sample is not "the model's answer"; it is what the pipeline does
under a *stated* tool policy.  The same image is therefore run under several
policies, and each run is recorded whole:

  no-tool            the baseline judgement, no tools served
  noise              the expert that separates in every format
  jpeg               only where the calibration measured it as applicable
  frequency          weak evidence, kept to teach corroboration
  noise+jpeg         orthogonal pair
  noise+frequency    orthogonal pair

Every record carries the original model judgement, each Evidence Token with its
calibrated likelihood and applicability, the halting policy's posterior and
reasons, the final verdict and confidence, the session counters, and NLL/Brier
scores for both the model's own probability and the structured posterior — so
G4-c can decide admission from the record alone.

The jpeg-applicability gate is read from the G2-b calibration (`cell_metrics`,
polarity-corrected): the expert is served only where it was measured to
separate, which excludes the q70 cell (corrected separation 0.431).

CPU: assembly, gating, scoring, resume and the dry run.  GPU: the sessions
themselves (`--dry-run` uses the mock client).

Usage:
  python scripts/generate_trajectories_g4.py --dry-run --limit 1
  python scripts/generate_trajectories_g4.py --limit 40 --policies no-tool noise
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from mllm.message_builder import BASELINE_SYSTEM_PROMPT, build_forensic_prompt
from mllm.mock_client import MockMLLMClient
from scripts.qwen_gain_baseline import _pseudo_probability
from state_machine.controller import ForensicStateMachine
from utils import config_fingerprint
from utils.logger import SessionLogger

VARIANTS_MANIFEST = os.path.join(PROJECT_ROOT, "sft_data", "variants", "manifest.json")
TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories")
REPORT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_report.json")
G2B_REPORT = os.path.join(PROJECT_ROOT, "calibration", "g2_expert_report.json")

# The expert each tool key dispatches to, and how the policy names itself.
EXPERT_BY_TOOL = {
    "freq": "frequency_expert_v2",
    "noise": "noise_expert",
    "jpeg": "jpeg_expert",
}

TOOL_POLICIES: Dict[str, dict] = {
    "no-tool": {"tools": (), "client": "baseline", "allow_exploration": False},
    "noise": {"tools": ("noise",), "client": "forensic"},
    "jpeg": {"tools": ("jpeg",), "client": "forensic", "requires": "jpeg"},
    "frequency": {"tools": ("freq",), "client": "forensic"},
    "noise+jpeg": {"tools": ("noise", "jpeg"), "client": "forensic", "requires": "jpeg"},
    "noise+frequency": {"tools": ("noise", "freq"), "client": "forensic"},
}

# A tool is served only where the calibration measured it as separating.  0.65
# is the same bar the G4-c admission rules use, so generation and admission
# cannot disagree about what "applicable" means.
APPLICABILITY_THRESHOLD = 0.65


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------

def load_applicability(report_path: str = G2B_REPORT) -> Dict[str, Dict[str, float]]:
    """
    Per-expert, per-cell polarity-corrected separation.

    The report's `auroc_fake_vs_real` is computed for "fake has the higher
    metric"; for an expert whose metric runs the other way, the separation it
    can actually offer is 1 - auroc.
    """
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    table: Dict[str, Dict[str, float]] = {}
    for name, entry in report.get("experts", {}).items():
        inverted = entry.get("semantics_aligned") is False
        cells = {}
        for cell, metrics in (entry.get("cell_metrics") or {}).items():
            auroc = metrics.get("auroc_fake_vs_real")
            if auroc is None:
                continue
            cells[cell] = float(1.0 - auroc if inverted else auroc)
        table[name] = cells
    return table


def tool_applicable(tool: str, treatment: str,
                    applicability: Dict[str, Dict[str, float]],
                    threshold: float = APPLICABILITY_THRESHOLD) -> Tuple[bool, str]:
    """
    May this tool be served on this treatment, and why (not)?

    Unknown experts or cells are refused rather than assumed safe: serving a
    tool whose separability was never measured is how a lucky guess becomes a
    training label.
    """
    expert = {"freq": "frequency_v2", "noise": "noise", "jpeg": "jpeg"}.get(tool)
    if expert is None:
        return False, f"unknown tool {tool!r}"
    cells = applicability.get(expert)
    if not cells:
        return False, f"{expert}: no calibration entry"
    separation = cells.get(treatment)
    if separation is None:
        return False, f"{expert}: no measurement for cell {treatment}"
    if separation < threshold:
        return False, f"{expert}: separation {separation:.3f} < {threshold} in {treatment}"
    return True, f"{expert}: separation {separation:.3f} in {treatment}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _nll(probability_fake: float, is_fake: bool) -> float:
    p = probability_fake if is_fake else 1.0 - probability_fake
    return -math.log(max(p, 1e-6))


def _brier(probability_fake: float, is_fake: bool) -> float:
    return (probability_fake - (1.0 if is_fake else 0.0)) ** 2


def score_record(record: dict) -> dict:
    """
    NLL/Brier for the model's own probability and for the structured posterior.

    Two numbers rather than one: the model's confidence is what it said, the
    posterior is what the evidence supported, and G4-c's gain test needs to
    compare them against the no-tool run.
    """
    is_fake = record["ground_truth"] == "Fake"
    scores = {}
    model_probability = record.get("model_probability")
    if model_probability is not None:
        scores["model_nll"] = round(_nll(model_probability, is_fake), 4)
        scores["model_brier"] = round(_brier(model_probability, is_fake), 4)
    posterior = record.get("posterior")
    if posterior is not None:
        scores["posterior_nll"] = round(_nll(posterior, is_fake), 4)
        scores["posterior_brier"] = round(_brier(posterior, is_fake), 4)
        scores["posterior_error"] = round(abs(posterior - (1.0 if is_fake else 0.0)), 4)
    return scores


# ---------------------------------------------------------------------------
# Session running
# ---------------------------------------------------------------------------

def build_experts(tools: Tuple[str, ...]) -> Dict[str, Any]:
    """Only the experts this policy serves — an unserved call finds no expert."""
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.jpeg import JPEGExpert
    from experts.noise import NoiseExpert

    registry = {"freq": FrequencyExpertV2, "noise": NoiseExpert, "jpeg": JPEGExpert}
    return {EXPERT_BY_TOOL[tool]: registry[tool]() for tool in tools}


def run_trajectory(variant: dict, policy_name: str, policy: dict,
                   client_factory: Callable[[str, Tuple[str, ...]], Any],
                   sft_dir: Optional[str] = None) -> dict:
    """Run one variant under one tool policy and return its record."""
    tools = tuple(policy["tools"])
    image_path = os.path.join(PROJECT_ROOT, variant["path"])
    client = client_factory(policy["client"], tools)
    fsm = ForensicStateMachine(
        client,
        build_experts(tools),
        logger=SessionLogger(sft_dir=sft_dir) if sft_dir else None,
        evidence_injection="text+image",
        allow_exploration=policy.get("allow_exploration", True),
    )

    started = time.perf_counter()
    result = fsm.run(image_path, ground_truth=variant["label"])
    elapsed = time.perf_counter() - started

    verdict = result["final_verdict"]
    with open(result["sft_data_path"], encoding="utf-8") as handle:
        trace = json.load(handle)
    metadata = trace.get("metadata", {})

    record = {
        "trajectory_id": f"{policy_name}__{variant['variant_id']}",
        "policy": policy_name,
        "tools_served": list(tools),
        "variant_id": variant["variant_id"],
        "source_id": variant["source_id"],
        "split": variant["split"],
        "generator": variant["generator"],
        "treatment": variant["treatment"],
        "container": variant["container"],
        "ground_truth": variant["label"],
        "model_candidate": verdict.get("model_candidate"),
        "final_verdict": verdict.get("verdict"),
        "confidence": verdict.get("confidence"),
        "report": verdict.get("report"),
        "posterior": verdict.get("posterior"),
        "conflict_score": verdict.get("conflict_score"),
        "policy_reasons": verdict.get("policy_reasons"),
        # The model's own probability, or None when it never proposed a
        # verdict: the policy's label is not the model's opinion.
        "model_probability": (
            round(_pseudo_probability({
                "verdict": verdict["model_candidate"],
                "confidence": verdict.get("confidence"),
            }), 4)
            if verdict.get("model_candidate") else None
        ),
        "evidence": result["evidence_chain"],
        "counters": {
            "model_turns": result["model_turn_count"],
            "expert_calls": result["expert_call_count"],
            "unique_evidence": result["unique_evidence_count"],
            "suppressed_duplicates": result["suppressed_duplicate_count"],
            "weighted_cost": result["weighted_cost"],
        },
        "halting_reason": result["halting_reason"],
        "elapsed_s": round(elapsed, 2),
        "trace_path": os.path.relpath(result["sft_data_path"], PROJECT_ROOT).replace(os.sep, "/"),
        "policy_version": metadata.get("policy_version"),
    }
    record["scores"] = score_record(record)
    return record


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def plan_trajectories(variants: Dict[str, dict], policies: List[str],
                      applicability: Dict[str, Dict[str, float]],
                      threshold: float = APPLICABILITY_THRESHOLD) -> Tuple[List[Tuple[dict, str]], List[dict]]:
    """
    Which (variant, policy) pairs to run, and which were gated out.

    A policy requiring an applicable tool is skipped — with the reason — for
    every treatment where that tool was not measured to separate.
    """
    planned: List[Tuple[dict, str]] = []
    skipped: List[dict] = []

    for variant in variants.values():
        for policy_name in policies:
            policy = TOOL_POLICIES[policy_name]
            # `requires` names the tool that must be applicable for this
            # policy to be generated at all; the policy's other tools are
            # served unconditionally.
            requirement = policy.get("requires")
            if requirement:
                ok, reason = tool_applicable(requirement, variant["treatment"],
                                             applicability, threshold)
                if not ok:
                    skipped.append({
                        "variant_id": variant["variant_id"],
                        "policy": policy_name,
                        "reason": reason,
                    })
                    continue
            planned.append((variant, policy_name))
    return planned, skipped


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=VARIANTS_MANIFEST)
    parser.add_argument("--policies", nargs="*", default=list(TOOL_POLICIES))
    parser.add_argument("--limit", type=int, default=None,
                        help="number of variants to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="MockMLLMClient (CPU plumbing validation only)")
    parser.add_argument("--output-dir", default=TRAJECTORIES_DIR)
    args = parser.parse_args()

    unknown = [p for p in args.policies if p not in TOOL_POLICIES]
    if unknown:
        raise SystemExit(f"unknown policies: {unknown}")

    with open(args.manifest, encoding="utf-8") as handle:
        manifest = json.load(handle)
    # The manifest is keyed by the variant's project-relative path; carry it in
    # the value so a record is self-contained.
    variants = {path: {**entry, "path": path}
                for path, entry in manifest["variants"].items()}
    if args.limit:
        variants = dict(list(variants.items())[:args.limit])

    applicability = load_applicability()
    planned, skipped = plan_trajectories(variants, args.policies, applicability)
    print(f"variants: {len(variants)}  policies: {len(args.policies)}  "
          f"planned: {len(planned)}  gated out: {len(skipped)}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.dry_run:
        def client_factory(variant, tools):
            return MockMLLMClient(mode="two_calls", seed=42)
        mode = "dry_run"
        sft_dir = os.path.join(PROJECT_ROOT, "traces", "dry_run_sessions")
    else:
        from mllm.qwen_client import QwenVLClient

        def client_factory(variant, tools):
            prompt = (BASELINE_SYSTEM_PROMPT if variant == "baseline"
                      else build_forensic_prompt(tools))
            return QwenVLClient(system_prompt=prompt)
        mode = "gpu"
        sft_dir = None

    records, done = [], 0
    for variant, policy_name in planned:
        path = os.path.join(args.output_dir, f"{variant['variant_id']}__{policy_name}.json")
        if os.path.exists(path):
            done += 1
            continue
        record = run_trajectory(variant, policy_name, TOOL_POLICIES[policy_name],
                                client_factory, sft_dir=sft_dir)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        records.append(record)
        if len(records) % 20 == 0:
            print(f"  {len(records)} trajectories written")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "config_fingerprint": config_fingerprint.compute(build_experts(("freq", "noise", "jpeg"))),
        "applicability_threshold": APPLICABILITY_THRESHOLD,
        "planned": len(planned),
        "already_present": done,
        "written": len(records),
        "gated_out": skipped,
        "aggregate": aggregate(records) if records else {},
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"written: {len(records)}  already present: {done}")
    if skipped:
        print(f"gated out example: {skipped[0]['reason']}")
    print(f"Report: {REPORT_PATH}")


def aggregate(records: List[dict]) -> dict:
    """Counts per policy/treatment plus mean scores, for a quick read."""
    by_policy: Dict[str, List[dict]] = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)

    summary = {}
    for policy, rows in sorted(by_policy.items()):
        summary[policy] = {
            "n": len(rows),
            "uncertain_rate": round(
                sum(1 for r in rows if r["final_verdict"] == "Uncertain") / len(rows), 4),
            "mean_calls": round(sum(r["counters"]["expert_calls"] for r in rows) / len(rows), 3),
            "mean_turns": round(sum(r["counters"]["model_turns"] for r in rows) / len(rows), 3),
            "mean_posterior_brier": round(
                sum(r["scores"].get("posterior_brier", 0.0) for r in rows) / len(rows), 4),
            "mean_model_brier": round(
                sum(r["scores"].get("model_brier", 0.0) for r in rows) / len(rows), 4),
        }
    return summary


if __name__ == "__main__":
    main()
