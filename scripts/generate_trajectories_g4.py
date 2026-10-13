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
from utils.batch import (LEGACY_BATCH, assert_not_frozen_batch, batch_paths,
                         write_batch_versions)
from mllm.message_builder import (TRAINING_BASELINE_PROMPT,
                                   build_forensic_prompt)
from mllm.mock_client import MockMLLMClient
from scripts.qwen_gain_baseline import _pseudo_probability
from state_machine.controller import ForensicStateMachine
from utils import config_fingerprint
from utils.final_v2 import RECORD_SCHEMA
from utils.logger import SessionLogger

VARIANTS_MANIFEST = os.path.join(PROJECT_ROOT, "sft_data", "variants", "manifest.json")
TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories")
REPORT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_report.json")


def report_path_for(output_dir: str, stamp: Optional[str] = None) -> str:
    """
    One report per run, inside the batch it describes.

    It used to be a fixed public path, so generating a batch overwrote the 237
    set's report — the numbers survived in git, but a reader of either batch
    would have been reading the other one's summary.  A fixed name *inside* the
    batch has the same defect one level down: regenerating a single arm (the
    no-tool arm was regenerated after the observation fix) overwrote the whole
    batch's summary with that arm's.  Hence the run stamp.
    """
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"report_{stamp}.json")
    if not os.path.exists(base):
        return base
    # Two runs inside the same second must not share a file.
    index = 2
    while os.path.exists(os.path.join(output_dir, f"report_{stamp}-{index}.json")):
        index += 1
    return os.path.join(output_dir, f"report_{stamp}-{index}.json")
G2B_REPORT = os.path.join(PROJECT_ROOT, "calibration", "g2_expert_report.json")
G7_APPLICABILITY = os.path.join(PROJECT_ROOT, "calibration", "g7_expert_report.json")

# The expert each tool key dispatches to, and how the policy names itself.
EXPERT_BY_TOOL = {
    "freq": "frequency_expert_v2",
    "noise": "noise_expert",
    "jpeg": "jpeg_expert",
    # G7: the primary detector.  Policies that serve it require it, so a
    # treatment where PROBE was not measured as separating produces no such
    # trajectory rather than a trajectory built on a tool that should not
    # have been served.
    "probe": "probe_dino_expert",
}

TOOL_POLICIES: Dict[str, dict] = {
    "no-tool": {"tools": (), "client": "baseline", "allow_exploration": False},
    "noise": {"tools": ("noise",), "client": "forensic"},
    "jpeg": {"tools": ("jpeg",), "client": "forensic", "requires": "jpeg"},
    "frequency": {"tools": ("freq",), "client": "forensic"},
    "noise+jpeg": {"tools": ("noise", "jpeg"), "client": "forensic", "requires": "jpeg"},
    "noise+frequency": {"tools": ("noise", "freq"), "client": "forensic"},
    "probe": {"tools": ("probe",), "client": "forensic", "requires": "probe"},
    "probe+noise": {"tools": ("probe", "noise"), "client": "forensic", "requires": "probe"},
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

    # G7: external experts live in their own report (they were measured on the
    # G7 fitting group, not on the G2 calibration run), same shape, same gate.
    if os.path.exists(G7_APPLICABILITY):
        with open(G7_APPLICABILITY, encoding="utf-8") as handle:
            external = json.load(handle)
        for name, entry in external.get("experts", {}).items():
            cells = {}
            for treatment, metrics in (entry.get("cell_metrics") or {}).items():
                auroc = metrics.get("auroc_fake_vs_real")
                if auroc is not None:
                    cells[treatment] = float(auroc)
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
    expert = {"freq": "frequency_v2", "noise": "noise", "jpeg": "jpeg",
              "probe": "probe_dino"}.get(tool)
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
    if "probe" in tools:
        from experts.probe_dino import ProbeDinoExpert
        registry["probe"] = ProbeDinoExpert
    return {EXPERT_BY_TOOL[tool]: registry[tool]() for tool in tools}


def prompt_switching_client(client):
    """
    One model for the whole run, with the system prompt swapped per policy.

    Both obvious alternatives were tried on this batch and both are wrong: a
    client per session pays a 12 s load for a 12 s session, and a client per
    tool set keeps several 16.7 GB models resident and runs the 24 GB card out
    of memory.  The client reads its prompt only when it builds messages, so
    one instance with the prompt changed underneath it is what fits both.
    """
    def factory(protocol: str, tools: Tuple[str, ...]):
        client.system_prompt = (TRAINING_BASELINE_PROMPT if protocol == "baseline"
                                else build_forensic_prompt(tools))
        return client

    return factory


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
        # The variant knows its format cell: the calibration weights then come
        # from that cell's own measurement (G7 A2), not from the png cell.
        condition=variant.get("treatment"),
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
        # Was the label the model's own conclusion, or the fallback for one
        # that never parsed?  Reports must not merge the two (§4.17 R5).
        "verdict_status": verdict.get("verdict_status"),
        "reference": verdict.get("reference"),
        # §4.17 R3: what the evidence ledger said about the closing turn —
        # carried here so the conversion step can refuse a session whose
        # fabrication survived its one correction instead of rendering it away.
        # The schema this record was written under: the conversion gate
        # requires the ledger fields from `g7_r4` on, and treats absent keys
        # in *older* records as legacy rather than as failures.
        "record_schema": RECORD_SCHEMA,
        "evidence_check": verdict.get("evidence_check"),
        # §4.17 R4: which turn the adopted conclusion came from, whether it
        # parsed, and whether any call in it was left unexecuted — the
        # conversion gate reads these before rendering a single sample.
        "conclusion": verdict.get("conclusion"),
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
        # The trace's own counter block, carried through as it is.  The
        # record used to keep five of its keys, so `skipped_duplicate_calls`
        # (41 of these 56 sessions had one) and `protocol_violations` were
        # simply absent — and a report reading them saw 0.  A missing key must
        # never look like a zero, so nothing is dropped.
        "counters": {
            **metadata.get("counters", metadata),
            "model_turns": result["model_turn_count"],
            "expert_calls": result["expert_call_count"],
            "unique_evidence": result["unique_evidence_count"],
            "weighted_cost": result["weighted_cost"],
            # `expert_calls` is the *executed* count; the requests the session
            # made are executions + skips.  Both names travel together so a
            # reader cannot mistake one for the other.
            "repeat_requests": (result["skipped_duplicate_calls"]
                                + result["suppressed_duplicate_count"]),
            "requests": (result["expert_call_count"]
                         + result["skipped_duplicate_calls"]),
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
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--policies", nargs="*", default=list(TOOL_POLICIES))
    parser.add_argument("--limit", type=int, default=None,
                        help="number of variants to process")
    parser.add_argument("--dry-run", action="store_true",
                        help="MockMLLMClient (CPU plumbing validation only)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch", default=LEGACY_BATCH)
    parser.add_argument("--allow-frozen", action="store_true",
                        help="只用于分析/重放：允许向冻结批次补写轨迹")
    args = parser.parse_args()
    # A frozen batch is a record of what was measured: adding sessions to it
    # would make the record say something that never happened in one run.
    assert_not_frozen_batch(args.batch, "生成新轨迹", allow=args.allow_frozen)
    paths = batch_paths(args.batch)
    if args.manifest is None:
        args.manifest = paths["variants"]
    if args.output_dir is None:
        args.output_dir = paths["trajectories"]

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

        # One instance, built with the baseline prompt and re-prompted per
        # policy by the factory below.
        client_factory = prompt_switching_client(
            QwenVLClient(system_prompt=TRAINING_BASELINE_PROMPT))
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
    report["policies"] = list(args.policies)
    report["batch"] = args.batch
    with open(report_path_for(args.output_dir), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    # §4.17 工程收口: the four versions travel with the data, not only inside
    # one report: code, policy, prompt digest, expert specs, calibration.
    versions_path = write_batch_versions(
        args.output_dir, experts=build_experts(("freq", "noise", "jpeg")),
        extra={"batch": args.batch, "policies": list(args.policies), "mode": mode,
               "report": os.path.basename(report_path_for(args.output_dir))})
    print(f"versions: {versions_path}")
    print(f"written: {len(records)}  already present: {done}")
    if skipped:
        print(f"gated out example: {skipped[0]['reason']}")
    print(f"Report: {report_path_for(args.output_dir, report['generated_at'])}")


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
