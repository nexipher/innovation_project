#!/usr/bin/env python3
"""
What the v3 acceptance batch actually did (G7 §4.17, R5 段).

Reads the trajectories of one batch and reports, separately and by name:

  合法主动结案   model_concluded — the model ended the session with a legal
                 conclusion, which is the label (this is the v3 contract)
  合法弃权       model_concluded and the label is Uncertain — an abstention
                 the model chose
  解析失败       no_parseable_conclusion — **not** an abstention (§4.17 R5)
  外部截断       budget_exhausted / loop_protection / format_failure — the
                 policy ended the session; the closing turn then had to supply
                 the conclusion
  证据纠正       the one correction the evidence ledger can ask for, and what
                 it found (`evidence_check`), plus candidate findings

It also prints the PROBE-alone reference for the same sessions, computed
offline from the probe token already in each record (no extra GPU work): the
calibrated P(Fake) of PROBE's own score, read through the same table the
session used, so "the model with tools" and "the primary detector alone" can
be compared on identical images.

Usage:
  python scripts/g7_b2_report.py --batch g7_b2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.batch import batch_paths, load_trajectory_records

CONCLUSION = "model_concluded"
TRUNCATIONS = ("budget_exhausted", "loop_protection", "format_failure")


def _probe_reference(record: dict, table) -> Optional[str]:
    """PROBE alone: its own score, read through the calibration table."""
    for token in record.get("evidence") or []:
        if token.get("source") != "probe_dino_expert":
            continue
        calibrated = token.get("calibrated_likelihood") or {}
        fake = calibrated.get("Fake")
        if fake is None:
            continue
        return "Fake" if float(fake) >= 0.5 else "Real"
    return None


def summarise(records: List[dict], table=None) -> dict:
    by_status = Counter(record.get("verdict_status") for record in records)
    by_reason = Counter(record.get("halting_reason") for record in records)
    labels = Counter(record.get("final_verdict") for record in records)

    concluded = [r for r in records if r.get("verdict_status") == CONCLUSION]
    abstained = [r for r in concluded if r.get("final_verdict") == "Uncertain"]
    unparsed = [r for r in records
                if r.get("verdict_status") == "no_parseable_conclusion"]
    truncated = [r for r in records if r.get("halting_reason") in TRUNCATIONS]

    checks = Counter((r.get("evidence_check") or {}).get("status") for r in records)
    defect_kinds = Counter(
        finding.get("kind")
        for record in records
        for finding in ((record.get("evidence_check") or {}).get("findings") or []))
    candidate_kinds = Counter(
        finding.get("kind")
        for record in records
        for finding in ((record.get("evidence_check") or {}).get("candidates") or []))

    counters = [r.get("counters") or {} for r in records]
    mean = lambda key: round(sum(c.get(key) or 0 for c in counters) / len(counters), 3) \
        if counters else None

    # PROBE alone, same images, read offline
    probe_labels = {}
    for record in records:
        if table is None:
            break
        label = _probe_reference(record, table)
        if label:
            probe_labels[record["trajectory_id"]] = label
    agreement = {
        "sessions_with_probe": len(probe_labels),
        "agree": sum(1 for record in records
                     if probe_labels.get(record["trajectory_id"])
                     == record.get("final_verdict")),
        "probe_correct": sum(1 for record in records
                             if probe_labels.get(record["trajectory_id"])
                             == record.get("ground_truth")),
        "session_correct": sum(1 for record in records
                               if record.get("final_verdict")
                               == record.get("ground_truth")),
    }

    per_condition = defaultdict(Counter)
    for record in records:
        per_condition[record.get("treatment")][record.get("verdict_status")] += 1

    return {
        "sessions": len(records),
        "verdict_status": dict(by_status),
        "labels": dict(labels),
        "halting_reasons": dict(by_reason),
        "legal_autonomous_conclusions": len(concluded),
        "legal_abstentions": len(abstained),
        "unparsed_conclusions": len(unparsed),
        "externally_truncated": len(truncated),
        "evidence_check": dict(checks),
        "evidence_defect_kinds": dict(defect_kinds),
        "evidence_candidate_kinds": dict(candidate_kinds),
        "protocol_violations": sum(c.get("protocol_violations") or 0
                                   for c in counters),
        "mean_turns": mean("model_turns"),
        "mean_expert_calls": mean("expert_calls"),
        "mean_weighted_cost": mean("weighted_cost"),
        "probe_alone": agreement,
        "per_condition_status": {key: dict(value) for key, value in per_condition.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default="g7_b2")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    paths = batch_paths(args.batch)
    records = list(load_trajectory_records(paths["trajectories"]).values())
    if not records:
        raise SystemExit(f"{paths['trajectories']} 里没有轨迹")

    from utils.config_fingerprint import calibration_identity, prompt_digest
    from utils.reliability import ReliabilityTable

    table = ReliabilityTable.load()
    summary = summarise(records, table)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch": args.batch,
        "versions": {
            "halting_policy": records[0].get("policy_version"),
            "prompt_digest": prompt_digest(),
            "calibration": calibration_identity(),
        },
        **summary,
    }
    output = args.output or os.path.join(PROJECT_ROOT, "calibration",
                                         f"g7_report_{args.batch}.json")
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"会话 {summary['sessions']}｜策略 {report['versions']['halting_policy']}"
          f"｜提示词 {report['versions']['prompt_digest']}"
          f"｜校准表 {report['versions']['calibration']['digest']}")
    print(f"  合法主动结案 {summary['legal_autonomous_conclusions']}"
          f"（其中合法弃权 {summary['legal_abstentions']}）"
          f"｜解析失败 {summary['unparsed_conclusions']}"
          f"｜外部截断 {summary['externally_truncated']}")
    print(f"  标签 {summary['labels']}｜停止原因 {summary['halting_reasons']}")
    print(f"  证据检查 {summary['evidence_check']}｜缺陷 {summary['evidence_defect_kinds']}"
          f"｜候选 {summary['evidence_candidate_kinds']}")
    print(f"  协议违规 {summary['protocol_violations']}｜平均轮数 {summary['mean_turns']}"
          f"｜平均调用 {summary['mean_expert_calls']}｜平均成本 {summary['mean_weighted_cost']}")
    print(f"  PROBE 单独对照：{summary['probe_alone']}")
    print(f"Report: {os.path.relpath(output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
