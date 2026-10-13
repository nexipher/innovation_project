#!/usr/bin/env python3
"""
Replay the tightened evidence ledger over the raw g7_b1 sessions (§4.17 R4).

Read-only.  The A1 audit characterised the 210 raw sessions with its own
detectors; this replay asks a different question: with the field-bound ledger
and the errors actually repaired, **which sessions' adopted conclusion would
the runtime check stop on** — and of those, which fabricate an expert result or
a value rather than merely quoting a number loosely.

What it cannot say: whether the one correction would have succeeded, because
that needs the model.  So it reports "would have entered the correction path"
(and, separately, "would still fail if the correction changed nothing") — the
upper bound of the gate's reach, not its final count.

Usage:
  python scripts/g7_ledger_replay.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.batch import load_trajectory_records
from utils.evidence_ledger import Ledger

DEFAULT_TRACES = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_g7_b1")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_ledger_replay.json")


def load_trace(record: dict) -> Optional[dict]:
    path = record.get("trace_path")
    if not path:
        return None
    try:
        with open(os.path.join(PROJECT_ROOT, path), encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def adopted_texts(trace: dict) -> List[str]:
    """The text(s) a session would have adopted as its conclusion.

    The closing call is the last assistant turn; for sessions that halted on a
    candidate, the verdict turn is.  Both are checked, so a fabrication cannot
    hide behind which branch the old controller took.
    """
    turns = [turn.get("value") or "" for turn in trace.get("conversations", [])
             if (turn.get("from") or "") == "gpt"]
    texts: List[str] = []
    if turns:
        texts.append(turns[-1])
    report = (trace.get("final_verdict") or {}).get("report") or ""
    if report and report not in texts:
        texts.append(report)
    return texts


def system_texts(trace: dict) -> List[str]:
    return [turn.get("value") or "" for turn in trace.get("conversations", [])
            if (turn.get("from") or "") == "user"
            and (turn.get("value") or "").lstrip().startswith("[System:")]


def replay_record(record: dict) -> Optional[dict]:
    trace = load_trace(record)
    if trace is None:
        return None
    metadata = trace.get("metadata") or {}
    served = metadata.get("available_experts") or []
    ledger = Ledger.from_session(
        trace.get("evidence_chain") or [],
        served=served,
        system_texts=system_texts(trace),
    )
    findings: List[dict] = []
    for text in adopted_texts(trace):
        findings.extend(ledger.findings(text))
    # The same defect quoted twice in two texts is one defect.
    seen, unique = set(), []
    for finding in findings:
        key = (finding["kind"], finding.get("sentence"), finding.get("source"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(finding)
    return {
        "trajectory_id": record.get("trajectory_id"),
        "policy": record.get("policy"),
        "ground_truth": record.get("ground_truth"),
        "model_candidate": trace.get("final_verdict", {}).get("model_candidate"),
        "final_verdict": trace.get("final_verdict", {}).get("verdict"),
        "measured": sorted(ledger.measured),
        "findings": unique,
    }


def run(traces_dir: str = DEFAULT_TRACES) -> dict:
    records = load_trajectory_records(traces_dir)
    sessions = []
    skipped: List[str] = []
    for record in records.values():
        result = replay_record(record)
        if result is None:
            skipped.append(record.get("trajectory_id"))
            continue
        if result["findings"]:
            sessions.append(result)

    kinds = Counter(finding["kind"] for session in sessions
                    for finding in session["findings"])
    per_policy = Counter(session["policy"] for session in sessions)
    fabricated = [session for session in sessions
                  if any(finding["kind"] in ("fabricated_expert_result",
                                             "fabricated_value",
                                             "misattributed_value",
                                             "misattributed_field",
                                             "skipped_request_as_new_evidence")
                         for finding in session["findings"])]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "traces_dir": os.path.relpath(traces_dir, PROJECT_ROOT).replace(os.sep, "/"),
        "replayed": len(records) - len(skipped),
        "unreadable": skipped,
        "sessions_with_findings": len(sessions),
        "sessions_with_evidence_fabrication": len(fabricated),
        "findings_by_kind": dict(kinds),
        "sessions_by_policy": dict(per_policy),
        "note": ("只读回放：命中即“会进入一次纠正路径”，不是最终 failed 数；"
                 "纠正是否成功需要模型才能判定"),
        "sessions": sessions,
    }


def main() -> None:
    report = run()
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"回放会话 {report['replayed']}｜有发现 {report['sessions_with_findings']}"
          f"｜含证据编造 {report['sessions_with_evidence_fabrication']}")
    print(f"按类型：{report['findings_by_kind']}")
    print(f"按策略：{report['sessions_by_policy']}")
    print(f"Report: {os.path.relpath(OUTPUT_PATH, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
