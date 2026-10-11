#!/usr/bin/env python3
"""
G7 entry 3, step 4: what the human is asked to read, and what is checked first.

The plan's four questions — was a score dressed up as a physical artifact, were
the corroborating experts actually called and actually read, do the probability
and the label agree, and did anything get repeated, dropped or stopped too
early — are answered by a person looking at whole trajectories.  What this
script does is make that reading affordable: it flags the cases worth reading
closely and leaves the rest as a list.

Nothing here is a verdict.  A flag means "a person should look at this line",
never "this sample is wrong" — the same discipline the D2 screener was held to
after the vision judge turned out to be able to look but not to judge.

Usage:
  python scripts/g7_acceptance_review.py                       # the real run
  python scripts/g7_acceptance_review.py --dry-run             # the mock run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_acceptance_report.json")
DRY_RUN_REPORT_PATH = os.path.join(PROJECT_ROOT,
                                   "calibration", "g7_acceptance_report_dry_run.json")
TRACES_DIR = os.path.join(PROJECT_ROOT, "traces", "g7_acceptance")
DRY_RUN_TRACES_DIR = os.path.join(PROJECT_ROOT, "traces", "dry_run_sessions")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_acceptance_review.md")

_CALL_TAG = re.compile(r"<call_(\w+)>", re.I)
_SENTENCE_SPLIT = re.compile(r"[。；;\n]")

# Naming a score as if it were a physical trace is the failure mode this
# project keeps meeting: the model is handed a number and writes the physics
# itself.  The flag fires on co-occurrence inside one sentence, not on either
# vocabulary alone.
SCORE_WORDS = ("probe", "判别分数", "鉴别器", "分数", "概率", "p(fake)", "aigi")
PHYSICAL_WORDS = ("纹理", "噪声", "残差", "频域", "频率", "边缘", "像素级",
                  "pixel", "prnu", "传感器", "区域", "热图", "artifact", "伪影")


def load_trace(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def session_text(trace: dict) -> Tuple[str, str, List[str]]:
    """(assistant text, user-facing system text, every call tag emitted)."""
    assistant, system, calls = [], [], []
    for turn in trace.get("conversations", []):
        role = (turn.get("from") or turn.get("role") or "").lower()
        value = turn.get("value") or turn.get("content") or ""
        if not value:
            continue
        if role in ("assistant", "gpt"):
            assistant.append(value)
            calls.extend(match.group(1).lower() for match in _CALL_TAG.finditer(value))
        else:
            system.append(value)
    return "\n".join(assistant), "\n".join(system), calls


def probe_artifact_sentences(text: str) -> List[str]:
    flagged = []
    for sentence in _SENTENCE_SPLIT.split(text):
        lowered = sentence.lower()
        if any(word in lowered for word in SCORE_WORDS) and \
                any(word in lowered for word in PHYSICAL_WORDS):
            flagged.append(sentence.strip()[:160])
    return flagged


def review_session(record: dict, trace: Optional[dict]) -> dict:
    """The four questions, as far as a machine can take them."""
    findings: List[dict] = []
    if trace is None:
        return {"sample_id": record["sample_id"], "findings": [
            {"kind": "missing_trace", "detail": "找不到会话文件"}]}

    assistant_text, system_text, emitted = session_text(trace)
    tokens = trace.get("evidence_chain", [])
    sources = [token.get("source") for token in tokens]
    metadata = trace.get("metadata", {})
    verdict = trace.get("final_verdict", {})

    # 1. a call the model emitted that produced no token
    tag_to_source = {"probe": "probe_dino_expert", "freq": "frequency_expert_v2",
                     "noise": "noise_expert", "jpeg": "jpeg_expert"}
    emitted_counts = Counter(emitted)
    landed_sources = Counter(sources)
    for tag, count in emitted_counts.items():
        source = tag_to_source.get(tag)
        if source is None:
            findings.append({"kind": "call_without_token",
                             "detail": f"<call_{tag}> 不在工具箱内，调用被丢弃"})
            continue
        if landed_sources[source] < count - metadata.get("suppressed_duplicate_count", 0):
            findings.append({
                "kind": "call_without_token",
                "detail": f"发出 {count} 次 <call_{tag}>，只落地 {landed_sources[source]} 个 token"})

    # 2. a score written as physics
    for sentence in probe_artifact_sentences(assistant_text):
        findings.append({"kind": "score_as_artifact", "detail": sentence})

    # 3. the corroborating experts: called, and then actually read
    probe_probability = next((t.get("calibrated_likelihood", {}).get("Fake")
                              for t in tokens if t.get("source") == "probe_dino_expert"), None)
    for token in tokens:
        source = token.get("source")
        if source == "probe_dino_expert":
            continue
        name = (source or "").replace("_expert", "")
        if name and name.split("_")[0] not in assistant_text.lower():
            findings.append({"kind": "aux_token_unread",
                             "detail": f"{source} 的 token 在链上，但结论文本从未提到它"})

    # 4. probability vs label
    posterior = verdict.get("posterior")
    if posterior is not None and verdict.get("verdict") in ("Real", "Fake"):
        expected = "Fake" if posterior >= 0.5 else "Real"
        if verdict["verdict"] != expected:
            findings.append({"kind": "probability_label_mismatch",
                             "detail": f"后验 {posterior} 对应 {expected}，标签却是 {verdict['verdict']}"})
    if probe_probability is not None and record.get("probe_token_p_fake") is not None:
        if abs(probe_probability - record["probe_token_p_fake"]) > 1e-6:
            findings.append({"kind": "probe_probability_drift",
                             "detail": f"报告 {record['probe_token_p_fake']} vs 会话 {probe_probability}"})

    # 5. repeats, drops, early stops
    if metadata.get("suppressed_duplicate_count", 0) > 0:
        findings.append({"kind": "duplicate_call",
                         "detail": f"重复调用被抑制 {metadata['suppressed_duplicate_count']} 次"})
    if verdict.get("verdict") == "Uncertain" and metadata.get("halting_reason") == "no_expected_gain":
        utilities = metadata.get("tool_utilities") or {}
        positive = {k: round(v, 3) for k, v in utilities.items() if v > 0}
        if positive:
            findings.append({"kind": "stopped_with_gain_left",
                             "detail": f"no_expected_gain 停止，但仍有工具净收益为正：{positive}"})

    return {
        "sample_id": record["sample_id"],
        "label": record["label"],
        "treatment": record["treatment"],
        "origin": record.get("origin"),
        "verdict": record["verdict"],
        "posterior": record.get("posterior"),
        "halting_reason": record.get("halting_reason"),
        "calls": record.get("call_tags"),
        "tokens": [
            {"source": token.get("source"),
             "direction": token.get("direction"),
             "p_fake": (token.get("calibrated_likelihood") or {}).get("Fake"),
             "applicability": token.get("applicability"),
             "consistency": token.get("consistency")}
            for token in tokens],
        "probe_p_fake": probe_probability,
        "findings": findings,
        "trace_path": record.get("sft_data_path"),
    }


def markdown(report: dict, sessions: List[dict]) -> str:
    counts = Counter(finding["kind"] for session in sessions for finding in session["findings"])
    lines = [
        "# G7 入口 3 验收审阅单",
        "",
        f"- 模式：`{report['mode']}`｜样本 {report['samples']}｜批次源 {report['sources']}",
        f"- 汇总：{json.dumps(report['summary'], ensure_ascii=False)}",
        "",
        "> 自动标记只说明「这一行值得人看」，不构成结论（D2 的教训：视觉预筛能看不能判，",
        "> 文本层标记同理——它们负责把 120 条缩到该细看的那几条）。",
        "",
        f"## 自动标记（{sum(counts.values())} 处）",
        "",
    ]
    if counts:
        for kind, count in counts.most_common():
            lines.append(f"- `{kind}`：{count} 处")
    else:
        lines.append("- 无")
    lines += ["", "## 逐会话", ""]

    for session in sessions:
        lines.append(f"### {session['sample_id']} ｜GT {session.get('label')}｜"
                     f"{session.get('treatment')}｜{session.get('origin')}")
        lines.append(f"- 判定 {session.get('verdict')}（后验 {session.get('posterior')}）"
                     f"｜停止 {session.get('halting_reason')}｜调用 {session.get('calls')}")
        for token in session.get("tokens", []):
            lines.append(f"  - `{token['source']}` → {token['direction']}"
                         f"（P(Fake)={token['p_fake']}，{token['applicability']}，"
                         f"一致性 {token['consistency']}）")
        for finding in session["findings"]:
            lines.append(f"  - ⚠️ **{finding['kind']}**：{finding['detail']}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", default=None)
    parser.add_argument("--traces-dir", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report_path = args.report or (DRY_RUN_REPORT_PATH if args.dry_run else REPORT_PATH)
    traces_dir = args.traces_dir or (DRY_RUN_TRACES_DIR if args.dry_run else TRACES_DIR)
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)

    sessions = []
    for record in report["records"]:
        path = record.get("sft_data_path")
        trace = load_trace(os.path.join(PROJECT_ROOT, path)) if path else None
        sessions.append(review_session(record, trace))

    flagged = [s for s in sessions if s["findings"]]
    print(f"会话 {len(sessions)}｜有标记 {len(flagged)}")
    for kind, count in Counter(f["kind"] for s in sessions for f in s["findings"]).most_common():
        print(f"  {kind:26s} {count}")

    output = args.output or OUTPUT_PATH.replace(".md", "_dry_run.md" if args.dry_run else ".md")
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(markdown(report, sessions))
    print(f"\n审阅单：{os.path.relpath(output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
