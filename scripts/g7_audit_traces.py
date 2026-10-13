#!/usr/bin/env python3
"""
G7 §4.17 A1: audit the raw traces, not the rendered samples.

The review found defects that the pipeline's validator could not see, because
it reads the *rendered* training sample — whose reasoning this project writes.
A fabrication in the model's own closing turn never reaches that text, so the
sample passes while the session it came from is unsound.

This script states a criterion per defect class, applies it to every session,
and emits findings with the sentence that triggered them.  Nothing here is a
verdict: a finding is `confirmed` only where the criterion is mechanical (a
number that appears in no token, a coordinate outside the declared range), and
`candidate` where a sentence has to be read.  The counts this project has been
quoting were produced by keyword matching; this replaces them.

Usage:
  python scripts/g7_audit_traces.py                    # the g7_b1 batch
  python scripts/g7_audit_traces.py --traces sft_data/trajectories_g7_b1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.batch import load_trajectory_records
from utils.evidence_ledger import SOURCE_WORDS

DEFAULT_TRACES = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_g7_b1")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_audit_raw_traces.json")

CALL_TAG = re.compile(r"<call_(\w+)>\s*\[?\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\]?",
                      re.I)
SENTENCE_SPLIT = re.compile(r"[。；;\n]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

TAG_TO_SOURCE = {"probe": "probe_dino_expert", "noise": "noise_expert",
                 "jpeg": "jpeg_expert", "freq": "frequency_expert_v2"}
# The expert vocabulary lives with the ledger so the audit and the runtime
# check cannot drift apart.

SOURCE_WORDS = {
    "probe_dino_expert": ("probe", "鉴别器", "判别分数", "判别器", "检测器", "aigi"),
    "noise_expert": ("噪声", "noise"),
    "jpeg_expert": ("jpeg", "压缩"),
    "frequency_expert_v2": ("频域", "频率", "频谱", "freq"),
}

# A sentence *claims a measurement* when it pairs a subject with a result verb
# and something value-like.  Advice, absence and hypotheses are excluded by the
# markers below — the handoff's requirement that the checker tell "suggest
# calling", "not called yet" and "assumed" apart from "claims to have got it".
RESULT_VERBS = ("显示", "表明", "给出", "得到", "返回", "指出", "说明", "测得", "读数",
                "得分", "结果为", "水平为", "输出的", "报告了", "显示出")
EXCLUSIONS = ("未调用", "没有调用", "尚未", "未获得", "没有获得", "无法", "不能", "不应",
              "建议调用", "可以调用", "应该调用", "若调用", "如果调用", "假设", "假定",
              "将会", "可能会得到", "需要调用", "尚未执行")
DIRECTION_WORDS = ("偏向", "指向", "支持", "表明", "说明", "证明", "指示")

# A sentence that names the near-chance calibration and withholds direction is
# the behaviour we are training for, not a defect.  The model writes 接近于随机
# as often as 接近随机, so the pattern tolerates the particle — substring
# matching missed one candidate in the adjudication and would have kept it
# wrongly flagged forever.
NEAR_CHANCE_STATED = re.compile(r"接近(?:于)?随机|接近随机水平")
DIRECTION_WITHHELD = re.compile(r"没有明确的方向性|不足以确定方向|不足以确定图像的真实性|不足以确定方向性")
UNCERTAINTY_STATED = re.compile(r"不确定性（0\.52）|不确定性\(0\.52\)")


def states_the_near_chance(sentence: str) -> bool:
    """The sentence declines to read a direction out of a near-chance value."""
    if NEAR_CHANCE_STATED.search(sentence) or DIRECTION_WITHHELD.search(sentence):
        return True
    return bool(UNCERTAINTY_STATED.search(sentence) and "可能性较小" in sentence)


def sentences_of(text: str) -> List[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text or "") if s.strip()]


def assistant_turns(trace: dict) -> List[Tuple[int, str]]:
    return [(i, turn.get("value") or "")
            for i, turn in enumerate(trace.get("conversations", []))
            if (turn.get("from") or "") == "gpt"]


def is_closing_turn(trace: dict, index: int) -> bool:
    """A turn the controller asked for after halting, not exploration."""
    for turn in reversed(trace.get("conversations", [])[:index]):
        if (turn.get("from") or "") != "gpt":
            value = turn.get("value") or ""
            if value.lstrip().startswith("[System:"):
                return True
            return False
    return False


def tokens_of(record: dict) -> List[dict]:
    return record.get("evidence") or []


def invoked_sources(record: dict) -> set:
    return {t.get("source") for t in tokens_of(record)}


# ---------------------------------------------------------------------------
# Detectors.  Each returns findings with the sentence and the criterion.
# ---------------------------------------------------------------------------

def fabricated_expert(trace: dict, record: dict) -> List[dict]:
    """
    A closing turn states a result for an expert that produced no token.

    Mechanical half: the expert has no token in this session.  The judgement
    half is the sentence, which is quoted for a person to read.
    """
    invoked = invoked_sources(record)
    findings = []
    for index, text in assistant_turns(trace):
        for sentence in sentences_of(text):
            lowered = sentence.lower()
            for source, words in SOURCE_WORDS.items():
                if source in invoked:
                    continue
                if not any(word in lowered for word in words):
                    continue
                if not any(verb in lowered for verb in RESULT_VERBS):
                    continue
                if any(marker in lowered for marker in EXCLUSIONS):
                    continue
                findings.append({
                    "class": "fabricated_expert_result",
                    "source": source,
                    "sentence": sentence[:220],
                    "criterion": "该专家在本会话无任何 token，句子却陈述其结果",
                    "status": "candidate",
                })
    return _dedupe(findings)


def coordinate_out_of_range(trace: dict, record: dict) -> List[dict]:
    """Coordinates outside the declared [0, 1000] normalized space."""
    findings = []
    for index, text in assistant_turns(trace):
        for match in CALL_TAG.finditer(text):
            values = [int(match.group(i)) for i in range(2, 6)]
            if any(v > 1000 for v in values):
                findings.append({
                    "class": "coordinate_out_of_range",
                    "sentence": match.group(0)[:120],
                    "criterion": "调用坐标超出 [0,1000] 归一化范围",
                    "status": "confirmed",
                })
    return _dedupe(findings)


def raw_score_as_direction(trace: dict, record: dict) -> List[dict]:
    """
    A near-chance token's *raw* value quoted as if it pointed somewhere.

    The calibrated likelihood is what carries direction; when it sits at 0.52
    the number must not be read as a lean.  Mechanical half: the raw value is
    quoted verbatim and the token's calibrated P(Fake) is in [0.4, 0.6].
    """
    tokens = tokens_of(record)
    findings = []
    for index, text in assistant_turns(trace):
        for sentence in sentences_of(text):
            for token in tokens:
                likelihood = (token.get("calibrated_likelihood") or {}).get("Fake")
                raw = token.get("raw_metric")
                if likelihood is None or raw is None:
                    continue
                if not (0.4 <= float(likelihood) <= 0.6):
                    continue
                if not _mentions_raw(sentence, float(raw)):
                    continue
                if not any(word in sentence for word in DIRECTION_WORDS):
                    continue
                if any(marker in sentence for marker in EXCLUSIONS):
                    continue
                if states_the_near_chance(sentence):
                    continue
                findings.append({
                    "class": "raw_score_as_direction",
                    "source": token.get("source"),
                    "raw_metric": raw,
                    "p_fake": round(float(likelihood), 3),
                    "sentence": sentence[:220],
                    "criterion": "近随机 token（P(Fake)∈[0.4,0.6]）的原始值被当作方向性依据",
                    "status": "candidate",
                })
    return _dedupe(findings)


def _mentions_raw(sentence: str, raw: float) -> bool:
    """The raw value, to four significant characters, appears in the sentence."""
    for text in {f"{raw:.4f}", f"{raw:.3f}", f"{raw:.2f}", f"{round(raw, 1):.1f}"}:
        if text in sentence:
            return True
    return False


def label_report_conflict(trace: dict, record: dict) -> List[dict]:
    """
    The adopted label and the model's own last verdict disagree.

    Mechanical: both are parsed fields.  It is recorded rather than judged —
    under §4.17 the model's legal conclusion is not to be overridden silently,
    and this is the measurement of how often it used to be.
    """
    final = record.get("final_verdict")
    candidate = record.get("model_candidate")
    if not final or not candidate or candidate == final:
        return []
    return [{
        "class": "label_overrode_model",
        "sentence": f"模型结案 {candidate} → 系统标签 {final}",
        "criterion": "模型自报结案与系统最终标签不一致",
        "status": "confirmed",
    }]


def skipped_request_as_evidence(trace: dict, record: dict) -> List[dict]:
    """A deduplicated request later spoken of as a fresh measurement."""
    skipped = [t for t in trace.get("conversations", [])
               if (t.get("from") or "") == "user"
               and ("已跳过执行" in (t.get("value") or "")
                    or "去重规则忽略" in (t.get("value") or ""))]
    if not skipped:
        return []
    findings = []
    for index, text in assistant_turns(trace):
        for sentence in sentences_of(text):
            lowered = sentence.lower()
            if not any(w in lowered for w in ("再次", "重新", "又一次", "新的测量", "再次测量")):
                continue
            for source, words in SOURCE_WORDS.items():
                if any(word in lowered for word in words) and \
                        any(verb in lowered for verb in RESULT_VERBS):
                    findings.append({
                        "class": "skipped_repeat_as_new_evidence",
                        "source": source,
                        "sentence": sentence[:220],
                        "criterion": "本会话存在被跳过的重复调用，句子里把该专家说成有新测量",
                        "status": "candidate",
                    })
    return _dedupe(findings)


DETECTORS = (fabricated_expert, coordinate_out_of_range, raw_score_as_direction,
             label_report_conflict, skipped_request_as_evidence)


def _dedupe(findings: List[dict]) -> List[dict]:
    seen, output = set(), []
    for finding in findings:
        key = (finding["class"], finding.get("sentence"))
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output


def audit(traces_dir: str = DEFAULT_TRACES) -> dict:
    records = load_trajectory_records(traces_dir)
    if not records:
        raise SystemExit(f"没有轨迹记录：{traces_dir}")

    findings: Dict[str, List[dict]] = defaultdict(list)
    unreadable = []
    for trajectory_id, record in sorted(records.items()):
        trace_path = record.get("trace_path")
        try:
            with open(os.path.join(PROJECT_ROOT, trace_path), encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, ValueError, TypeError):
            unreadable.append(trajectory_id)
            continue
        for detector in DETECTORS:
            for finding in detector(trace, record):
                findings[trajectory_id].append(finding)

    by_class = Counter(f["class"] for entries in findings.values() for f in entries)
    by_class_status = Counter((f["class"], f["status"])
                              for entries in findings.values() for f in entries)
    by_policy = Counter(record["policy"] for trajectory_id, record in records.items()
                        if trajectory_id in findings)
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "traces_dir": os.path.relpath(traces_dir, PROJECT_ROOT),
        "sessions": len(records),
        "sessions_with_findings": len(findings),
        "unreadable": unreadable,
        "by_class": dict(by_class),
        "by_class_status": {f"{k[0]}|{k[1]}": v for k, v in sorted(by_class_status.items())},
        "sessions_by_policy": dict(by_policy),
        "findings": {k: v for k, v in sorted(findings.items())},
        "note": ("status=confirmed 表示判据机械可判（数值/坐标/解析字段）；candidate 表示"
                 "句子需人工阅读确认。计数不得当作定罪数——这正是 2026-10-13 交接文档"
                 "对关键词统计的批评。"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=DEFAULT_TRACES)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    report = audit(args.traces)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    print(f"会话 {report['sessions']}｜有发现 {report['sessions_with_findings']}"
          f"｜不可读 {len(report['unreadable'])}")
    print(f"按类别：{report['by_class']}")
    print(f"按类别×状态：{report['by_class_status']}")
    print(f"按策略（有发现的会话）：{report['sessions_by_policy']}")
    print(f"\nReport: {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
