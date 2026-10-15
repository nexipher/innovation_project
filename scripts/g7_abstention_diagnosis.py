#!/usr/bin/env python3
"""
G7 §4.17 A3: the abstention diagnosis the handoff asks for.

An abstention is not one thing.  This script keeps four moments apart, because
collapsing them is how "the model abstained" becomes a number that means
nothing:

  first_candidate     the first parseable verdict the model proposed
  final_candidate     the last parseable verdict the model proposed
  system_label        the label the controller adopted
  parse_failure       turns where no verdict could be parsed at all

Reasons are classified into the seven categories of the handoff, each with the
sentence that triggered it, and every classification is a *candidate*: the
point is to sort the pile for a person, not to rule that an abstention was
wrong.  Both the whole batch (210) and the admitted subset (145) are reported,
so the admission step's selection effect is visible rather than implicit.
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
from scripts.g7_audit_traces import (
    EXCLUSIONS, RESULT_VERBS, SENTENCE_SPLIT, SOURCE_WORDS, states_the_near_chance)
from utils.batch import load_trajectory_records

# g7_b1 was archived on 2026-10-15: read it through the mapping, or this
# diagnosis reports an empty batch as if it were a result.
from utils.batch import resolve_path as _resolve_path  # noqa: E402

TRACES_DIR = _resolve_path(os.path.join(PROJECT_ROOT, "sft_data",
                                        "trajectories_g7_b1"))[0]
BATCH_DIR = _resolve_path(os.path.join(PROJECT_ROOT, "sft_data", "train",
                                       "batches", "g7_b1"))[0]
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_abstention_diagnosis.json")

VERDICT_PATTERN = re.compile(r'<verdict>\s*(\{.*?\})\s*</verdict>', re.S)

REASONS = ("near_chance_calibration", "real_conflict_or_limit", "demands_second_expert",
           "global_local_confusion", "score_layer_misuse", "process_anomaly", "other")

NEEDS_ANOTHER = re.compile(r"(没有其他专家|缺乏其他专家|缺少其他专家|没有更多证据|需要更多专家|"
                           r"无法获得更多|没有独立的第二|缺少独立验证|需要额外验证)")
GLOBAL_LOCAL = re.compile(r"(无法定位|不能定位|不指出.{0,6}区域|没有提供.{0,6}区域|"
                          r"缺乏.{0,6}区域|无法确定.{0,8}(具体|局部).{0,6}(位置|区域)|"
                          r"整图测量|无法判断.{0,6}局部)")
# A conflict claim must be about evidence, and it must not be a denial:
# "毛发颜色不一致" is not an evidentiary conflict, and "没有直接冲突" is the
# opposite of one.  Both slipped through the first version.
CONFLICT = re.compile(r"(冲突|相互矛盾|相互抵消|方向相反)")
CONFLICT_SUBJECT = re.compile(r"(证据|专家|测量|结论|后验|工具|信号|得分|判别)")
CONFLICT_DENIED = re.compile(r"(没有.{0,6}冲突|不存在.{0,6}冲突|无.{0,4}明显冲突|未.{0,4}冲突)")
# "JPEG 压缩专家" contains 压缩 and 专家; a bare technology word is not an
# applicability limit.  The limit has to be *claimed about this measurement*.
LIMIT = re.compile(r"(失效|不适用|质量\s*[≤<=]\s*70|重压缩后|"
                   r"q70\s*(格|上|下)?\s*(失效|不适用|不可用)|适用条件)")
LAYER = re.compile(r"(原始分数|原始值|得分|score).{0,12}(表明|指向|支持)")
PROCESS = re.compile(r"(已跳过执行|去重规则忽略|重复调用|模型停滞|没有新的测量)")


def parse_verdicts(trace: dict) -> Tuple[List[str], int]:
    """Every parseable verdict in order, and how many turns failed to parse."""
    verdicts, failures = [], 0
    for turn in trace.get("conversations", []):
        if (turn.get("from") or "") != "gpt":
            continue
        value = turn.get("value") or ""
        if "<verdict>" not in value:
            continue
        match = VERDICT_PATTERN.search(value)
        if not match:
            failures += 1
            continue
        try:
            verdicts.append(json.loads(match.group(1)).get("verdict"))
        except ValueError:
            failures += 1
    return verdicts, failures


UNCERTAINTY_PHRASES = ("不确定性", "无法确定", "无法判断", "不足以", "保留不确定",
                       "无法得出", "不能确定", "无法进一步")


def sentences(trace: dict) -> List[str]:
    """
    The prose, without the machine-readable verdict block.

    The block is JSON on its own lines; splitting it into "sentences" produces
    fragments like `</verdict>`, which is what the first version quoted back at
    the reviewer for every unclassified case — a citation that says nothing.
    """
    output = []
    for turn in trace.get("conversations", []):
        if (turn.get("from") or "") != "gpt":
            continue
        value = VERDICT_PATTERN.sub(" ", turn.get("value") or "")
        value = re.sub(r"</?verdict>", " ", value)
        output.extend(s.strip() for s in SENTENCE_SPLIT.split(value) if s.strip())
    return output


def rationale_sentence(trace: dict) -> str:
    """The sentence that best explains the abstention, for quoting."""
    candidates = sentences(trace)
    for sentence in reversed(candidates):
        if any(phrase in sentence for phrase in UNCERTAINTY_PHRASES):
            return sentence
    return max(candidates, key=len) if candidates else "（无助手句子）"


def classify(trace: dict, record: dict) -> Dict[str, List[dict]]:
    """Reason candidates for one abstention, one list per category."""
    tokens = record.get("evidence") or []
    likelihoods = [(t.get("calibrated_likelihood") or {}).get("Fake") for t in tokens]
    likelihoods = [float(p) for p in likelihoods if p is not None]
    text = "\n".join(sentences(trace))

    findings = defaultdict(list)

    def add(category: str, sentence: str, why: str) -> None:
        findings[category].append({"sentence": sentence[:220], "why": why,
                                   "status": "candidate"})

    # 1. the calibration itself is near neutral
    if likelihoods and max(abs(2 * p - 1) for p in likelihoods) <= 0.2:
        add("near_chance_calibration",
            max((s for s in sentences(trace) if "随机" in s), key=len, default="（无句子）"),
            f"全部 token 的 |2p−1| ≤ 0.2（{['%.2f' % p for p in likelihoods]}）")

    # 2. a real conflict between tokens, or an applicability limit
    for sentence in sentences(trace):
        if CONFLICT.search(sentence) and CONFLICT_SUBJECT.search(sentence) \
                and not CONFLICT_DENIED.search(sentence) \
                and (LIMIT.search(sentence) or _signed_conflict(tokens)):
            add("real_conflict_or_limit", sentence, "句子陈述冲突，且证据本身方向相反或存在适用性限制")
            break
    for sentence in sentences(trace):
        if LIMIT.search(sentence) and ("专家" in sentence or "测量" in sentence
                                       or "证据" in sentence):
            add("real_conflict_or_limit", sentence, "引用了具体适用性限制（如重压缩后失效）")
            break

    # 3. demanding a second expert although a signal exists
    for sentence in sentences(trace):
        if NEEDS_ANOTHER.search(sentence) and _decisive_present(likelihoods):
            add("demands_second_expert", sentence,
                "已有决定性校准信号，却以“缺少其他专家”为由弃权")
            break

    # 4. treating a global task as a localisation one
    for sentence in sentences(trace):
        if GLOBAL_LOCAL.search(sentence):
            add("global_local_confusion", sentence,
                "以“无法定位/不给区域”为由拒答全局分类")
            break

    # 5. reading the raw score instead of the calibrated value
    for sentence in sentences(trace):
        if LAYER.search(sentence) and not states_the_near_chance(sentence) \
                and not any(marker in sentence for marker in EXCLUSIONS):
            for token in tokens:
                p = (token.get("calibrated_likelihood") or {}).get("Fake")
                raw = token.get("raw_metric")
                if p is None or raw is None or not (0.4 <= float(p) <= 0.6):
                    continue
                if f"{float(raw):.4f}" in sentence or f"{float(raw):.3f}" in sentence:
                    add("score_layer_misuse", sentence,
                        f"引用原始值 {raw}（该 token 校准 P(Fake)={p}）作为方向依据")
                    break
            break

    # 6. the session itself misbehaved
    for sentence in sentences(trace):
        if PROCESS.search(sentence):
            add("process_anomaly", sentence, "会话内出现重复/跳过/停滞等流程异常")
            break

    if not findings:
        add("other", rationale_sentence(trace), "七类规则均未命中，需人工判读")
    return dict(findings)


def _signed_conflict(tokens: List[dict]) -> bool:
    signs = []
    for token in tokens:
        p = (token.get("calibrated_likelihood") or {}).get("Fake")
        if p is None:
            continue
        signs.append(1 if float(p) >= 0.5 else -1)
    return bool(signs) and len(set(signs)) > 1


def _decisive_present(likelihoods: List[float]) -> bool:
    return any(abs(2 * p - 1) >= 0.6 for p in likelihoods)


def diagnose(traces_dir: str = TRACES_DIR) -> dict:
    records = load_trajectory_records(traces_dir)
    # Sample ids carry an "f2_" prefix; trajectory ids do not.  Compare on the
    # field that exists for exactly this purpose.
    admitted = set()
    for name in ("sft_tool_positive.json", "sft_no_tool_positive.json",
                 "sft_honest_abstention.json"):
        path = os.path.join(BATCH_DIR, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            admitted |= {s["metadata"]["trajectory_id"] for s in json.load(handle)}

    moments = Counter()
    per_session = {}
    for trajectory_id, record in sorted(records.items()):
        trace = json.load(open(os.path.join(PROJECT_ROOT, record["trace_path"]),
                              encoding="utf-8"))
        verdicts, failures = parse_verdicts(trace)
        first = verdicts[0] if verdicts else None
        last = verdicts[-1] if verdicts else None
        system_label = record.get("final_verdict")
        moments["sessions"] += 1
        moments[f"first_{first}"] += 1
        moments[f"last_{last}"] += 1
        moments[f"system_{system_label}"] += 1
        if failures:
            moments["sessions_with_parse_failure"] += 1
        if system_label == "Uncertain":
            per_session[trajectory_id] = {
                "policy": record.get("policy"),
                "ground_truth": record.get("ground_truth"),
                "halting_reason": record.get("halting_reason"),
                "model_first": first, "model_last": last,
                "system_label": system_label,
                "calls": len(record.get("evidence") or []),
                "admitted": trajectory_id in admitted,
                "reasons": classify(trace, record),
            }

    other_with_signal = 0
    for entry in per_session.values():
        if "other" not in entry["reasons"]:
            continue
        record = records.get(next(k for k, v in per_session.items() if v is entry))
        likelihoods = [float((t.get("calibrated_likelihood") or {}).get("Fake"))
                       for t in (record.get("evidence") or [])
                       if (t.get("calibrated_likelihood") or {}).get("Fake") is not None]
        if _decisive_present(likelihoods):
            other_with_signal += 1

    by_reason = Counter()
    by_reason_admitted = Counter()
    for entry in per_session.values():
        for reason in entry["reasons"]:
            by_reason[reason] += 1
            if entry["admitted"]:
                by_reason_admitted[reason] += 1

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sessions": len(records),
        "moments": dict(moments),
        "abstentions": len(per_session),
        "abstentions_admitted": sum(1 for e in per_session.values() if e["admitted"]),
        "by_reason": dict(by_reason),
        "other_with_decisive_token": other_with_signal,
        "by_reason_admitted_subset": dict(by_reason_admitted),
        "by_policy": dict(Counter(e["policy"] for e in per_session.values())),
        "by_truth": dict(Counter(str(e["ground_truth"]) for e in per_session.values())),
        "by_call_count": dict(Counter(e["calls"] for e in per_session.values())),
        "by_halting_reason": dict(Counter(e["halting_reason"] for e in per_session.values())),
        "per_session": per_session,
        "note": ("分类是候选标记，不是判错：目标是让“有依据的弃权”与“流程异常的弃权”分开，"
                 "不是降低 Uncertain 占比。每条都给了触发句，可逐条复核。"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", default=TRACES_DIR)
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    report = diagnose(args.traces)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    m = report["moments"]
    print(f"会话 {report['sessions']}｜四个时刻："
          f"首次候选 {m.get('first_Uncertain',0)} Uncertain / "
          f"最后候选 {m.get('last_Uncertain',0)} Uncertain / "
          f"系统标签 {m.get('system_Uncertain',0)} Uncertain｜"
          f"解析失败会话 {m.get('sessions_with_parse_failure',0)}")
    print(f"弃权 {report['abstentions']}（其中入选 {report['abstentions_admitted']}）"
          f"｜按策略 {report['by_policy']}")
    print(f"按原因（全部）：{report['by_reason']}")
    print(f"按原因（入选子集）：{report['by_reason_admitted_subset']}")
    print(f"按停止原因：{report['by_halting_reason']}")
    print(f"\nReport: {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
