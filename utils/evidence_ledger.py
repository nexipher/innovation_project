"""
The evidence ledger: what was actually measured, and what may be said about it
(G7 §4.17 R3).

The audit found 45 sentences in which the model reports a result for an expert
that produced no token — including one that fabricated a whole
`calibrated_likelihood: Real=0.551` field.  The pipeline could not catch any of
them, because the validator reads the rendered sample and the fabrication lives
in the model's own closing turn.

This module is the authority that was missing.  It is built from executed
results only — expert, evidence_id, raw metric, calibrated likelihood, scope —
plus the set of requests that were skipped and the tools that were served.  A
claim about a measurement either matches the ledger or it does not.

Two boundaries matter as much as the checks:

  * advising a call, saying one was not made, or hypothesising a result are not
    claims and must not be flagged (the audit's first version flagged them);
  * a flag is evidence for a person or a correction prompt to act on, never a
    verdict on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

SENTENCE_SPLIT = re.compile(r"[。；;\n]")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

RESULT_VERBS = ("显示", "表明", "给出", "得到", "返回", "指出", "说明", "测得", "读数",
                "得分", "结果为", "水平为", "输出的", "报告了", "显示出")
# Advice, absence, hypothesis: the sentence is not claiming a measurement.
NOT_A_CLAIM = ("未调用", "没有调用", "尚未", "未获得", "没有获得", "无法获得", "无法",
               "不能", "不应", "建议调用", "可以调用", "应该调用", "若调用", "如果调用",
               "假设", "假定", "将会", "可能会得到", "需要调用", "尚未执行", "跳过")

SOURCE_WORDS: Dict[str, tuple] = {
    # 探针 is how the model actually names it in Chinese prose; leaving it out
    # meant eight of the audit's sentences were invisible to the check.
    "probe_dino_expert": ("probe", "探针", "鉴别器", "判别分数", "判别器", "检测器", "aigi"),
    "noise_expert": ("噪声", "noise"),
    "jpeg_expert": ("jpeg", "压缩"),
    "frequency_expert_v2": ("频域", "频率", "频谱", "freq"),
}


@dataclass
class Ledger:
    """What the session actually produced."""

    measured: Dict[str, dict] = field(default_factory=dict)   # source -> token
    skipped: List[str] = field(default_factory=list)          # sources asked twice
    served: List[str] = field(default_factory=list)           # tools the policy offered
    values: List[float] = field(default_factory=list)         # every number in the ledger

    @classmethod
    def from_session(cls, evidence_chain: Iterable[dict],
                     skipped_sources: Iterable[str] = (),
                     served: Iterable[str] = ()) -> "Ledger":
        measured = {}
        values: List[float] = []
        for token in evidence_chain or []:
            source = token.get("source")
            if not source:
                continue
            measured[source] = token
            for key in ("raw_metric", "strength"):
                value = token.get(key)
                if isinstance(value, (int, float)):
                    values.append(float(value))
            likelihood = (token.get("calibrated_likelihood") or {})
            for value in likelihood.values():
                if isinstance(value, (int, float)):
                    values.append(float(value))
        return cls(measured=measured, skipped=sorted(set(skipped_sources)),
                   served=list(served), values=values)

    # ------------------------------------------------------------------

    def ran(self, source: str) -> bool:
        return source in self.measured

    def summary(self) -> str:
        """One line a correction prompt can carry: what ran, what did not."""
        ran = ", ".join(sorted(self.measured)) or "（无）"
        unrun = ", ".join(sorted(set(self.served) - set(self.measured))) or "（无）"
        text = f"已执行的专家：{ran}；本会话未执行的专家：{unrun}"
        if self.skipped:
            text += f"；其中被跳过（重复请求）：{', '.join(self.skipped)}"
        return text

    def permitted_numbers(self) -> List[float]:
        return list(self.values)

    # ------------------------------------------------------------------

    def findings(self, text: str) -> List[dict]:
        """Every way `text` says something the ledger does not support."""
        output: List[dict] = []
        for sentence in SENTENCE_SPLIT.split(text or ""):
            sentence = sentence.strip()
            if not sentence:
                continue
            # A skipped expert is unrun *and* was asked twice; when the
            # sentence describes a re-measurement, the precise finding is the
            # skipped one, and reporting both double-counts the same defect.
            skipped = self._skipped_as_new(sentence)
            output.extend(skipped)
            covered = {finding.get("source") for finding in skipped}
            output.extend(finding for finding in self._claim_for_unrun_expert(sentence)
                          if finding.get("source") not in covered)
            output.extend(self._value_from_nowhere(sentence))
        return _dedupe(output)

    def _claim_for_unrun_expert(self, sentence: str) -> List[dict]:
        lowered = sentence.lower()
        if not any(verb in lowered for verb in RESULT_VERBS):
            return []
        if any(marker in lowered for marker in NOT_A_CLAIM):
            return []
        findings = []
        for source, words in SOURCE_WORDS.items():
            if self.ran(source):
                continue
            if any(word in lowered for word in words):
                findings.append({
                    "kind": "fabricated_expert_result", "source": source,
                    "sentence": sentence[:220],
                    "why": f"{source} 在本会话没有产生任何 token，句子却陈述了它的结果",
                })
        return findings

    def _value_from_nowhere(self, sentence: str) -> List[dict]:
        """A number presented as a measurement that appears in no token."""
        if not any(verb in sentence for verb in RESULT_VERBS):
            return []
        if any(marker in sentence for marker in NOT_A_CLAIM):
            return []
        if not any(word in sentence.lower()
                   for words in SOURCE_WORDS.values() for word in words):
            return []
        allowed = self.permitted_numbers()
        for number in NUMBER.findall(sentence):
            value = float(number)
            if value in (0.0, 1.0, 1000.0) or value > 1000:
                continue                      # coordinates, booleans, ranges
            if any(abs(value - known) < 1e-3 for known in allowed):
                continue
            # Rounding of a known value is fine: 3.0912 vs 3.091196
            if any(abs(value - known) < 0.05 for known in allowed):
                continue
            return [{
                "kind": "fabricated_value", "value": value,
                "sentence": sentence[:220],
                "why": "句子给出的数值在本会话的任何 token 中都不存在",
            }]
        return []

    def _skipped_as_new(self, sentence: str) -> List[dict]:
        lowered = sentence.lower()
        if not self.skipped:
            return []
        if not any(word in lowered for word in ("再次", "重新", "又一次", "新的测量", "再次测量")):
            return []
        if not any(verb in lowered for verb in RESULT_VERBS):
            return []
        for source in self.skipped:
            words = SOURCE_WORDS.get(source, ())
            if any(word in lowered for word in words):
                return [{
                    "kind": "skipped_request_as_new_evidence", "source": source,
                    "sentence": sentence[:220],
                    "why": "该专家的重复请求被跳过，句子里却当成了一次新的测量",
                }]
        return []


def correction_note(ledger: Ledger, findings: List[dict]) -> str:
    """
    The single correction the pipeline is allowed to send (R3).

    It states what the ledger contains and forbids the specific defect, without
    suggesting a label: the model is asked to rewrite from the measurements it
    has, and to say plainly if they do not settle the question.
    """
    kinds = sorted({finding["kind"] for finding in findings})
    detail = "；".join(f"“{finding['sentence'][:60]}…”（{finding['why']}）"
                       for finding in findings[:3])
    return (
        f"[System: 你的结论引用了本会话不存在的证据：{'、'.join(kinds)}。"
        f"{ledger.summary()}。{detail}。"
        "请只使用上面“已执行”的专家返回过的数值重写结论；"
        "如果这些测量不足以判定，就如实说明还缺什么测量，"
        "不要为任何未执行的专家编造结果、也不要引用任何 token 里没有的数值。]"
    )


def _dedupe(findings: List[dict]) -> List[dict]:
    seen, output = set(), []
    for finding in findings:
        key = (finding["kind"], finding.get("sentence"), finding.get("source"))
        if key in seen:
            continue
        seen.add(key)
        output.append(finding)
    return output
