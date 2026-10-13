"""
The evidence ledger: what was actually measured, and what may be said about it
(G7 §4.17 R3/R4).

The audit found 45 sentences in which the model reports a result for an expert
that produced no token — including one that fabricated a whole
`calibrated_likelihood: Real=0.551` field.  The pipeline could not catch any of
them, because the validator reads the rendered sample and the fabrication lives
in the model's own closing turn.

This module is the authority that was missing.  A claim about a measurement
either matches the ledger or it does not, and the ledger is **bound by
expert → evidence_id → field → value**, not by "the number appears somewhere":

  * an expert's numbers may only be used for that expert (the review's
    "PROBE score quoted as the noise expert's result" is `misattributed_value`
    and names the real owner);
  * a value may only be called a calibrated probability when it comes from the
    likelihood fields (the review's "raw metric presented as a probability" is
    `misattributed_field`);
  * P(Real) and P(Fake) are different fields, so swapping them is visible.

Three boundaries matter as much as the checks:

  * advising a call, saying one was not made, or hypothesising a result are not
    claims and must not be flagged (the audit's first version flagged them);
  * negation is judged **per clause**, not per sentence: "噪声专家显示低噪声，
    但不能单独确定真假" still states a measurement and must be checked;
  * a flag is evidence for a person or a correction prompt to act on, never a
    verdict on its own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SENTENCE_SPLIT = re.compile(r"[。；;\n]")
CLAUSE_SPLIT = re.compile(r"[，,、]")
# A leading-digit guard keeps ids (E-1a2b), versions (v2) and identifiers out.
NUMBER = re.compile(r"(?<![A-Za-z0-9_.\-])(\d+(?:\.\d+)?)\s*([%％])?")
COORDINATE_BLOCK = re.compile(r"\[[^\[\]]*\]")
DIMENSION = re.compile(r"\d+(?:\.\d+)?\s*[×xX*]\s*\d+(?:\.\d+)?")
# Call tags are requests, not measurements; the bbox inside them is a
# coordinate, and "1. <call_noise>…" is a list marker, not a value.  Leaving
# them in made the replay flag "1" as the JPEG expert's reliability.
CALL_TAG = re.compile(r"</?call_\w+>", re.I)
# A list marker is followed by a space or a tag — never by a digit, which is
# what separates "1. 的测量结果" from the value "0.0750" (reading that as a
# marker turned every "0.XXXX" at a clause start into "XXXX").
ENUMERATOR = re.compile(r"(^|[\s（(])(\d+)\s*[.）)]\s*(?=\s|<)|(^|[\s（(])(\d+)、\s*")
# "校准概率为 3.09" states a value without any of the verbs above; a bare
# copula only counts as a claim when a number follows it, so "因为…" and
# "认为…" are not read as measurements.
ASSIGNMENT = re.compile(r"(?:为|是|等于|约等于|≈|=)\s*约?\s*\d")

RESULT_VERBS = ("显示", "表明", "给出", "得到", "返回", "指出", "说明", "测得", "读数",
                "得分", "结果为", "水平为", "输出的", "报告了", "显示出")
# Advice, absence, hypothesis, denial: the clause is not claiming a measurement.
NOT_A_CLAIM = ("未调用", "没有调用", "尚未", "未获得", "没有获得", "无法获得", "无法",
               "不能", "不应", "建议调用", "可以调用", "应该调用", "若调用", "如果调用",
               "假设", "假定", "将会", "可能会得到", "需要调用", "尚未执行", "跳过",
               "没有", "无需", "不再")

SOURCE_WORDS: Dict[str, tuple] = {
    # 探针 is how the model actually names it in Chinese prose; leaving it out
    # meant eight of the audit's sentences were invisible to the check.
    "probe_dino_expert": ("probe", "探针", "鉴别器", "判别分数", "判别器", "检测器",
                          "探测器", "aigi"),
    "noise_expert": ("噪声", "noise"),
    "jpeg_expert": ("jpeg", "压缩"),
    "frequency_expert_v2": ("频域", "频率", "频谱", "freq"),
}

# What kind of quantity a clause claims the number to be.
#
# Only an explicit *calibrated* probability (or a P(Real)/P(Fake) class claim)
# is checked strictly against the likelihood fields.  A bare "概率" is how the
# model describes PROBE's own detection score, and that score is a probability
# in the detector's terms — flagging it would punish accurate narration.  What
# must not pass is a raw metric *presented as the calibrated value*.
STRICT_CUES = ("校准概率", "校准后概率", "似然")
LABELLED_STRICT_CUES = (("p(real)", "prob_real"), ("p（real）", "prob_real"),
                        ("p(fake)", "prob_fake"), ("p（fake）", "prob_fake"))
REAL_CUES = ("真实", "real", "真图")
FAKE_CUES = ("伪造", "fake", "生成图", "ai生成", "ai 生成")
LOOSE_PROBABILITY_CUES = ("概率", "可能性")
# A number introduced as a score is not being called a probability, whatever
# else the clause mentions.
METRIC_CUES = ("得分", "分数", "原始", "度量", "读数", "水平", "指标")
# Numbers the system itself put in the conversation (reference posterior,
# conflict score) are citable — but only as system numbers.
SYSTEM_CUES = ("系统", "策略", "参考后验", "后验", "冲突度")
# The model's own confidence is not an expert measurement.
SELF_CUES = ("置信度", "confidence", "把握")

TOKEN_TEXT_FIELDS = ("reasoning", "interpretation_text", "phenomenon",
                     "counter_explanation", "evidence_name")
# Structured fields whose values are probabilities, and the display tolerance
# each class of claim is allowed.  0.05 is *not* a general tolerance: a
# probability claim must be good to 0.005 unless it is written more precisely,
# so a fabricated "Fake=1" or "Real=0" cannot slip through.
PROBABILITY_TOLERANCE_CAP = 0.005
METRIC_TOLERANCE_CAP = 0.05


def _decimals(literal: str) -> int:
    return len(literal.split(".", 1)[1]) if "." in literal else 0


def _matches(claim: float, known: float, decimals: int, cap: float) -> bool:
    """A claim matches when it is a correct rounding of the known value."""
    tolerance = min(0.5 * 10 ** (-decimals), cap)
    return abs(claim - known) <= tolerance


def _numbers(text: str) -> List[float]:
    return [float(match.group(1)) for match in NUMBER.finditer(text or "")]


@dataclass
class ExpertRecord:
    """One executed measurement: which expert, which token, which fields."""

    source: str
    evidence_id: str = ""
    fields: Dict[str, float] = field(default_factory=dict)
    text_values: List[float] = field(default_factory=list)

    def probability_fields(self) -> Dict[str, float]:
        return {name: value for name, value in self.fields.items()
                if name.startswith("calibrated_likelihood.")}

    def metric_fields(self) -> Dict[str, float]:
        return {name: value for name, value in self.fields.items()
                if not name.startswith("calibrated_likelihood.")}


@dataclass
class Ledger:
    """What the session actually produced."""

    measured: Dict[str, ExpertRecord] = field(default_factory=dict)
    skipped: List[str] = field(default_factory=list)          # sources asked twice
    served: List[str] = field(default_factory=list)           # tools the policy offered
    system_values: List[float] = field(default_factory=list)  # numbers the system injected

    @classmethod
    def from_session(cls, evidence_chain: Iterable[dict],
                     skipped_sources: Iterable[str] = (),
                     served: Iterable[str] = (),
                     system_texts: Iterable[str] = ()) -> "Ledger":
        measured: Dict[str, ExpertRecord] = {}
        for token in evidence_chain or []:
            source = token.get("source")
            if not source:
                continue
            record = ExpertRecord(source=source,
                                  evidence_id=str(token.get("evidence_id") or ""))
            for key in ("raw_metric", "strength", "reliability"):
                value = token.get(key)
                if isinstance(value, (int, float)):
                    record.fields[key] = float(value)
            for name, value in (token.get("calibrated_likelihood") or {}).items():
                if isinstance(value, (int, float)):
                    record.fields[f"calibrated_likelihood.{name}"] = float(value)
            for name, value in (token.get("condition_metadata") or {}).items():
                if isinstance(value, (int, float)):
                    record.fields[f"condition_metadata.{name}"] = float(value)
            for key in TOKEN_TEXT_FIELDS:
                record.text_values.extend(_numbers(str(token.get(key) or "")))
            measured[source] = record
        system_values: List[float] = []
        for text in system_texts or ():
            system_values.extend(_numbers(text))
        return cls(measured=measured, skipped=sorted(set(skipped_sources)),
                   served=list(served), system_values=system_values)

    # ------------------------------------------------------------------

    def ran(self, source: str) -> bool:
        return source in self.measured

    def summary(self) -> str:
        """
        One line a correction prompt can carry: what ran, with which numbers.

        The metric fields are printed so a rewrite has the right values in
        front of it; the probability fields are referred to by count, because
        the note must not put a label name (Real / Fake) into the model's
        mouth.  The tokens themselves carry those values.
        """
        parts = []
        for source in sorted(self.measured):
            record = self.measured[source]
            metrics = ", ".join(f"{name}={value:g}"
                                for name, value in sorted(record.metric_fields().items()))
            likelihood = record.probability_fields()
            tail = f"；校准概率 {len(likelihood)} 项（数值见该专家的 token）" if likelihood else ""
            parts.append(f"{source}（{record.evidence_id or '无 id'}；{metrics}{tail}）")
        ran = "；".join(parts) or "（无）"
        unrun = ", ".join(sorted(set(self.served) - set(self.measured))) or "（无）"
        text = f"已执行的专家：{ran}；本会话未执行的专家：{unrun}"
        if self.skipped:
            text += f"；其中被跳过（重复请求）：{', '.join(self.skipped)}"
        return text

    def permitted_numbers(self) -> List[float]:
        """Every number this session is allowed to quote, wherever it came from."""
        values: List[float] = list(self.system_values)
        for record in self.measured.values():
            values.extend(record.fields.values())
            values.extend(record.text_values)
        return values

    def field_index(self) -> Dict[str, Dict[str, float]]:
        """source → field → value, for readers that need the binding itself."""
        return {source: dict(record.fields) for source, record in self.measured.items()}

    # ------------------------------------------------------------------

    def findings(self, text: str) -> List[dict]:
        """Every way `text` says something the ledger does not support."""
        output: List[dict] = []
        for sentence in SENTENCE_SPLIT.split(text or ""):
            sentence = sentence.strip()
            if not sentence:
                continue
            output.extend(self._sentence_findings(sentence))
        return _dedupe(output)

    # -- sentence level -------------------------------------------------

    def _sentence_findings(self, sentence: str) -> List[dict]:
        cleaned = COORDINATE_BLOCK.sub(" ", sentence)
        cleaned = DIMENSION.sub(" ", CALL_TAG.sub(" ", cleaned))
        cleaned = ENUMERATOR.sub(r"\1", cleaned)
        clauses = [clause.strip() for clause in CLAUSE_SPLIT.split(cleaned) if clause.strip()]
        findings: List[dict] = []
        subjects: List[str] = []
        skipped_here: set = set()
        for clause in clauses:
            named = _named_sources(clause)
            if named:
                subjects = named
            if not subjects:
                continue
            if not _is_claim(clause):
                continue
            for source in subjects:
                if self.ran(source):
                    continue
                findings.append({
                    "kind": "fabricated_expert_result", "source": source,
                    "sentence": clause[:220],
                    "why": f"{source} 在本会话没有产生任何 token，句子却陈述了它的结果",
                })
            skipped = self._skipped_as_new(clause, subjects)
            skipped_here.update(finding["source"] for finding in skipped)
            findings.extend(skipped)
            ran_subjects = [source for source in subjects if self.ran(source)]
            if ran_subjects:
                findings.extend(self._numbers_in(clause, ran_subjects))
        # A skipped expert is unrun *and* was asked twice; when the clause
        # describes a re-measurement, the precise finding is the skipped one,
        # and reporting both double-counts the same defect.
        return [finding for finding in findings
                if not (finding["kind"] == "fabricated_expert_result"
                        and finding.get("source") in skipped_here)]

    # -- number checking -------------------------------------------------

    def _numbers_in(self, clause: str, sources: Sequence[str]) -> List[dict]:
        cues = _cues(clause)
        system_cue = any(kind == "system" for _, kind in cues)
        findings: List[dict] = []
        for match in NUMBER.finditer(clause):
            cue = _cue_covering(cues, match.start())
            if cue == "self":
                continue
            label = _strict_label(cue, cues)
            strict = label != ""          # "" = the clause calls it no kind of probability
            literal = match.group(1)
            value = float(literal)
            if match.group(2):
                value /= 100.0
            decimals = _decimals(literal)
            finding = self._check_number(
                value, decimals, sources, strict, label, system_cue)
            if finding is not None:
                finding["value"] = value
                finding["sentence"] = clause.strip()[:220]
                findings.append(finding)
        return findings

    def _check_number(self, value: float, decimals: int, sources: Sequence[str],
                      strict: bool, label: Optional[str],
                      system_cue: bool) -> Optional[dict]:
        metric_cap = METRIC_TOLERANCE_CAP
        prob_cap = PROBABILITY_TOLERANCE_CAP
        wrong_field: Optional[Tuple[str, str]] = None
        for source in sources:
            record = self.measured[source]
            if strict:
                swapped: Optional[str] = None
                for name, known in record.probability_fields().items():
                    if not _matches(value, known, decimals, prob_cap):
                        continue
                    if _field_allowed(name, label):
                        return None
                    # Right number, wrong class: the token says this value is
                    # P(Real) and the sentence calls it P(Fake).
                    swapped = swapped or name
                if swapped is not None:
                    wrong_field = wrong_field or (source, swapped)
                if system_cue and _matches_any(self.system_values, value, decimals, prob_cap):
                    return None
                for name, known in record.metric_fields().items():
                    if _matches(value, known, decimals, metric_cap):
                        wrong_field = wrong_field or (source, name)
                if wrong_field is None:
                    for known in record.text_values:
                        if _matches(value, known, decimals, metric_cap):
                            return None
            else:
                for known in list(record.fields.values()) + record.text_values:
                    if _matches(value, known, decimals, metric_cap):
                        return None
                if system_cue and _matches_any(self.system_values, value, decimals, metric_cap):
                    return None
        if wrong_field is not None:
            source, name = wrong_field
            if name.startswith("calibrated_likelihood."):
                why = (f"{value:g} 是 {source} 的 {name}，句子把它说成了另一种概率"
                       "（方向被说反）")
            else:
                why = (f"{value:g} 是 {source} 的 {name}（原始度量），"
                       "不是该专家的校准概率，句子把它说成了概率")
            return {
                "kind": "misattributed_field", "source": source,
                "field": name,
                "evidence_id": self.measured[source].evidence_id,
                "why": why,
            }
        for other, record in self.measured.items():
            if other in sources:
                continue
            for name, known in record.fields.items():
                if _matches(value, known, decimals,
                            prob_cap if name.startswith("calibrated_likelihood.") else metric_cap):
                    return {
                        "kind": "misattributed_value", "source": other,
                        "field": name, "evidence_id": record.evidence_id,
                        "why": (f"{value:g} 是 {other} 的 {name}，"
                                "不属于句子提到的专家"),
                    }
        return {
            "kind": "fabricated_value",
            "why": "句子给出的数值在本会话的任何 token 中都不存在",
        }

    def _skipped_as_new(self, clause: str, sources: Sequence[str]) -> List[dict]:
        lowered = clause.lower()
        if not self.skipped:
            return []
        if not any(word in lowered for word in ("再次", "重新", "又一次", "新的测量", "再次测量")):
            return []
        findings = []
        for source in self.skipped:
            if source not in sources:
                continue
            findings.append({
                "kind": "skipped_request_as_new_evidence", "source": source,
                "sentence": clause[:220],
                "why": "该专家的重复请求被跳过，句子里却当成了一次新的测量",
            })
        return findings


def _named_sources(clause: str) -> List[str]:
    lowered = clause.lower()
    return [source for source, words in SOURCE_WORDS.items()
            if any(word in lowered for word in words)]


def _is_claim(clause: str) -> bool:
    lowered = clause.lower()
    if not any(verb in lowered for verb in RESULT_VERBS) and not ASSIGNMENT.search(clause):
        return False
    return not any(marker in lowered for marker in NOT_A_CLAIM)


def _cues(clause: str) -> List[Tuple[int, str]]:
    """
    Every cue in the clause with its position.

    The strict cues are masked out before the loose ones are searched, so the
    "概率" inside "校准概率" cannot be read as a loose cue and silently drop
    the strictness the sentence asked for.
    """
    lowered = clause.lower()
    masked = list(lowered)
    cues: List[Tuple[int, str]] = []

    def blank(start: int, width: int) -> None:
        for index in range(start, min(start + width, len(masked))):
            masked[index] = " "

    for word in STRICT_CUES:
        start = lowered.find(word)
        while start != -1:
            cues.append((start, "strict"))
            blank(start, len(word))
            start = lowered.find(word, start + 1)
    for word, kind in LABELLED_STRICT_CUES:
        start = lowered.find(word)
        while start != -1:
            cues.append((start, kind))
            blank(start, len(word))
            start = lowered.find(word, start + 1)

    remaining = "".join(masked)
    for words, kind in ((REAL_CUES, "real"), (FAKE_CUES, "fake"),
                        (LOOSE_PROBABILITY_CUES, "loose"), (METRIC_CUES, "metric"),
                        (SYSTEM_CUES, "system"), (SELF_CUES, "self")):
        for word in words:
            start = remaining.find(word)
            while start != -1:
                cues.append((start, kind))
                start = remaining.find(word, start + 1)
    return sorted(cues)


def _cue_covering(cues: List[Tuple[int, str]], position: int) -> Optional[str]:
    """The nearest cue at or before the number; a trailing cue still counts."""
    before = [kind for start, kind in cues if start <= position]
    if before:
        return before[-1]
    after = [kind for start, kind in cues if start > position]
    return after[0] if after else None


def _strict_label(cue: Optional[str], cues: List[Tuple[int, str]]) -> Optional[str]:
    """The likelihood field a strict claim is about, or None for "any"."""
    strict_present = any(kind in ("strict", "prob_real", "prob_fake")
                         for _, kind in cues)
    if cue == "prob_real":
        return "real"
    if cue == "prob_fake":
        return "fake"
    if cue == "strict":
        return None
    if cue in ("real", "fake") and strict_present:
        return cue
    return ""            # "" means: not a strict claim at all


def _field_allowed(field_name: str, label: Optional[str]) -> bool:
    if not label:
        return True
    name = field_name.split(".", 1)[1].lower() if "." in field_name else ""
    return name == label


def _matches_any(values: Sequence[float], value: float, decimals: int, cap: float) -> bool:
    return any(_matches(value, known, decimals, cap) for known in values)


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
