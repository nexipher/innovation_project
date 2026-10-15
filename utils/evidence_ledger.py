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
# "PROBE 原始分数 0.3431" states a value with no verb at all — the number
# follows the quantity's own name, which is a claim just as much as
# "得分为 0.3431" is.
CUE_VALUE = re.compile(r"(得分|分数|原始|度量|读数|水平|指标|概率|可能性|似然)"
                       r"\s*(?:为|是|等于|=|约|≈)?\s*[约≈]?\s*\d")

RESULT_VERBS = ("显示", "表明", "给出", "得到", "返回", "指出", "说明", "测得", "读数",
                "得分", "结果为", "水平为", "输出的", "报告了", "显示出")
# What makes a clause *not* a claim: the act of measuring is what is negated
# ("未获得结果", "没有调用").  A negative *result* is still a claim —
# "noise 专家显示没有相机噪声" says a measurement was taken and came back
# clean, so a bare "没有" must not exempt the clause (it did: a session where
# noise never ran passed with that sentence).
NON_EXECUTION_MARKERS = (
    "未调用", "没有调用", "未执行", "尚未执行", "未做", "尚未做",
    "未获得", "没有获得", "未得到", "没有取得", "无法获得", "无法得到",
    "无法给出", "不能给出", "未能获得", "未产生", "未返回", "未产生结果",
    "尚未获得", "尚未", "跳过", "已跳过",
    "建议调用", "可以调用", "应该调用", "需要调用", "需要先调用",
    "若调用", "如果调用", "假设", "假定", "可能会得到", "将会",
)
# The same idea as a pattern: "没有再次测量" negates the *act*.  A bare "没有"
# cannot be a marker (a negative result is still a measurement), but
# "没有 + (再次) + 测量/调用/获得" is a statement about the act.
NON_EXECUTION_PATTERN = re.compile(
    r"(没有|尚未|未曾|不曾|未|无法|不能)\s*(再次|重新|重复)?\s*"
    r"(测量|调用|执行|得到|获得|取得|产生|返回|读取|给出)")
# A finding's severity: a defect triggers the one correction and can fail the
# session; a candidate is recorded for a person and never blocks on its own.
DEFECT = "defect"
CANDIDATE = "candidate"

SOURCE_WORDS: Dict[str, tuple] = {
    # 探针 is how the model actually names it in Chinese prose; leaving it out
    # meant eight of the audit's sentences were invisible to the check.
    "probe_dino_expert": ("probe", "探针", "鉴别器", "判别分数", "判别器", "检测器",
                          "探测器", "aigi"),
    # 阶段 4.4 起 NPR / UnivFD 就是默认工具箱里的工具，而这里曾经只有旧四家：
    # 模型写"NPR 的分数是 0.93"时，整句话对账本是**不可见**的——不查数字、不查方向、
    # 不查编造。名字按提示词给模型的写法收（display_name 里的那些词）。
    "npr_expert": ("npr", "邻域", "上采样", "upsampling", "neighboring pixel"),
    "univfd_expert": ("univfd", "univ-fd", "clip", "线性头"),
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

# --- direction (what a measurement is used to infer) -----------------------
#
# The binding above checks *attribution*: right expert, right field, right
# value.  It cannot see the other failure the review found — quoting a real
# number and reading the opposite direction out of it ("PROBE 原始分数
# 0.3431，表明图像更可能真实").  That is a separate check with its own two
# verdicts, because the two cases differ in what they mean.
# Direction is read *clause by clause*, and only from a clause that draws an
# inference ("表明/说明/指向/支持/意味着"): listing a measurement is not an
# assertion about it.  A clause that negates the inference, hedges it, or
# mentions both directions is left alone — the check would rather miss a
# finding than invent one (22 flags on real traces were mostly this mistake
# before these rules: see the replay).
INFER_CONNECTIVES = ("表明", "说明", "指向", "支持", "意味着", "证实", "证明", "暗示",
                     "推断", "得出")
MENTION_REAL = ("真实", "自然", "real")
# Bare "生成" is not a generation claim ("由相机传感器生成的"), so the Fake
# vocabulary names the AI-generation phrasings instead.
MENTION_FAKE = ("伪造", "ai生成", "合成", "生成图像", "生成图", "生成痕迹",
                "人工生成", "fake")
NEGATED_ASSERTION = ("并不", "不直接", "不指向", "未指向", "不应", "不能", "难以",
                     "无法", "不足", "没有足够", "未提供", "不明确", "不确定",
                     "并没有", "尚不能", "很难", "未发现", "不足以",
                     "而不是", "并非", "不是", "没有", "无")
# Detector names contain "AI 生成" without asserting a generation claim.
DETECTOR_PHRASE = re.compile(
    r"(?:ai\s*生成|aigi|agi)?\s*(?:探测器|检测器|鉴别器|判别器|鉴定器|分级器)")
# Carrying a citation into the next sentence only makes sense when that
# sentence is *about* it — "。这表明图像更可能真实。" — not when it opens a
# new topic ("然而，由于图像中存在一些视觉异常…这可能暗示了伪造").
ANAPHORIC_START = re.compile(
    r"^\s*[（(]?\s*(?:这|那|它|其|该|上述|此|据此|由此|因此|故而)(?!外|时)")
# A weak number may be reported and may lean; it may not be turned into a
# direction.  These markers are the allowed hedging (entry 4's policy).
HEDGES = ("接近随机", "接近于随机", "略偏", "略倾向于", "不足以", "不能单独",
          "无法单独", "轻微的", "有限的", "不确定性", "保留", "谨慎", "弱证据",
          "仅能", "只能说明", "不能确定", "无法确定", "不足以确定", "相当",
          "差距不大", "无明显差异")
# A calibrated direction is only decisive enough to contradict when it lies
# outside the band the calibration itself calls near chance (utils.weak_evidence
# uses the same numbers).  Inside it, a definite direction is an over-read and
# goes to a person, not to a correction.
NEAR_CHANCE_LOW = 0.4
NEAR_CHANCE_HIGH = 0.6

TOKEN_TEXT_FIELDS = ("reasoning", "interpretation_text", "phenomenon",
                     "counter_explanation", "evidence_name")
# Sub-metrics an expert names in its own prose ("blockiness=0.8779").  Quoting
# one *with its name* is quoting the expert; lifting the bare number is a role
# question for a person, because prose also carries calibration statistics and
# thresholds that are not measurements of this image.
NAMED_TEXT_VALUE = re.compile(r"([A-Za-z][A-Za-z _\-]{1,24}?)\s*[:=]\s*(\d+(?:\.\d+)?)")
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
    # 模型当时**真的能看到**的字段（阶段 4.3）。引用了此表以外的字段值，
    # 就是把没给它看过的东西说成测量——数值可能千真万确，引用仍不成立。
    delivered: Dict[str, float] = field(default_factory=dict)
    text_values: List[float] = field(default_factory=list)
    # What this measurement supports, and on what basis.  `support_label` when
    # there is no calibration: the direction is then the expert's own label,
    # polarity-corrected, and much weaker ground for a conflict claim.
    direction: Optional[str] = None          # "Fake" / "Real" / None
    direction_gap: Optional[float] = None    # |P(Fake) - P(Real)| when calibrated
    direction_basis: str = ""
    named_text_fields: Dict[str, float] = field(default_factory=dict)

    def probability_fields(self) -> Dict[str, float]:
        return {name: value for name, value in self.fields.items()
                if name.startswith("calibrated_likelihood.")}

    def metric_fields(self) -> Dict[str, float]:
        return {name: value for name, value in self.fields.items()
                if not name.startswith("calibrated_likelihood.")}


def _registry_field(source: str) -> str:
    """登记表说这个专家的哪一个字段是"那个数"（未登记则退回 raw_metric）。"""
    try:
        from utils.score_semantics import entry

        return entry(source)["raw_score"]["field"]
    except Exception:
        return "raw_metric"


def is_slim_pack(token: dict) -> bool:
    """这条记录是不是**给模型看的**精简包（阶段 4.3）：字段名不同，数值在 score 里。"""
    return (isinstance(token, dict) and isinstance(token.get("expert"), str)
            and "source" not in token
            and ("score" in token or "reading" in token or "status" in token))


def ledger_view(token: dict) -> dict:
    """
    一条**投递记录** → 账本看的形状。

    运行时喂进来的是证据链上的完整 token（`source`/`raw_metric`/…），转换器的
    `audit_dialogue` 喂的却是 trace 里**当时真正投递的那段 JSON**——阶段 4.3 起
    那是一份精简包，用的是给模型看的字段名（`expert`/`scope`/`score.value`），而且
    没有 `support`/`calibrated_likelihood` 可读方向。账本只认完整 token 的话，
    精简会话里的每一条记录都会在 `source` 上落空、整条被跳过；于是同一句完全正确的
    引用（"PROBE 的分数 0.93 指向 AI 生成"）被判成 `fabricated_expert_result`
    ——实测如此，新主线的每一个样本都会倒在审核这一关。

    所以这里把精简包翻成内部形状：`expert`→`source`，`score.value`→登记表指定的那个
    字段（不是写死的 raw_metric），方向取自**登记表逐值规则**（比 support 标签硬）。
    失败包的形状同样要认：它没有 score，于是 `delivered` 为空——"跑过但什么都没给"
    与"根本没跑"是两件事，账本必须分得开。
    """
    if not is_slim_pack(token):
        return token
    source = str(token["expert"])
    score = token.get("score") if isinstance(token.get("score"), dict) else {}
    view = {
        "source": source,
        "evidence_id": token.get("evidence_id"),
        "evidence_name": score.get("name") or token.get("measurement"),
        "measurement_scope": token.get("scope"),
        "status": token.get("status", "measured"),
        "model_delivered": "slim",          # 投递方式就写在投递记录里
        "direction_for_this_value": score.get("direction_for_this_value"),
        "phenomenon": token.get("reading") or "",
    }
    value = score.get("value")
    if isinstance(value, (int, float)):
        view[_registry_field(source)] = float(value)
    return {key: item for key, item in view.items() if item is not None}


def _delivered_fields(token: dict) -> Optional[Dict[str, float]]:
    """
    模型当时能看到的数值字段。

    `model_delivered` 是投递时盖的戳：`slim`（只给一个主分数，阶段 4.3 起）、
    `full`（v2 复算与旧批次：整个 token）、`none`（本轮不注入证据）。
    没有戳的 token 一律按 `full` 算——它们正是旧批次，当时投递的就是完整 token。
    """
    mode = token.get("model_delivered")
    if mode is None or mode == "full":
        return None                      # None = 调用方按"全部字段"处理
    if mode == "none":
        return {}
    source = token.get("source")
    if not source:
        return {}
    try:
        from utils.score_semantics import SemanticsError, entry

        field = entry(source)["raw_score"]["field"]
    except Exception:
        field = "raw_metric"
    value = token.get(field)
    if value is None:
        value = token.get("raw_metric", token.get("strength"))
    return {field: float(value)} if isinstance(value, (int, float)) else {}


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
            token = ledger_view(token)
            source = token.get("source")
            if not source:
                continue
            record = ExpertRecord(source=source,
                                  evidence_id=str(token.get("evidence_id") or ""))
            for key in ("raw_metric", "strength", "reliability"):
                value = token.get(key)
                if isinstance(value, (int, float)):
                    record.fields[key] = float(value)
            delivered = _delivered_fields(token) if _delivered_fields(token) is not None \
                else dict(record.fields)
            record.delivered = delivered
            for name, value in (token.get("calibrated_likelihood") or {}).items():
                if isinstance(value, (int, float)):
                    record.fields[f"calibrated_likelihood.{name}"] = float(value)
            for name, value in (token.get("condition_metadata") or {}).items():
                if isinstance(value, (int, float)):
                    record.fields[f"condition_metadata.{name}"] = float(value)
            for key in TOKEN_TEXT_FIELDS:
                text = str(token.get(key) or "")
                record.text_values.extend(_numbers(text))
                for name, literal in NAMED_TEXT_VALUE.findall(text):
                    record.named_text_fields[name.strip().lower()] = float(literal)
            (record.direction, record.direction_gap,
             record.direction_basis) = _token_direction(token)
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
        """
        Every way `text` says something the ledger does not support.

        A finding carries a `severity`: `defect` findings are the ones the
        correction loop acts on; `candidate` findings are recorded for a
        person.  Not finding anything is **not** a certificate of fidelity —
        the checks cover the patterns that have been defined and nothing else.
        """
        output: List[dict] = []
        recent: List[str] = []
        for sentence in SENTENCE_SPLIT.split(text or ""):
            sentence = sentence.strip()
            if not sentence:
                continue
            claims, cited = self._sentence_findings(sentence)
            output.extend(claims)
            # A direction is often asserted in the sentence after the number
            # ("…0.3431。这表明图像更可能真实。"), so a citation is carried one
            # sentence forward — but only into a sentence that names no expert
            # of its own, or "然而，噪声残留专家的得分…" would be judged
            # against PROBE from the sentence before it.
            if cited:
                sources = list(cited)
            elif recent and not _named_sources(sentence) and ANAPHORIC_START.match(sentence):
                sources = list(recent)
            else:
                sources = []
            output.extend(self._direction_findings(sentence, sources))
            recent = list(cited)
        return _dedupe(output)

    # -- sentence level -------------------------------------------------

    def _sentence_findings(self, sentence: str) -> Tuple[List[dict], List[str]]:
        """Findings in one sentence, and the experts whose values it cites."""
        cleaned = COORDINATE_BLOCK.sub(" ", sentence)
        cleaned = DIMENSION.sub(" ", CALL_TAG.sub(" ", cleaned))
        cleaned = ENUMERATOR.sub(r"\1", cleaned)
        clauses = [clause.strip() for clause in CLAUSE_SPLIT.split(cleaned) if clause.strip()]
        findings: List[dict] = []
        subjects: List[str] = []
        cited: List[str] = []
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
                    "severity": DEFECT,
                    "sentence": clause[:220],
                    "why": f"{source} 在本会话没有产生任何 token，句子却陈述了它的结果",
                })
            skipped = self._skipped_as_new(clause, subjects)
            skipped_here.update(finding["source"] for finding in skipped)
            findings.extend(skipped)
            ran_subjects = [source for source in subjects if self.ran(source)]
            if ran_subjects:
                clause_findings, clause_cited = self._numbers_in(clause, ran_subjects)
                findings.extend(clause_findings)
                cited.extend(clause_cited)
        # A skipped expert is unrun *and* was asked twice; when the clause
        # describes a re-measurement, the precise finding is the skipped one,
        # and reporting both double-counts the same defect.
        kept = [finding for finding in findings
                if not (finding["kind"] == "fabricated_expert_result"
                        and finding.get("source") in skipped_here)]
        return kept, cited

    # -- direction ------------------------------------------------------

    def _direction_findings(self, sentence: str, cited: Sequence[str]) -> List[dict]:
        """
        What the sentence *infers* from the values it cites.

        Attribution can be perfect and the inference still wrong: quoting
        PROBE's real raw score and reading "更可能真实" out of it passed every
        check the ledger had.  A calibrated direction outside the near-chance
        band contradicted in the same breath is a defect; a direction asserted
        out of a near-chance value is an over-read and goes to a person.
        """
        if not cited:
            return []
        asserted = _asserted_direction(sentence)
        if not asserted:
            return []
        if any(hedge in sentence for hedge in HEDGES):
            # The sentence keeps the measurement in proportion; entry 4's
            # policy allows that, whichever side of the band it is on.
            return []
        findings: List[dict] = []
        for source in cited:
            record = self.measured.get(source)
            if record is None:
                continue
            if record.direction is None:
                # 数值**不承载方向**（登记表逐值规则判 Uncertain，例如 JPEG 格里的
                # NPR 低分），句子却从它推出了方向。这正是用户 2026-10-15 划的那条
                # 界线："低分不得解释成强 Real 证据"——交人工裁决，不自动改。
                if record.direction_basis == "registry_rule:no_direction":
                    findings.append({
                        "kind": "direction_from_neutral_value", "source": source,
                        "severity": CANDIDATE, "asserted": asserted,
                        "evidence_id": record.evidence_id,
                        "sentence": sentence[:220],
                        "why": (f"{source} 的这个数值在登记表里不承载方向"
                                f"（Uncertain），句子却断言了 {asserted}；交人工裁决"),
                    })
                continue
            if record.direction == asserted:
                continue
            gap = record.direction_gap
            if record.direction_basis != "calibrated_likelihood":
                basis = ("登记表的逐值方向规则" if record.direction_basis == "registry_rule"
                         else "support 标签（未经校准）")
                findings.append({
                    "kind": "direction_conflict", "source": source,
                    "severity": CANDIDATE, "asserted": asserted,
                    "evidence_id": record.evidence_id,
                    "sentence": sentence[:220],
                    "why": (f"{source} 的方向来自{basis}，"
                            f"与句中断言的 {asserted} 相反；交人工裁决"),
                })
                continue
            if gap is not None and gap < (NEAR_CHANCE_HIGH - NEAR_CHANCE_LOW):
                findings.append({
                    "kind": "direction_overread", "source": source,
                    "severity": CANDIDATE, "asserted": asserted,
                    "evidence_id": record.evidence_id,
                    "sentence": sentence[:220],
                    "why": (f"{source} 的校准概率处于近随机箱"
                            f"（|P(Fake)−P(Real)|={gap:.2f}），句子却断言了明确方向"),
                })
                continue
            findings.append({
                "kind": "direction_conflict", "source": source,
                "severity": DEFECT, "asserted": asserted,
                "evidence_id": record.evidence_id,
                "field_direction": record.direction,
                "sentence": sentence[:220],
                "why": (f"{source} 的校准方向是 {record.direction}，"
                        f"句子断言相反方向 {asserted}"),
            })
        return findings

    # -- number checking -------------------------------------------------

    def _numbers_in(self, clause: str,
                    subjects: Sequence[str]) -> Tuple[List[dict], List[str]]:
        cues = _cues(clause)
        system_cue = any(kind == "system" for _, kind in cues)
        positions = _mention_positions(clause, subjects)
        findings: List[dict] = []
        cited: List[str] = []
        for match in NUMBER.finditer(clause):
            cue = _cue_covering(cues, match.start())
            if cue == "self":
                continue
            primary = _nearest_source(positions, match.start(), subjects)
            label = _strict_label(cue, cues)
            strict = label != ""          # "" = the clause calls it no kind of probability
            literal = match.group(1)
            value = float(literal)
            decimals = _decimals(literal)
            if match.group(2):
                value /= 100.0
                # "93.9%" is displayed to a tenth of a *percent*: the rounding
                # precision in the fraction domain is two digits finer.
                decimals += 2
            finding = self._check_number(
                value, decimals, primary, subjects, strict, label, system_cue, clause)
            if finding is not None:
                finding["value"] = value
                finding["sentence"] = clause.strip()[:220]
                findings.append(finding)
            elif primary is not None and primary not in cited:
                # Only a value that resolved cleanly is a *citation*: the
                # direction check is about reading a real measurement the
                # wrong way, and a fabricated number is already its own defect.
                cited.append(primary)
        return findings, cited

    def _check_number(self, value: float, decimals: int, primary: Optional[str],
                      subjects: Sequence[str], strict: bool, label: Optional[str],
                      system_cue: bool, clause: str = "") -> Optional[dict]:
        """
        Judge one number: it belongs to the *nearest* named expert first.

        A sentence may name several experts; letting any mentioned expert's
        fields license the value was still too loose — so the nearest mention
        is the owner, a match on another mentioned expert is `misattributed_
        value` naming it, and a match that exists only in an expert's prose is
        a candidate ("this is a calibration statistic, not a measurement of
        this image").
        """
        metric_cap = METRIC_TOLERANCE_CAP
        prob_cap = PROBABILITY_TOLERANCE_CAP
        wrong_field: Optional[Tuple[str, str]] = None
        text_only: Optional[str] = None

        if primary is not None:
            record = self.measured[primary]
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
                    wrong_field = wrong_field or (primary, swapped)
                if system_cue and _matches_any(self.system_values, value, decimals, prob_cap):
                    return None
                for name, known in record.metric_fields().items():
                    if _matches(value, known, decimals, metric_cap):
                        wrong_field = wrong_field or (primary, name)
                if wrong_field is None:
                    if _names_its_own_submetric(record, clause, value, decimals):
                        return None
                    if _matches_any(record.text_values, value, decimals, metric_cap):
                        text_only = primary
            else:
                # 先看"模型能引用的字段"；命中即通过（这是正常路径）。
                if _matches_any(list(record.delivered.values()), value, decimals, metric_cap):
                    return None
                # 数值对得上，但对的是**没投递给它**的字段（阶段 4.3）：
                # 典型是 noise 的 raw_metric 与 strength 方向不同、量纲不同，
                # 引用另一个字段的数值等于在讨论一个它没见过的量。
                for name, known in record.fields.items():
                    if name in record.delivered:
                        continue
                    if _matches(value, known, decimals, metric_cap):
                        return {
                            "kind": "undelivered_field_reference", "source": primary,
                            "field": name, "severity": DEFECT,
                            "evidence_id": record.evidence_id,
                            "why": (f"{value:g} 是 {primary} 的 {name}，但本轮只向模型"
                                    f"投递了 {sorted(record.delivered) or ['（未投递）']}；"
                                    "引用未被投递的字段不能算忠实引用"),
                        }
                if _matches_any(list(record.fields.values()), value, decimals, metric_cap):
                    return None
                if system_cue and _matches_any(self.system_values, value, decimals, metric_cap):
                    return None
                if _names_its_own_submetric(record, clause, value, decimals):
                    return None
                if _matches_any(record.text_values, value, decimals, metric_cap):
                    text_only = primary

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
                "field": name, "severity": DEFECT,
                "evidence_id": self.measured[source].evidence_id,
                "why": why,
            }
        for other in subjects:
            if other == primary:
                continue
            record = self.measured[other]
            for name, known in record.fields.items():
                if _matches(value, known, decimals,
                            prob_cap if name.startswith("calibrated_likelihood.") else metric_cap):
                    return {
                        "kind": "misattributed_value", "source": other,
                        "field": name, "severity": DEFECT,
                        "evidence_id": record.evidence_id,
                        "why": (f"{value:g} 是 {other} 的 {name}，"
                                "不属于句子最近提到的专家"),
                    }
        for other, record in self.measured.items():
            if other in subjects:
                continue
            for name, known in record.fields.items():
                if _matches(value, known, decimals,
                            prob_cap if name.startswith("calibrated_likelihood.") else metric_cap):
                    return {
                        "kind": "misattributed_value", "source": other,
                        "field": name, "severity": DEFECT,
                        "evidence_id": record.evidence_id,
                        "why": (f"{value:g} 是 {other} 的 {name}，"
                                "不属于句子提到的专家"),
                    }
        if text_only is not None:
            return {
                "kind": "text_value_as_measurement", "source": text_only,
                "severity": CANDIDATE,
                "evidence_id": self.measured[text_only].evidence_id,
                "why": (f"{value:g} 只出现在 {text_only} 的自由文本里"
                        "（可能是校准统计或阈值），不能据此断言它是本图的测量；交人工裁决"),
            }
        return {
            "kind": "fabricated_value", "severity": DEFECT,
            "why": "句子给出的数值在本会话的任何 token 中都不存在",
        }

    def _skipped_as_new(self, clause: str, sources: Sequence[str]) -> List[dict]:
        lowered = clause.lower()
        if not self.skipped:
            return []
        if not any(word in lowered for word in ("再次", "重新", "又一次", "新的测量", "再次测量")):
            return []
        if any(marker in lowered for marker in ("没有再次", "未再次", "没有重新", "未重新",
                                               "并未再次", "不曾再", "不再")):
            return []
        findings = []
        for source in self.skipped:
            if source not in sources:
                continue
            findings.append({
                "kind": "skipped_request_as_new_evidence", "source": source,
                "severity": DEFECT, "sentence": clause[:220],
                "why": "该专家的重复请求被跳过，句子里却当成了一次新的测量",
            })
        return findings


def _named_sources(clause: str) -> List[str]:
    lowered = clause.lower()
    return [source for source, words in SOURCE_WORDS.items()
            if any(word in lowered for word in words)]


def _is_claim(clause: str) -> bool:
    """A statement about a measurement — including a *negative* one."""
    lowered = clause.lower()
    if (not any(verb in lowered for verb in RESULT_VERBS)
            and not ASSIGNMENT.search(clause) and not CUE_VALUE.search(lowered)):
        return False
    if any(marker in lowered for marker in NON_EXECUTION_MARKERS):
        return False
    return not NON_EXECUTION_PATTERN.search(lowered)


def _names_its_own_submetric(record: ExpertRecord, clause: str, value: float,
                             decimals: int) -> bool:
    """The clause quotes a named sub-metric of this expert, name and all."""
    lowered = clause.lower()
    return any(name in lowered and _matches(value, known, decimals, METRIC_TOLERANCE_CAP)
               for name, known in record.named_text_fields.items())


def _token_direction(token: dict) -> Tuple[Optional[str], Optional[float], str]:
    """
    What the measurement supports, and on what basis.

    The calibration is the authority when it is there; otherwise the expert's
    own support label is used, polarity-corrected by `semantics_aligned`, and
    marked as the weaker basis it is.
    """
    # 精简包（阶段 4.3）：方向来自登记表的**逐值规则**（`direction_for_this_value`）。
    # 这是我们自己核过的判读，比专家自报的 support 标签硬；但它不是校准概率，
    # 所以自成一类基础，冲突仍交人工裁决。`Uncertain` 是"这个数值不承载方向"。
    pack_direction = token.get("direction_for_this_value")
    if pack_direction:
        if pack_direction == "Uncertain":
            return None, None, "registry_rule:no_direction"
        return ("Fake" if pack_direction == "AI-generated" else "Real"), None, "registry_rule"
    likelihood = token.get("calibrated_likelihood") or {}
    real = likelihood.get("Real")
    fake = likelihood.get("Fake")
    if isinstance(real, (int, float)) and isinstance(fake, (int, float)):
        gap = round(abs(float(fake) - float(real)), 4)
        if float(fake) > float(real):
            return "Fake", gap, "calibrated_likelihood"
        if float(real) > float(fake):
            return "Real", gap, "calibrated_likelihood"
        return None, 0.0, "calibrated_likelihood"
    support = str(token.get("support") or "").strip()
    if support in ("Real", "AI-generated"):
        label = "Real" if support == "Real" else "Fake"
        if token.get("semantics_aligned") is False:
            label = "Fake" if label == "Real" else "Real"
        return label, None, "support_label"
    return None, None, ""


def _mask_sources(text: str) -> str:
    """Blank out expert names so "AI 生成探测器" is not read as a direction."""
    masked = DETECTOR_PHRASE.sub(lambda match: " " * len(match.group(0)), text.lower())
    for words in SOURCE_WORDS.values():
        for word in words:
            masked = masked.replace(word, " " * len(word))
    return masked


def _asserted_direction(text: str) -> Optional[str]:
    """
    The direction the text asserts, or None.

    One class must be mentioned, in a clause that also draws an inference and
    does not negate or hedge it.  Both classes in one clause, or conflicting
    clauses, mean the text is talking about directions rather than asserting
    one — and are not a finding.
    """
    asserted: set = set()
    for clause in CLAUSE_SPLIT.split(_mask_sources(text)):
        if not clause.strip():
            continue
        if not any(conn in clause for conn in INFER_CONNECTIVES):
            continue
        if any(marker in clause for marker in NEGATED_ASSERTION):
            continue
        real = any(word in clause for word in MENTION_REAL)
        fake = any(word in clause for word in MENTION_FAKE)
        if real and not fake:
            asserted.add("Real")
        elif fake and not real:
            asserted.add("Fake")
    return asserted.pop() if len(asserted) == 1 else None


def _mention_positions(clause: str, sources: Sequence[str]) -> List[Tuple[int, str]]:
    lowered = clause.lower()
    positions: List[Tuple[int, str]] = []
    for source in sources:
        for word in SOURCE_WORDS.get(source, ()):
            start = lowered.find(word)
            while start != -1:
                positions.append((start, source))
                start = lowered.find(word, start + 1)
    return sorted(positions)


def _nearest_source(positions: Sequence[Tuple[int, str]], position: int,
                    fallback: Sequence[str]) -> Optional[str]:
    """The expert mentioned closest to this number; a trailing one still counts."""
    before = [source for start, source in positions if start <= position]
    if before:
        return before[-1]
    after = [source for start, source in positions if start > position]
    if after:
        return after[0]
    return fallback[0] if fallback else None


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
