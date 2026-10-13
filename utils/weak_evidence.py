"""
Weak evidence, read honestly (G7 entry 4).

The acceptance run found the model doing something the calibration forbids: it
takes a measurement that is near chance (P(Fake) ∈ [0.4, 0.6]) and writes it
into its conclusion as if it pointed somewhere — 30 of 34 such tokens were
quoted with a direction attached, across 28 of 120 sessions.  The label was
unaffected (the posterior decides that), but a corpus built from those traces
teaches the next model to over-read weak numbers, which is exactly the failure
this project spent G2 and G3 removing from the experts themselves.

What is allowed and what is not:

  allowed      "该分数接近随机，略偏向真实，不足以单独支撑判定"
  allowed      naming the number and saying it carries little weight
  flagged      "明确表明图像是伪造的" / "决定性证据" / "强力支持" about a
               near-chance token
  flagged      a near-chance token listed in `primary_evidence` of a confident
               verdict

A flag is a sorting decision, not a verdict: D2 established that an automatic
screener in this project gets to order the queue and nothing more.  The sample
goes to the top of the review pile; a person decides.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

# The band the calibration itself calls "near chance": both outer bins of the
# quantile mapping say more than this, and the middle bin (0.52 for PROBE)
# says nothing.
WEAK_LOW = 0.4
WEAK_HIGH = 0.6

# Strong claims.  A near-chance number may be reported; it may not be promoted.
STRONG_WORDS = ("明确表明", "明确证明", "确凿", "决定性", "决定性证据", "强有力", "有力",
                "强力支持", "充分证明", "毫无疑问", "无可置疑", "确证", "铁证",
                "strong evidence", "conclusive", "decisive", "proves", "confirms")
# ... unless the sentence is denying them, which is the behaviour we want.
# "无法得出明确的结论" and "不能作为决定性证据" are the two shapes the model
# actually uses; both must read as restraint, not as promotion.
DENIALS = ("不构成", "不足以", "不能作为", "不能据此", "并非", "不表示",
           "没有构成", "称不上", "算不上", "无法", "难以", "不能得出")

_SENTENCE_SPLIT = re.compile(r"[。；;\n]")

# How each expert is named in prose — the same vocabulary the acceptance
# review aid uses, because the traces are written in it.
EXPERT_MENTIONS: Dict[str, tuple] = {
    "probe_dino_expert": ("probe", "鉴别器", "判别分数", "判别器", "检测器", "aigi"),
    "frequency_expert_v2": ("频域", "频率", "频谱", "freq"),
    "noise_expert": ("噪声", "noise"),
    "jpeg_expert": ("jpeg", "压缩"),
}


# The ledger's finding kinds, said in the reviewer's vocabulary.  The audit
# asked for five named checks; four of them are the ledger's, applied to the
# sample's own supervised turns instead of to the runtime conclusion.
LEDGER_REASONS = {
    "misattributed_field": "分数层次混用：原始度量被写成校准概率（或方向说反）",
    "misattributed_value": "数值归属错误：引用了另一位专家的数值",
    "fabricated_expert_result": "虚构工具结果：为该专家的测量编造了结果",
    "fabricated_value": "数值在任何 token 中都不存在",
    "direction_conflict": "校准方向误读：与校准给出的方向相反",
    "direction_overread": "校准方向误读：从近随机数值读出明确方向",
    "text_value_as_measurement": "把自由文本里的数字当作本图测量",
    "skipped_request_as_new_evidence": "把被跳过的重复请求当成一次新测量",
}
# Words that turn a global measurement into a regional finding.
LOCALISERS = ("该区域", "这个区域", "此区域", "局部", "左上", "右上", "左下", "右下",
              "中心区域", "边界处", "边缘区域", "区域内有", "该处", "此处")


def ledger_flags(sample: dict) -> List[dict]:
    """
    The runtime's own authority, applied to every supervised turn.

    Weak-evidence screening used to read only the near-chance band; the audit
    found other classes of the same family — a raw metric called a
    probability, a value borrowed from another expert, a direction read the
    wrong way, a tool whose result was invented.  The ledger already decides
    those, so the screener asks it rather than growing a second vocabulary.
    """
    from utils.evidence_ledger import Ledger

    tokens = sample.get("evidence_chain") or []
    if not tokens:
        return []
    ledger = Ledger.from_session(tokens, served=sample.get("tools_served") or ())
    output = []
    for turn in sample.get("conversations") or []:
        if (turn.get("from") or "").lower() not in ("gpt", "assistant"):
            continue
        for finding in ledger.findings(turn.get("value") or ""):
            output.append({
                "kind": finding["kind"],
                "severity": finding.get("severity", "defect"),
                "source": finding.get("source"),
                "evidence_id": finding.get("evidence_id"),
                "sentence": (finding.get("sentence") or "")[:200],
                "reason": LEDGER_REASONS.get(finding["kind"], finding["kind"]),
            })
    return output


def localisation_flags(sample: dict) -> List[dict]:
    """A global measurement reported as a finding about a place in the image."""
    tokens = [token for token in (sample.get("evidence_chain") or [])
              if token.get("measurement_scope") == "global"]
    if not tokens:
        return []
    output = []
    for turn in sample.get("conversations") or []:
        if (turn.get("from") or "").lower() not in ("gpt", "assistant"):
            continue
        for sentence in _SENTENCE_SPLIT.split(turn.get("value") or ""):
            if not any(word in sentence for word in LOCALISERS):
                continue
            for token in tokens:
                if _mentions(sentence, token.get("source") or ""):
                    output.append({
                        "kind": "global_evidence_localised",
                        "severity": "candidate",
                        "source": token.get("source"),
                        "evidence_id": token.get("evidence_id"),
                        "sentence": sentence.strip()[:200],
                        "reason": "整图测量被写成区域结论",
                    })
    return output


_VERDICT_BLOCK = re.compile(r"<verdict>.*?</verdict>", re.S | re.I)


def _stated_direction(text: str) -> Optional[str]:
    """
    The direction the report's own prose states, or None.

    Weaker than the runtime's check on purpose: a closing report may assert a
    direction without an inference verb ("图像是真实的"), and here the
    question is only whether it agrees with the label it is filed under.  The
    <verdict> block is removed first — the label is compared separately, and
    its JSON would otherwise cancel every detection.
    """
    from utils.evidence_ledger import (CLAUSE_SPLIT, MENTION_FAKE, MENTION_REAL,
                                       NEGATED_ASSERTION, _mask_sources)

    prose = _VERDICT_BLOCK.sub(" ", text or "")
    real = fake = 0
    for clause in CLAUSE_SPLIT.split(_mask_sources(prose)):
        if any(marker in clause for marker in NEGATED_ASSERTION):
            continue
        mentions_real = any(word in clause for word in MENTION_REAL)
        mentions_fake = any(word in clause for word in MENTION_FAKE)
        if mentions_real and not mentions_fake:
            real += 1
        elif mentions_fake and not mentions_real:
            fake += 1
    if real and not fake:
        return "Real"
    if fake and not real:
        return "Fake"
    return None


def label_report_flags(sample: dict) -> List[dict]:
    """The label says one thing and the report's own direction says another."""
    verdict = (sample.get("final_verdict") or {}).get("verdict")
    if verdict not in ("Real", "Fake"):
        return []
    assistant = [turn.get("value") or "" for turn in sample.get("conversations") or []
                 if (turn.get("from") or "").lower() in ("gpt", "assistant")]
    if not assistant:
        return []
    argued = _stated_direction(assistant[-1])
    if argued is not None and argued != verdict:
        return [{
            "kind": "label_report_conflict",
            "severity": "defect",
            "source": None,
            "evidence_id": None,
            "sentence": assistant[-1][:200],
            "reason": f"结案标签 {verdict} 与结案文本断言的 {argued} 方向相反",
        }]
    return []


def repeat_request_flags(sample: dict) -> List[dict]:
    """Turns that re-requested a measurement already delivered (candidate)."""
    turns = sample.get("fidelity") or {}
    indexes = ((turns.get("dialogue_audit") or {}).get("repeat_request_turns") or [])
    if not indexes:
        return []
    return [{
        "kind": "repeat_request_turn",
        "severity": "candidate",
        "source": None,
        "evidence_id": None,
        "sentence": f"第 {index} 轮重复请求了已测量过的专家（系统已跳过执行）",
        "reason": "重复请求轮：无进展动作，不应作为动作示范",
    } for index in indexes]


def weak_tokens(sample: dict) -> List[dict]:
    """The tokens whose calibrated likelihood sits in the near-chance band."""
    output = []
    for token in sample.get("evidence_chain") or []:
        likelihood = (token.get("calibrated_likelihood") or {}).get("Fake")
        if likelihood is None:
            continue
        if WEAK_LOW <= float(likelihood) <= WEAK_HIGH:
            output.append(token)
    return output


def _mentions(sentence: str, source: str) -> bool:
    lowered = sentence.lower()
    return any(word in lowered for word in EXPERT_MENTIONS.get(source, ()))


def _promotes(sentence: str) -> Optional[str]:
    """The strong word used, unless the sentence denies it."""
    lowered = sentence.lower()
    if any(denial in lowered for denial in DENIALS):
        return None
    for word in STRONG_WORDS:
        if word in lowered:
            return word
    return None


def flags(sample: dict) -> List[dict]:
    """
    Every way this sample mishandles its evidence.

    Two families: the near-chance promotion this module started with (which
    token, which sentence, which word), and the ledger's own checks run over
    every supervised turn — score-layer mixing, direction misreading,
    fabricated tools, free-text numbers, a global measurement localised, and a
    label that contradicts its own report.

    Empty means no known pattern matched.  It is not a certificate: the
    checks cover what has been defined, and a person still reads the sample.
    """
    structural = (ledger_flags(sample) + localisation_flags(sample)
                  + label_report_flags(sample) + repeat_request_flags(sample))
    weak = weak_tokens(sample)
    if not weak:
        return structural

    prose = []
    for turn in sample.get("conversations") or []:
        if (turn.get("from") or "").lower() in ("gpt", "assistant"):
            prose.append(turn.get("value") or "")
    text = "\n".join(prose)

    output = []
    for token in weak:
        source = token.get("source") or ""
        probability = (token.get("calibrated_likelihood") or {}).get("Fake")
        for sentence in _SENTENCE_SPLIT.split(text):
            if not _mentions(sentence, source):
                continue
            word = _promotes(sentence)
            if word:
                output.append({
                    "evidence_id": token.get("evidence_id"),
                    "source": source,
                    "p_fake": round(float(probability), 3),
                    "word": word,
                    "sentence": sentence.strip()[:200],
                    "reason": "近随机证据被写成强/决定性依据",
                })

    # A confident verdict resting on a near-chance token: the same overclaim,
    # one level up — the sample's own summary says this proves it.
    verdict = sample.get("final_verdict") or {}
    confidence = verdict.get("confidence")
    primary = verdict.get("primary_evidence") or []
    weak_ids = {token.get("evidence_id") for token in weak}
    if confidence is not None and float(confidence) >= 0.8:
        for evidence_id in primary:
            if evidence_id in weak_ids:
                output.append({
                    "evidence_id": evidence_id,
                    "source": next((t.get("source") for t in weak
                                    if t.get("evidence_id") == evidence_id), ""),
                    "p_fake": None,
                    "word": "primary_evidence",
                    "sentence": f"置信度 {confidence} 的结论把近随机证据列为主证据",
                    "reason": "近随机证据进入自信结论的主证据",
                })
    return output + structural


def summarise(samples: List[dict]) -> dict:
    """Batch-level view: how many samples lean on weak evidence, and where."""
    per_sample = {sample.get("id"): flags(sample) for sample in samples}
    flagged = {sid: entries for sid, entries in per_sample.items() if entries}
    return {
        "samples": len(samples),
        "samples_with_weak_tokens": sum(1 for s in samples if weak_tokens(s)),
        "flagged": len(flagged),
        "by_source": _count(flagged, "source"),
        "by_word": _count(flagged, "word"),
        "details": flagged,
    }


def _count(flagged: Dict[str, List[dict]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entries in flagged.values():
        for entry in entries:
            counts[str(entry.get(key))] = counts.get(str(entry.get(key)), 0) + 1
    return counts
