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
    Every way this sample promotes a near-chance measurement.

    Returns one entry per problem: which token, which sentence, which word.
    Empty means the sample reports its weak evidence as weak.
    """
    weak = weak_tokens(sample)
    if not weak:
        return []

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
    return output


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
