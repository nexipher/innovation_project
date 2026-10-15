"""
EvidenceRectifier — one direction authority per token (plan.md §4.10 G3-b).

A token carries a measurement (raw_metric + strength), a calibrated
likelihood, the expert's own claim (`support`), free text (`reasoning`,
`phenomenon`, `interpretation_text`) and usage conditions.  Nothing kept those
in agreement, so one token could assert two opposite things — the G2-e
handover measured frequency_v2 contradicting itself in 15 of 64 samples, and
G2-d showed the model follows whichever field it happens to trust.

The rectifier makes the precedence explicit and enforces it:

    0. **`score_semantics.json` decides the direction** (stage 4.2b): the
       expert's score → direction rule, verified per expert, and able to say
       "this value carries no direction" (NPR's low JPEG scores).  In v3 this
       is the authority, because the calibration table is no longer read;
    1. failing that, `calibrated_likelihood` decides (v1/v2 replay, where the
       mapping exists);
    2. `support` is rewritten to it, the expert's original kept as
       `support_raw`;
    3. free text may not contradict it: offending sentences are replaced by a
       canonical calibration sentence, the measurements and the rest of the
       prose stay;
    4. `applicability_conditions` travel unchanged as the caveat.

Without a calibration entry there is nothing to rectify against, so the
expert's claim survives — marked `expert_claim_uncalibrated` — and the G1
consistency gate remains the only guard.

Direction words use the vocabulary the pipeline already speaks:
"Real" / "AI-generated" / "Uncertain".
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from utils.evidence_consistency import EvidenceConsistencyChecker

# A calibrated likelihood this close to even carries no direction: forcing one
# would turn a near-chance band into a claim.
NEUTRAL_LOW = 0.4
NEUTRAL_HIGH = 0.6

# An explicit claim about what the calibration says; a correct band caveat
# ("this is the band where the two classes separate least") never matches it.
_NEAR_CHANCE_CLAIM = re.compile(
    r"(calibrated likelihood is near chance|near the calibrated chance|"
    r"校准似然[^。.;\n]{0,6}(接近随机|随机水平)|(?:后验|校准)[^。.;\n]{0,4}位于随机水平)", re.I)

SOURCE_CALIBRATED = "calibrated_likelihood"
SOURCE_REGISTRY = "score_semantics"
SOURCE_EXPERT = "expert_claim_uncalibrated"


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

CANONICAL_SENTENCE = {
    "Real": ("Calibration: this measurement band corresponds to camera-source "
             "statistics (P(Fake)={p:.2f}); the benign reading applies."),
    "AI-generated": ("Calibration: this measurement band corresponds to "
                     "generated-source statistics (P(Fake)={p:.2f}); the "
                     "generative reading applies."),
    "Uncertain": ("Calibration: this measurement band is near chance "
                  "(P(Fake)={p:.2f}) and does not determine a direction."),
}

CANONICAL_SENTENCE[SOURCE_REGISTRY] = {
    "Real": ("Score semantics: this expert's score ({raw:.4g}) sits on the "
             "camera side of its recorded threshold ({threshold}); the benign "
             "reading applies."),
    "AI-generated": ("Score semantics: this expert's score ({raw:.4g}) sits on "
                     "the generated side of its recorded threshold "
                     "({threshold}); the generative reading applies."),
    "Uncertain": ("Score semantics: this expert's score ({raw:.4g}) does not "
                  "determine a direction under the recorded conditions "
                  "({note})"),
}


class EvidenceRectifier:
    """Rewrite a token's direction fields to agree with the calibration."""

    @staticmethod
    def direction_from_calibration(token: dict) -> Optional[str]:
        """
        The direction the empirical mapping supports, or None when the token
        carries no calibration entry.
        """
        likelihood = (token.get("calibrated_likelihood") or {}).get("Fake")
        if likelihood is None:
            return None
        likelihood = float(likelihood)
        if likelihood >= NEUTRAL_HIGH:
            return "AI-generated"
        if likelihood <= NEUTRAL_LOW:
            return "Real"
        return "Uncertain"

    @classmethod
    def _direction_from_registry(cls, token: dict) -> "Optional[tuple]":
        """
        The registry's verdict on this token's score, or None when the expert
        is not registered (legacy tokens, mocks, offline experts).

        `raw` is the expert's primary field per the registry — not whichever
        numeric field happens to be present, because noise's two fields differ
        in scale and the wrong one would be read against the wrong threshold.
        """
        from utils.score_semantics import SemanticsError, direction_for_value, entry

        source = token.get("source")
        if not source:
            return None
        try:
            record = entry(source)
        except SemanticsError:
            return None
        field = record["raw_score"]["field"]
        raw = token.get(field)
        if raw is None:
            raw = token.get("raw_metric", token.get("strength"))
        if raw is None:
            return None
        direction, rule = direction_for_value(source, float(raw))
        canonical = CANONICAL_SENTENCE[SOURCE_REGISTRY][direction].format(
            raw=float(raw), threshold=record["threshold"]["value"],
            note=rule.get("note", "该条件下不构成方向"))
        return direction, canonical

    @classmethod
    def _apply_direction(cls, token: dict, direction: str, canonical: str,
                         source: str) -> None:
        """Steps 2–3 of the precedence: support and free text follow the direction."""
        if "support_raw" not in token and token.get("support") != direction:
            token["support_raw"] = token.get("support")
        token["support"] = direction
        token["direction"] = direction
        token["direction_source"] = source
        original_text = token.get("interpretation_text")
        if original_text and original_text != canonical:
            token["interpretation_text_raw"] = original_text
        token["interpretation_text"] = canonical

    @classmethod
    def rectify(cls, token: dict, prefer_registry: bool = False) -> dict:
        """
        Apply the precedence order to one token, in place.

        `prefer_registry` is the caller's policy decision, not a default:
        v3 (no calibration table) must take its direction from the registry;
        **v2 replay must not**, because it has to reproduce the batches it
        produced, and those directions came from the calibration table.

        Returns the same token for call chaining.
        """
        # 0. 登记表（v3 的权威）：分数 → 方向，逐专家核实过，且能表达"该值不带方向"
        registry = cls._direction_from_registry(token) if prefer_registry else None
        if registry is not None:
            direction, canonical = registry
            cls._apply_direction(token, direction, canonical, SOURCE_REGISTRY)
            if direction != "Uncertain":
                cls._correct_near_chance_claims(token, canonical)
            cls._rewrite_contradicting_prose(token, direction, canonical)
            return token

        direction = cls.direction_from_calibration(token)
        if direction is None:
            # Nothing to rectify against; the expert's claim stands, labelled
            # so downstream readers know it was never calibrated.
            token["direction_source"] = SOURCE_EXPERT
            return token

        likelihood = float(token["calibrated_likelihood"]["Fake"])
        canonical = CANONICAL_SENTENCE[direction].format(p=likelihood)
        cls._apply_direction(token, direction, canonical, SOURCE_CALIBRATED)
        if not (NEUTRAL_LOW < likelihood < NEUTRAL_HIGH):
            cls._correct_near_chance_claims(token, canonical)
        cls._rewrite_contradicting_prose(token, direction, canonical)
        return token

    @classmethod
    def _correct_near_chance_claims(cls, token: dict, canonical: str) -> None:
        """
        A token may not *assert* near-chance while its measurement decides.

        The review found "the calibrated likelihood is near chance" sitting next
        to P(Fake)=0.861 in one token: near-chance wording contradicts neither
        direction, so the direction rewrite above cannot see it.
        """
        for field in ("reasoning", "phenomenon"):
            text = token.get(field)
            if not text or not _NEAR_CHANCE_CLAIM.search(str(text)):
                continue
            rewritten = cls._replace_matching_sentences(
                str(text), _NEAR_CHANCE_CLAIM, canonical)
            if rewritten != text:
                token[f"{field}_raw"] = text
                token[field] = rewritten

    @classmethod
    def _rewrite_contradicting_prose(cls, token: dict, direction: str,
                                     canonical: str) -> None:
        """Free text may not argue the other way; the measurement survives."""
        contradictory = cls._contradicting_claims(direction)
        for field in ("reasoning", "phenomenon"):
            text = token.get(field)
            if not text:
                continue
            rewritten = cls._rewrite_sentences(str(text), contradictory, canonical)
            if rewritten != text:
                token[f"{field}_raw"] = text
                token[field] = rewritten

    @staticmethod
    def _contradicting_claims(direction: str):
        """
        Which asserted claim directions disagree with the authority.

        A band that says "Real" cannot be argued with generative wording; one
        that says "AI-generated" cannot be argued with normality wording; a
        near-chance band cannot be argued with either.
        """
        if direction == "Real":
            return {"generative"}
        if direction == "AI-generated":
            return {"normal"}
        return {"generative", "normal"}

    @staticmethod
    def _replace_matching_sentences(text: str, pattern, replacement: str) -> str:
        """Swap the sentences that match `pattern` for one canonical sentence."""
        sentences = _SENTENCE_SPLIT.split(text)
        if len(sentences) < 2:
            return text
        kept, replaced = [], False
        for sentence in sentences:
            if pattern.search(sentence):
                if not replaced:
                    kept.append(replacement)
                    replaced = True
                continue
            kept.append(sentence)
        return " ".join(kept)

    @classmethod
    def _rewrite_sentences(cls, text: str, contradictory, replacement: str) -> str:
        sentences = _SENTENCE_SPLIT.split(text)
        if len(sentences) < 2:
            # A single sentence is the whole statement: replacing it would
            # discard the measurement, so keep it and let the canonical
            # interpretation sentence carry the direction.
            return text
        kept = []
        replaced = False
        for sentence in sentences:
            claim = EvidenceConsistencyChecker._claim_direction(sentence)
            if claim in contradictory and not replaced:
                kept.append(replacement)
                replaced = True
            elif claim in contradictory:
                continue  # one canonical sentence is enough
            else:
                kept.append(sentence)
        return " ".join(kept)
