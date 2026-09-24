"""
EvidenceRectifier — one direction authority per token (plan.md §4.10 G3-b).

A token carries a measurement (raw_metric + strength), a calibrated
likelihood, the expert's own claim (`support`), free text (`reasoning`,
`phenomenon`, `interpretation_text`) and usage conditions.  Nothing kept those
in agreement, so one token could assert two opposite things — the G2-e
handover measured frequency_v2 contradicting itself in 15 of 64 samples, and
G2-d showed the model follows whichever field it happens to trust.

The rectifier makes the precedence explicit and enforces it:

    1. `calibrated_likelihood` decides the direction (it is the empirical
       mapping for that metric band);
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

SOURCE_CALIBRATED = "calibrated_likelihood"
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
    def rectify(cls, token: dict) -> dict:
        """
        Apply the precedence order to one token, in place.

        Returns the same token for call chaining.
        """
        direction = cls.direction_from_calibration(token)
        if direction is None:
            # Nothing to rectify against; the expert's claim stands, labelled
            # so downstream readers know it was never calibrated.
            token["direction_source"] = SOURCE_EXPERT
            return token

        likelihood = float(token["calibrated_likelihood"]["Fake"])
        canonical = CANONICAL_SENTENCE[direction].format(p=likelihood)

        # 2. the structured measurement wins over the expert's own label
        if "support_raw" not in token and token.get("support") != direction:
            token["support_raw"] = token.get("support")
        token["support"] = direction
        token["direction"] = direction
        token["direction_source"] = SOURCE_CALIBRATED

        # 3. free text may not contradict the measurement.  `interpretation_text`
        # is the expert's direction sentence by construction, so it is replaced
        # outright; the longer prose is edited sentence by sentence so the
        # measurements and the benign-cause discussion survive.
        original_text = token.get("interpretation_text")
        if original_text and original_text != canonical:
            token["interpretation_text_raw"] = original_text
        token["interpretation_text"] = canonical

        contradictory = cls._contradicting_claims(direction)
        for field in ("reasoning", "phenomenon"):
            text = token.get(field)
            if not text:
                continue
            rewritten = cls._rewrite_sentences(str(text), contradictory, canonical)
            if rewritten != text:
                token[f"{field}_raw"] = text
                token[field] = rewritten

        return token

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
