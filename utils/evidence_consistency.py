"""
Deterministic evidence-consistency checker (plan.md §4.8 G1).

Guarantees that within one Evidence Token the direction words agree with the
measured strength: an evidence claiming generative artifacts must not carry a
low strength, and an evidence claiming a normal capture must not carry a high
strength. The G0 audit found both failure classes in the legacy A-line data,
so the pipeline now rejects them at record time instead of propagating them
into conversations.

The checker is deliberately phrase-based (not model-based): it must be cheap,
deterministic and CPU-only.
"""

from typing import List

from config import STRENGTH_THRESHOLD_LOW, STRENGTH_THRESHOLD_HIGH, STRENGTH_TEXT_MAP

# Phrases that assert a generative/forgery finding (require high strength).
# They must be assertions, not negations — bare words like "GAN" or
# "upsampling" also appear inside correctly-negated low-strength texts.
GENERATIVE_MARKERS = (
    "artifacts common in",
    "deconvolution grid",
    "matching artificial generative",
    "artificial generative fingerprints",
    "characteristic of manipulated",
    "forgery marker",
    "markers of post-capture",
    "classic markers of",
    "variance anomalies detected",
    "evidence of tampering",
    "evidence of splicing",
)

# Phrases that assert a normal / camera-capture finding (require low strength).
NORMAL_MARKERS = (
    "no significant",
    "within normal",
    "normal range",
    "normal hardware",
    "no evidence of",
    "consistent with natural",
    "consistent with a single camera",
    "do not exhibit",
    "does not exhibit",
)


class EvidenceConsistencyChecker:
    """Stateless deterministic direction check for Evidence Tokens."""

    @staticmethod
    def _claim_direction(text: str) -> str | None:
        """Return 'generative', 'normal', 'mixed' or None for a text block."""
        lowered = str(text).lower()
        generative = any(marker in lowered for marker in GENERATIVE_MARKERS)
        normal = any(marker in lowered for marker in NORMAL_MARKERS)
        if generative and normal:
            return "mixed"
        if generative:
            return "generative"
        if normal:
            return "normal"
        return None

    @classmethod
    def check(cls, token: dict) -> dict:
        """
        Check one Evidence Token for internal direction consistency.

        Returns:
            dict with keys:
              - ok: True when no inconsistency was found;
              - failures: list of machine-readable failure names.
        """
        failures: List[str] = []
        strength = float(token.get("strength", 0.5))
        support = str(token.get("support", ""))

        expected_support = (
            "Real" if strength < STRENGTH_THRESHOLD_LOW
            else ("AI-generated" if strength >= STRENGTH_THRESHOLD_HIGH else "Uncertain")
        )
        if support and support != expected_support:
            failures.append("support_strength_mismatch")

        reasoning_claim = cls._claim_direction(token.get("reasoning", ""))
        if strength < STRENGTH_THRESHOLD_LOW and reasoning_claim == "generative":
            failures.append("low_strength_generative_claim")
        if strength >= STRENGTH_THRESHOLD_HIGH and reasoning_claim == "normal":
            failures.append("high_strength_normal_claim")

        phenomenon_claim = cls._claim_direction(token.get("phenomenon", ""))
        if strength < STRENGTH_THRESHOLD_LOW and phenomenon_claim == "generative":
            failures.append("low_strength_generative_phenomenon")
        if strength >= STRENGTH_THRESHOLD_HIGH and phenomenon_claim == "normal":
            failures.append("high_strength_normal_phenomenon")

        expected_text = (
            STRENGTH_TEXT_MAP["low"] if strength < STRENGTH_THRESHOLD_LOW
            else (STRENGTH_TEXT_MAP["high"] if strength >= STRENGTH_THRESHOLD_HIGH
                  else STRENGTH_TEXT_MAP["medium"])
        )
        interpretation = str(token.get("interpretation_text", ""))
        canonical_texts = set(STRENGTH_TEXT_MAP.values())
        if interpretation in canonical_texts and interpretation != expected_text:
            failures.append("interpretation_strength_mismatch")

        if support == "Real" and reasoning_claim == "generative":
            failures.append("support_reasoning_conflict")
        if support == "AI-generated" and reasoning_claim == "normal":
            failures.append("support_reasoning_conflict")

        return {"ok": not failures, "failures": failures}

    @classmethod
    def enforce(cls, token: dict) -> dict:
        """
        Apply the check to a token in place and return the check result.

        Failing tokens are marked with a `consistency` block and demoted to
        `support = "Uncertain"`, so downstream reasoning can never read a
        low-strength measurement as a generative finding.
        """
        result = cls.check(token)
        if not result["ok"]:
            token["consistency"] = {
                "status": "fail",
                "failures": list(result["failures"]),
            }
            token["support"] = "Uncertain"
        return result
