"""
Deterministic evidence-consistency checker (plan.md §4.8 G1).

Guarantees that within one Evidence Token the direction words agree with the
measured strength: an evidence claiming generative artifacts must not carry a
low strength, and an evidence claiming a normal capture must not carry a high
strength. The G0 audit found both failure classes in the legacy A-line data,
so the pipeline now rejects them at record time instead of propagating them
into conversations.

G2-e: the strength rules are written for an expert whose metric rises towards
Fake.  An expert measured to be inverted declares `semantics_aligned=False`
(noise and JPEG: a high metric is a camera/compression-history signature), and
its correct claims are the mirror image of those rules.  The check therefore
flips signs for such tokens — before this, every directional noise/JPEG claim
was demoted to Uncertain, leaving the model with nothing but abstentions.

The checker is deliberately phrase-based (not model-based): it must be cheap,
deterministic and CPU-only.
"""

from typing import List

from config import STRENGTH_THRESHOLD_LOW, STRENGTH_THRESHOLD_HIGH, STRENGTH_TEXT_MAP

# Mirror of the strength→support rule for inverted metrics (G2-b measured).
INVERTED_SUPPORT = {
    "Real": "AI-generated",
    "AI-generated": "Real",
    "Uncertain": "Uncertain",
}

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

# A marker inside a negation is not a claim: the shipped inverted-expert text
# says "This is NOT a forgery marker", which a bare substring test read as an
# assertion of forgery.
NEGATION_CUES = ("not ", "no ", "never ", "without ", "does not", "do not",
                 "cannot", "isn't", "n't ", "rather than", "instead of")
NEGATION_WINDOW = 32


class EvidenceConsistencyChecker:
    """Stateless deterministic direction check for Evidence Tokens."""

    @staticmethod
    def _asserts(text: str, marker: str) -> bool:
        """
        True when the marker occurs at least once outside a negation.

        A negation cue shortly before the marker (or before a previous
        occurrence) means that occurrence is a denial, not a claim; another
        occurrence later in the text may still assert it.
        """
        start = 0
        while True:
            index = text.find(marker, start)
            if index == -1:
                return False
            window = text[max(0, index - NEGATION_WINDOW):index]
            if not any(cue in window for cue in NEGATION_CUES):
                return True
            start = index + 1

    @classmethod
    def _claim_direction(cls, text: str) -> str | None:
        """Return 'generative', 'normal', 'mixed' or None for a text block."""
        lowered = str(text).lower()
        generative = any(cls._asserts(lowered, m) for m in GENERATIVE_MARKERS)
        normal = any(cls._asserts(lowered, m) for m in NORMAL_MARKERS)
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
        support = str(token.get("support", "")).strip()

        inverted = token.get("semantics_aligned") is False
        weak = strength < STRENGTH_THRESHOLD_LOW
        strong = strength >= STRENGTH_THRESHOLD_HIGH

        expected_support = "Real" if weak else ("AI-generated" if strong else "Uncertain")
        if inverted:
            expected_support = INVERTED_SUPPORT[expected_support]
        if support and support != expected_support:
            failures.append("support_strength_mismatch")

        # The one direction word each band cannot claim: a low aligned metric
        # cannot assert a generative finding, a high one cannot assert a
        # normal capture — mirrored for an inverted metric, where a low value
        # is the generative end and a high value the camera end.
        forbidden_weak = "normal" if inverted else "generative"
        forbidden_strong = "generative" if inverted else "normal"

        reasoning_claim = cls._claim_direction(token.get("reasoning", ""))
        if weak and reasoning_claim == forbidden_weak:
            failures.append(f"low_strength_{forbidden_weak}_claim")
        if strong and reasoning_claim == forbidden_strong:
            failures.append(f"high_strength_{forbidden_strong}_claim")

        phenomenon_claim = cls._claim_direction(token.get("phenomenon", ""))
        if weak and phenomenon_claim == forbidden_weak:
            failures.append(f"low_strength_{forbidden_weak}_phenomenon")
        if strong and phenomenon_claim == forbidden_strong:
            failures.append(f"high_strength_{forbidden_strong}_phenomenon")

        # Only the aligned experts draw their interpretation text from the
        # canonical map; inverted ones use their own mirrored wording, which
        # never matches here and so cannot be judged by it.
        expected_text = (
            STRENGTH_TEXT_MAP["low"] if weak
            else (STRENGTH_TEXT_MAP["high"] if strong else STRENGTH_TEXT_MAP["medium"])
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
