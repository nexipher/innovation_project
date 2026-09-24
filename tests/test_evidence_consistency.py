"""Tests for the deterministic evidence-consistency checker (G1 §4.8)."""

import json
import os
import re

import pytest

from state_machine.evidence_tokenizer import EvidenceTokenizer
from utils.evidence_consistency import EvidenceConsistencyChecker

REJECTED_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sft_data", "train", "final", "sft_rejected.json",
)

# The two counterexamples confirmed in the G0 audit (plan.md §4.7).
COUNTEREXAMPLE_CONTRADICTION = "session_20260721_111810_080862af-5aba-41b0-80ed-dab7b7683807"
COUNTEREXAMPLE_DUPLICATE = "session_20260721_111418_00eb3be4-a0cb-4682-96ea-0a074cd7efaa"

# Actual pre-fix expert reasoning that must be rejected at low strength.
OLD_FREQ_REASONING = (
    "This spectral pattern is mathematically consistent with upsampling / "
    "deconvolution grid artifacts common in GAN and Diffusion-based image synthesis."
)
OLD_NOISE_REASONING = (
    "Real camera sensors produce spatially homogeneous micro-noise (shot noise + PRNU). "
    "Splicing, inpainting, or AI-based local editing disrupts this homogeneity, causing "
    "either variance collapse (over-smoothing) or inflation (unnatural texture). The "
    "micro-noise pattern exhibits localised variance anomalies detected via SRM "
    "high-pass filtering."
)


def _token(strength, support, reasoning, phenomenon="phenomenon",
           interpretation_text="Statistical patterns align with normal hardware camera capture."):
    return {
        "evidence_name": "test_evidence",
        "strength": strength,
        "support": support,
        "reasoning": reasoning,
        "phenomenon": phenomenon,
        "interpretation_text": interpretation_text,
    }


class TestDirectionDetection:
    def test_low_strength_with_generative_reasoning_fails(self):
        result = EvidenceConsistencyChecker.check(
            _token(0.011, "Real", OLD_FREQ_REASONING))
        assert not result["ok"]
        assert "low_strength_generative_claim" in result["failures"]

    def test_old_noise_reasoning_at_low_strength_fails(self):
        result = EvidenceConsistencyChecker.check(
            _token(0.0432, "Real", OLD_NOISE_REASONING))
        assert not result["ok"]
        assert "low_strength_generative_claim" in result["failures"]

    def test_high_strength_with_normal_reasoning_fails(self):
        result = EvidenceConsistencyChecker.check(
            _token(0.9, "AI-generated",
                   "The micro-noise pattern exhibits uniform variance; no evidence of "
                   "local editing was found.",
                   interpretation_text="Severe statistical anomaly matching artificial generative fingerprints."))
        assert not result["ok"]
        assert "high_strength_normal_claim" in result["failures"]

    def test_consistent_low_strength_passes(self):
        reasoning = (
            "No significant periodic peaks detected in the high-frequency power spectrum. "
            "The spectral pattern is consistent with natural image content — real camera-"
            "captured images do not exhibit the structured high-frequency periodicity "
            "characteristic of GAN or Diffusion upsampling artifacts."
        )
        result = EvidenceConsistencyChecker.check(_token(0.011, "Real", reasoning))
        assert result["ok"], result["failures"]

    def test_consistent_high_strength_passes(self):
        reasoning = (
            "Strong periodic peaks detected in the high-frequency power spectrum. This "
            "spectral pattern is mathematically consistent with upsampling / deconvolution "
            "grid artifacts common in GAN and Diffusion-based image synthesis."
        )
        result = EvidenceConsistencyChecker.check(
            _token(0.85, "AI-generated", reasoning,
                   interpretation_text="Severe statistical anomaly matching artificial generative fingerprints."))
        assert result["ok"], result["failures"]


class TestStructuralRules:
    def test_support_strength_mismatch(self):
        result = EvidenceConsistencyChecker.check(
            _token(0.9, "Real", "neutral text"))
        assert "support_strength_mismatch" in result["failures"]

    def test_interpretation_strength_mismatch(self):
        result = EvidenceConsistencyChecker.check(
            _token(0.9, "AI-generated", "neutral text",
                   interpretation_text="Statistical patterns align with normal hardware camera capture."))
        assert "interpretation_strength_mismatch" in result["failures"]

    def test_support_reasoning_conflict(self):
        result = EvidenceConsistencyChecker.check(
            _token(0.5, "Real", OLD_FREQ_REASONING,
                   interpretation_text="Mild mathematical distortions noted; localized compression or blurring suspected."))
        assert "support_reasoning_conflict" in result["failures"]


class TestEnforce:
    def test_enforce_demotes_support_and_marks_token(self):
        token = _token(0.011, "Real", OLD_FREQ_REASONING)
        EvidenceConsistencyChecker.enforce(token)
        assert token["support"] == "Uncertain"
        assert token["consistency"]["status"] == "fail"
        assert "low_strength_generative_claim" in token["consistency"]["failures"]

    def test_enforce_leaves_consistent_token_untouched(self):
        token = _token(0.011, "Real", "No significant periodic peaks detected.")
        EvidenceConsistencyChecker.enforce(token)
        assert token["support"] == "Real"
        assert "consistency" not in token


# ---------------------------------------------------------------------------
# Integration: the checker must catch both G0 audit counterexamples.
# ---------------------------------------------------------------------------

def _load_rejected(session_id):
    with open(REJECTED_PATH, encoding="utf-8") as handle:
        records = json.load(handle)
    for record in records:
        if record["id"] == session_id:
            return record
    pytest.skip(f"{session_id} not present in rejection set")


def _extract_evidence(record):
    evidence = []
    for turn in record.get("conversations", []):
        value = turn.get("value")
        if turn.get("from") == "user" and isinstance(value, str) and value.lstrip().startswith("{"):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "source" in parsed:
                evidence.append(parsed)
    return evidence


def _region_pixels_from_legacy(token):
    match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", str(token.get("region", "")))
    return [int(g) for g in match.groups()] if match else [0, 0, 0, 0]


def test_checker_catches_contradiction_counterexample():
    """080862af: strength=0.011 (Real) but reasoning claims GAN grid artifacts."""
    record = _load_rejected(COUNTEREXAMPLE_CONTRADICTION)
    evidence = _extract_evidence(record)
    assert evidence, "counterexample must contain evidence"
    results = [EvidenceConsistencyChecker.check(token) for token in evidence]
    assert any(not result["ok"] for result in results), \
        "consistency checker must flag the contradiction counterexample"


def test_dedup_catches_duplicate_counterexample():
    """00eb3be4: identical noise evidence injected twice with shrinking regions."""
    record = _load_rejected(COUNTEREXAMPLE_DUPLICATE)
    evidence = _extract_evidence(record)
    ids = [
        EvidenceTokenizer.evidence_id(
            token.get("source", ""),
            _region_pixels_from_legacy(token),
            float(token.get("strength", 0.0)),
            token.get("evidence_name", ""),
        )
        for token in evidence
    ]
    assert len(ids) > len(set(ids)), \
        "duplicate-suppression ids must collide on the duplicate counterexample"
