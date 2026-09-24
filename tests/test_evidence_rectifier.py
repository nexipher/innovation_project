"""Tests for the EvidenceRectifier (G3-b): one direction authority per token."""

import pytest

from state_machine.evidence_rectifier import (
    CANONICAL_SENTENCE,
    SOURCE_CALIBRATED,
    SOURCE_EXPERT,
    EvidenceRectifier,
)
from utils.evidence_consistency import EvidenceConsistencyChecker


def _token(likelihood=None, support="AI-generated", strength=0.9,
           interpretation="Severe statistical anomaly matching artificial generative fingerprints.",
           reasoning="Severe anomaly detected. Variance anomalies detected in the block.",
           phenomenon="Anomaly found."):
    token = {
        "evidence_name": "test_evidence",
        "strength": strength,
        "support": support,
        "reasoning": reasoning,
        "phenomenon": phenomenon,
        "interpretation_text": interpretation,
    }
    if likelihood is not None:
        token["calibrated_likelihood"] = {"Real": round(1 - likelihood, 3),
                                          "Fake": likelihood}
    return token


class TestDirectionFromCalibration:
    def test_high_likelihood_is_generative(self):
        assert EvidenceRectifier.direction_from_calibration(
            _token(0.9)) == "AI-generated"

    def test_low_likelihood_is_real(self):
        assert EvidenceRectifier.direction_from_calibration(_token(0.1)) == "Real"

    def test_near_chance_carries_no_direction(self):
        # The neutral band is strictly between the two boundaries; the
        # boundaries themselves take a direction (asserted below).
        assert EvidenceRectifier.direction_from_calibration(
            _token(0.5)) == "Uncertain"
        assert EvidenceRectifier.direction_from_calibration(
            _token(0.41)) == "Uncertain"
        assert EvidenceRectifier.direction_from_calibration(
            _token(0.59)) == "Uncertain"

    def test_boundaries_take_the_direction(self):
        assert EvidenceRectifier.direction_from_calibration(_token(0.6)) == "AI-generated"
        assert EvidenceRectifier.direction_from_calibration(_token(0.39)) == "Real"

    def test_no_calibration_has_no_direction(self):
        assert EvidenceRectifier.direction_from_calibration(_token()) is None


class TestRectify:
    def test_expert_claim_is_overridden_and_kept_for_audit(self):
        token = _token(0.05, support="AI-generated")
        EvidenceRectifier.rectify(token)

        assert token["support"] == "Real"
        assert token["direction"] == "Real"
        assert token["direction_source"] == SOURCE_CALIBRATED
        assert token["support_raw"] == "AI-generated"

    def test_interpretation_is_replaced_by_the_calibration_sentence(self):
        token = _token(0.05)
        original = token["interpretation_text"]
        EvidenceRectifier.rectify(token)

        assert token["interpretation_text"] == CANONICAL_SENTENCE["Real"].format(p=0.05)
        assert token["interpretation_text_raw"] == original

    def test_contradicting_sentence_is_rewritten_and_measurements_survive(self):
        token = _token(0.05, reasoning=(
            "Blockiness measured 0.8977 and DCT anomaly 6.9282. "
            "Variance anomalies detected in the high-frequency band."))
        EvidenceRectifier.rectify(token)

        assert "0.8977" in token["reasoning"] and "6.9282" in token["reasoning"]
        assert "variance anomalies detected" not in token["reasoning"].lower()
        assert CANONICAL_SENTENCE["Real"].format(p=0.05) in token["reasoning"]
        assert token["reasoning_raw"].startswith("Blockiness measured")

    def test_agreeing_text_is_left_alone(self):
        token = _token(0.9, support="Real", reasoning=(
            "Blockiness measured 0.8977. "
            "This is NOT a forgery marker: high values do not indicate tampering."))
        EvidenceRectifier.rectify(token)

        assert token["reasoning"] == (
            "Blockiness measured 0.8977. "
            "This is NOT a forgery marker: high values do not indicate tampering.")
        assert "reasoning_raw" not in token

    def test_uncalibrated_token_keeps_the_expert_claim(self):
        token = _token(support="AI-generated")
        EvidenceRectifier.rectify(token)

        assert token["support"] == "AI-generated"
        assert token["direction_source"] == SOURCE_EXPERT
        assert "direction" not in token

    def test_near_chance_band_says_so_without_gutting_the_prose(self):
        token = _token(0.5, reasoning="Residual level 2.36. The band is close to chance.")
        EvidenceRectifier.rectify(token)

        assert token["support"] == "Uncertain"
        assert token["interpretation_text"] == CANONICAL_SENTENCE["Uncertain"].format(p=0.5)
        assert token["reasoning"] == "Residual level 2.36. The band is close to chance."

    def test_single_sentence_reasoning_is_not_destroyed(self):
        """One sentence is the whole measurement — never trade it for the note."""
        token = _token(0.05, reasoning="Variance anomalies detected.")
        EvidenceRectifier.rectify(token)

        assert token["reasoning"] == "Variance anomalies detected."

    def test_applicability_conditions_travel_unchanged(self):
        token = _token(0.05)
        token["applicability_conditions"] = "高 strength 统计上对应 Real。"
        EvidenceRectifier.rectify(token)

        assert token["applicability_conditions"] == "高 strength 统计上对应 Real。"


class TestRectifiedTokensPassTheGate:
    """After rectification the consistency gate is a verification, not a filter."""

    @pytest.mark.parametrize("likelihood", [0.05, 0.5, 0.95])
    def test_gate_accepts_every_band(self, likelihood):
        token = _token(likelihood, strength=0.9,
                       reasoning="Measured value 3.51. Variance anomalies detected here.")
        EvidenceRectifier.rectify(token)
        result = EvidenceConsistencyChecker.enforce(token)

        assert result["ok"], result["failures"]
        assert "consistency" not in token

    def test_gate_uses_calibration_over_strength_bands(self):
        """
        frequency_v2's two discretisations disagree: strength 0.247 says "low",
        the calibrated band says Fake 0.709.  The calibration must win.
        """
        token = _token(0.709, support="Real", strength=0.247,
                       interpretation="Mild mathematical distortions noted; localized compression or blurring suspected.",
                       reasoning="Spectral energy concentrated at 0.247.")
        EvidenceRectifier.rectify(token)
        result = EvidenceConsistencyChecker.enforce(token)

        assert token["support"] == "AI-generated"
        assert result["ok"], result["failures"]

    def test_gate_still_catches_a_hand_edited_contradiction(self):
        """The verification keeps its teeth for tokens that bypass the rectifier."""
        token = _token(0.05, strength=0.9, reasoning="Variance anomalies detected here.")
        EvidenceRectifier.rectify(token)
        token["support"] = "AI-generated"  # tampered with after rectification

        result = EvidenceConsistencyChecker.enforce(token)
        assert "support_calibration_mismatch" in result["failures"]
