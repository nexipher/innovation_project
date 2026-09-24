"""
G2-e admission decisions (plan.md §4.9): the experts the runtime ships, the
direction their tokens claim, and the call guidance the prompt gives.

G2-d showed that handing the untuned model contradictory evidence is worse
than handing it none.  These tests pin the fix: every shipped expert's claim
must agree with its G2-b measured direction, and the prompt must say what
each tool can and cannot show.
"""

import numpy as np
import pytest

from experts.frequency_v2 import FrequencyExpertV2
from experts.jpeg import JPEGExpert
from experts.noise import NoiseExpert
from mllm.message_builder import FORENSIC_SYSTEM_PROMPT
from state_machine.controller import ForensicStateMachine

PATCH = (np.random.default_rng(0).integers(0, 255, (128, 128, 3))
         .astype(np.uint8))


class TestMeasuredPolarity:
    """noise/jpeg: G2-b measured high metric ⇒ Real, the opposite of the
    expert's original 'anomaly ⇒ manipulation' sentence map."""

    @pytest.mark.parametrize("expert_class", [NoiseExpert, JPEGExpert])
    def test_metric_polarity_is_inverted(self, expert_class):
        assert expert_class.metric_polarity == -1

    @pytest.mark.parametrize("expert_class", [NoiseExpert, JPEGExpert])
    def test_high_strength_claims_real(self, expert_class):
        support, text = expert_class.classify_metric(0.9)
        assert support == "Real"
        assert "Real" in text or "camera" in text.lower()

    @pytest.mark.parametrize("expert_class", [NoiseExpert, JPEGExpert])
    def test_low_strength_claims_generated(self, expert_class):
        support, _ = expert_class.classify_metric(0.05)
        assert support == "AI-generated"

    @pytest.mark.parametrize("expert_class", [NoiseExpert, JPEGExpert])
    def test_mid_band_stays_uncertain(self, expert_class):
        assert expert_class.classify_metric(0.5)[0] == "Uncertain"

    @pytest.mark.parametrize("expert_class", [NoiseExpert, JPEGExpert])
    def test_high_band_text_does_not_assert_forgery(self, expert_class):
        """The old map said 'severe anomaly matching generative fingerprints'."""
        _, text = expert_class.classify_metric(0.9)
        for claim in ("generative fingerprints", "characteristic of manipulated",
                      "forgery marker"):
            assert claim not in text

    def test_aligned_expert_keeps_the_standard_map(self):
        """freq v2 measured as aligned, so its high band still points to Fake."""
        assert FrequencyExpertV2.metric_polarity == 1
        assert FrequencyExpertV2.classify_metric(0.9)[0] == "AI-generated"

    @pytest.mark.parametrize("expert_class", [NoiseExpert, JPEGExpert])
    def test_reasoning_inverts_with_the_claim(self, expert_class):
        """The narrative must not contradict the corrected support label."""
        result = expert_class().analyze(PATCH)
        assert result.support in ("Real", "AI-generated", "Uncertain")
        if result.strength >= 0.7:
            assert result.support == "Real"
            assert "does not indicate forgery" in result.reasoning or \
                "do not indicate tampering" in result.reasoning


class TestRuntimeRegistration:
    """What the pipeline ships: v2 for frequency, v1 disabled, ELA excluded."""

    def test_dispatch_reaches_v2(self):
        experts = {"frequency_expert_v2": FrequencyExpertV2(),
                   "noise_expert": NoiseExpert(),
                   "jpeg_expert": JPEGExpert()}
        fsm = ForensicStateMachine(None, experts)
        assert fsm._expert_by_call["freq"] == "frequency_expert_v2"

    def test_v1_is_not_dispatchable(self):
        """A registry built the old way must not silently keep working."""
        experts = {"frequency_expert": FrequencyExpertV2(),
                   "noise_expert": NoiseExpert()}
        fsm = ForensicStateMachine(None, experts)
        assert "freq" not in fsm._expert_by_call

    def test_ela_is_not_registered(self):
        """ELA's G2-b skill is compression history: it is deliberately absent."""
        fsm = ForensicStateMachine(None, {"noise_expert": NoiseExpert()})
        assert "ela" not in fsm._expert_by_call
        assert len(fsm.EXPERT_KEY_MAP) == 3

    def test_source_names_are_distinct(self):
        assert FrequencyExpertV2.source_name == "frequency_expert_v2"
        from experts.frequency import FrequencyExpert
        assert FrequencyExpert.source_name == "frequency_expert"
        assert FrequencyExpertV2.source_name != FrequencyExpert.source_name

    def test_call_tags_map_to_shipped_sources(self):
        assert ForensicStateMachine.EXPERT_KEY_MAP == {
            "freq": "frequency_expert_v2",
            "noise": "noise_expert",
            "jpeg": "jpeg_expert",
        }


def _prompt_text() -> str:
    """The prompt is line-wrapped; compare on collapsed whitespace."""
    return " ".join(FORENSIC_SYSTEM_PROMPT.split())


class TestPromptGuidance:
    """The prompt is the model's only interface to the calibration."""

    def test_prompt_states_the_inverted_directions(self):
        text = _prompt_text()
        assert "COUNTER-INTUITIVE" in text
        assert "a HIGH level points to Real" in text
        assert "NOT to forgery" in text

    def test_prompt_makes_calibration_authoritative(self):
        assert "calibrated_likelihood` is the authoritative direction" in \
            _prompt_text()

    def test_prompt_forbids_cross_expert_strength_comparison(self):
        assert "NOT comparable between experts" in _prompt_text()

    def test_prompt_marks_frequency_as_weak(self):
        assert "WEAK evidence" in FORENSIC_SYSTEM_PROMPT

    def test_prompt_keeps_the_format_contract(self):
        for token in ("<call_freq>", "<call_noise>", "<call_jpeg>",
                      "<planning>", "<verdict>", "<reasoning>"):
            assert token in FORENSIC_SYSTEM_PROMPT

    def test_prompt_never_advertises_a_retired_tool(self):
        assert "<call_ela>" not in FORENSIC_SYSTEM_PROMPT
