"""
Halting policy v3: the model concludes, the reference advises (G7 §4.17 R5).

The v2 rules this replaces: a candidate the posterior disagreed with was
overridden until the model repeated itself, an `Uncertain` candidate counted
as agreement and then carried the *posterior's* label, and a call with no
expected gain was refused.  The A-segment audit counted 68 sessions whose
model conclusion was silently replaced because of the first two.
"""

import pytest

from config import MAX_EXPERT_CALLS, MAX_MODEL_TURNS
from state_machine.halting_v3 import HaltingPolicyV3

EXPERTS = {"frequency_expert_v2": 0.112, "noise_expert": 0.691, "jpeg_expert": 0.944}


def _token(likelihood, reliability=0.9, source="jpeg_expert"):
    return {
        "source": source,
        "reliability": reliability,
        "calibrated_likelihood": {"Real": round(1 - likelihood, 3), "Fake": likelihood},
    }


def _decide(**overrides):
    kwargs = dict(
        model_turns=2, expert_calls=1,
        evidence_chain=[_token(0.95)],
        candidate_verdict=None,
        available_experts=EXPERTS,
        called_experts={"jpeg_expert"},
    )
    kwargs.update(overrides)
    return HaltingPolicyV3.decide(**kwargs)


class TestTheModelOwnsTheLabel:
    def test_a_conclusion_the_reference_disagrees_with_is_kept(self):
        decision = _decide(candidate_verdict={"verdict": "Real", "confidence": 0.9})
        assert decision.action == "halt"
        assert decision.verdict == "Real", "参考后验不得改写模型标签"
        assert decision.primary_reason == HaltingPolicyV3.MODEL_CONCLUDED
        assert "reference_disagrees" in decision.advisory
        assert decision.reference_label == "Fake"      # recorded, not imposed

    def test_uncertain_is_the_models_conclusion_not_agreement(self):
        decision = _decide(candidate_verdict={"verdict": "Uncertain", "confidence": 0.4})
        assert decision.action == "halt" and decision.verdict == "Uncertain"
        assert "model_abstained" in decision.advisory
        assert "reference_disagrees" not in decision.advisory

    def test_a_matching_conclusion_carries_no_disagreement_note(self):
        decision = _decide(candidate_verdict={"verdict": "Fake", "confidence": 0.8})
        assert decision.verdict == "Fake" and decision.advisory == []

    def test_the_models_confidence_is_reported_as_given(self):
        decision = _decide(candidate_verdict={"verdict": "Fake", "confidence": 0.63})
        assert decision.confidence == 0.63

    def test_confidence_is_never_taken_from_the_posterior(self):
        decision = _decide(candidate_verdict={"verdict": "Fake"})
        assert decision.confidence == 0.5              # the model said nothing


class TestAdvisoryNotAuthority:
    def test_no_expected_gain_is_a_note_and_not_a_stop(self):
        decision = _decide(available_experts={"jpeg_expert": 0.0})
        assert decision.action == "continue"
        assert "reference_no_expected_gain" in decision.advisory

    def test_a_continue_decision_has_no_label_of_its_own(self):
        decision = _decide()
        assert decision.verdict is None and decision.action == "continue"

    def test_with_no_evidence_there_is_no_reference_label(self):
        decision = _decide(evidence_chain=[], expert_calls=0, called_experts=set())
        assert decision.reference_label is None


class TestStopReasonsAreSeparate:
    def test_a_repeated_action_stops_as_loop_protection(self):
        decision = _decide(repeated_actions=HaltingPolicyV3.REPEATED_ACTION_LIMIT)
        assert decision.action == "halt"
        assert decision.primary_reason == HaltingPolicyV3.LOOP_PROTECTION
        assert decision.verdict is None, "停止不等于替模型给出标签"

    def test_invalid_actions_stop_as_a_format_failure(self):
        decision = _decide(invalid_actions=HaltingPolicyV3.INVALID_ACTION_LIMIT)
        assert decision.primary_reason == HaltingPolicyV3.FORMAT_FAILURE
        assert decision.verdict is None

    def test_the_budget_stops_and_still_leaves_the_label_to_the_model(self):
        decision = _decide(model_turns=MAX_MODEL_TURNS)
        assert decision.primary_reason == HaltingPolicyV3.BUDGET_EXHAUSTED
        assert decision.verdict is None

    def test_an_illegal_label_is_a_protocol_problem(self):
        decision = _decide(candidate_verdict={"verdict": "fake", "confidence": 0.9})
        assert decision.action == "continue"
        assert decision.primary_reason == HaltingPolicyV3.ILLEGAL_CONCLUSION

    def test_an_expert_call_budget_stops_too(self):
        decision = _decide(expert_calls=MAX_EXPERT_CALLS)
        assert decision.primary_reason == HaltingPolicyV3.BUDGET_EXHAUSTED

    def test_exploring_comes_before_any_stop_reason(self):
        decision = _decide(evidence_chain=[], expert_calls=0, called_experts=set())
        assert decision.action == "continue"
        assert decision.primary_reason == HaltingPolicyV3.EXPLORING
