"""Tests for the posterior-driven halting policy (G3-c §4.10)."""

import math

import pytest

from config import (
    MAX_EXPERT_CALLS,
    MAX_MODEL_TURNS,
    POLICY_CONFIDENT_POSTERIOR,
)
from state_machine.halting_v2 import HaltingDecision, HaltingPolicyV2

EXPERTS = {"frequency_expert_v2": 0.112, "noise_expert": 0.691, "jpeg_expert": 0.944}


def _token(likelihood, reliability, source="jpeg_expert", direction=None):
    token = {
        "source": source,
        "reliability": reliability,
        "calibrated_likelihood": {"Real": round(1 - likelihood, 3), "Fake": likelihood},
    }
    if direction:
        token["direction"] = direction
    return token


class TestPosterior:
    def test_no_evidence_is_even(self):
        assert HaltingPolicyV2.posterior([]) == pytest.approx(0.5)

    def test_strong_fake_evidence_raises_p_fake(self):
        chain = [_token(0.9, 0.972)]  # jpeg: high P(Fake), strong separation
        assert HaltingPolicyV2.posterior(chain) > 0.6

    def test_real_evidence_lowers_p_fake(self):
        chain = [_token(0.1, 0.972)]
        assert HaltingPolicyV2.posterior(chain) < 0.4

    def test_uncalibrated_token_cannot_move_the_posterior(self):
        """An expert's own claim is not evidence about its own accuracy."""
        chain = [{"source": "x", "support": "AI-generated", "strength": 0.99}]
        assert HaltingPolicyV2.posterior(chain) == pytest.approx(0.5)

    def test_near_chance_expert_barely_moves_it(self):
        weak = HaltingPolicyV2.posterior([_token(0.9, 0.505)])
        strong = HaltingPolicyV2.posterior([_token(0.9, 0.972)])
        assert abs(weak - 0.5) < abs(strong - 0.5)

    def test_conflicting_experts_cancel(self):
        chain = [_token(0.95, 0.972), _token(0.05, 0.9, source="noise_expert")]
        assert HaltingPolicyV2.posterior(chain) == pytest.approx(0.5, abs=0.05)


class TestConflictScore:
    def test_single_token_has_no_conflict(self):
        assert HaltingPolicyV2.conflict_score([_token(0.9, 0.9)]) == pytest.approx(0.0)

    def test_agreement_has_no_conflict(self):
        chain = [_token(0.9, 0.9), _token(0.8, 0.8, source="noise_expert")]
        assert HaltingPolicyV2.conflict_score(chain) == pytest.approx(0.0, abs=1e-9)

    def test_opposite_contributions_conflict_fully(self):
        chain = [_token(0.99, 1.0), _token(0.01, 1.0, source="noise_expert")]
        assert HaltingPolicyV2.conflict_score(chain) == pytest.approx(1.0, abs=0.02)

    def test_uncertain_tokens_do_not_create_conflict(self):
        chain = [_token(0.5, 0.9), _token(0.5, 0.9, source="noise_expert")]
        assert HaltingPolicyV2.conflict_score(chain) == pytest.approx(0.0)

    def test_weights_scale_the_disagreement(self):
        # A loud voice against a whisper is a lopsided disagreement, not a tie.
        chain = [_token(0.95, 0.99), _token(0.05, 0.505)]
        assert HaltingPolicyV2.conflict_score(chain) < 0.5


class TestToolUtilities:
    def test_called_experts_have_no_gain(self):
        utilities = HaltingPolicyV2.tool_utilities([], EXPERTS, {"jpeg_expert"})
        assert utilities["jpeg_expert"] < 0
        assert utilities["noise_expert"] > utilities["jpeg_expert"]

    def test_a_decided_case_makes_tools_less_valuable(self):
        open_case = HaltingPolicyV2.tool_utilities([], EXPERTS, set())
        settled = HaltingPolicyV2.tool_utilities(
            [_token(0.99, 0.99), _token(0.99, 0.99, source="noise_expert")],
            EXPERTS, set())
        assert settled["frequency_expert_v2"] < open_case["frequency_expert_v2"]

    def test_weak_expert_can_be_worth_less_than_its_cost(self):
        weak_only = {"frequency_expert_v2": 0.112}
        utilities = HaltingPolicyV2.tool_utilities([], weak_only, set())
        assert utilities["frequency_expert_v2"] < 0


def _decide(chain, candidate=None, turns=1, calls=0, called=None):
    return HaltingPolicyV2.decide(
        model_turns=turns, expert_calls=calls, evidence_chain=chain,
        candidate_verdict=candidate, available_experts=EXPERTS,
        called_experts=called or set())


class TestDecisions:
    def test_empty_session_explores(self):
        decision = _decide([])
        assert decision.action == "continue"
        assert HaltingPolicyV2.NO_EVIDENCE in decision.all_reasons
        assert decision.next_expert  # recommends one

    def test_a_confident_posterior_accepts_a_matching_candidate(self):
        chain = [_token(0.95, 0.99, source="noise_expert"),
                 _token(0.95, 0.99)]
        decision = _decide(chain, candidate={"verdict": "Fake"})
        assert decision.action == "halt"
        assert decision.verdict == "Fake"
        assert decision.primary_reason == HaltingPolicyV2.CANDIDATE_ACCEPTED

    def test_a_candidate_cannot_bypass_an_open_conflict(self):
        chain = [_token(0.99, 1.0), _token(0.01, 1.0, source="noise_expert")]
        decision = _decide(chain, candidate={"verdict": "Fake"})
        assert decision.action == "continue"
        assert HaltingPolicyV2.CANDIDATE_OVERRIDDEN in decision.all_reasons

    def test_a_candidate_against_the_posterior_is_overridden(self):
        chain = [_token(0.02, 0.99, source="noise_expert")]
        decision = _decide(chain, candidate={"verdict": "Fake"})
        assert decision.action == "continue"
        assert HaltingPolicyV2.CANDIDATE_OVERRIDDEN in decision.all_reasons

    def test_a_candidate_without_enough_evidence_is_overridden(self):
        chain = [_token(0.7, 0.6)]
        decision = _decide(chain, candidate={"verdict": "Fake"})
        assert decision.action == "continue"
        assert HaltingPolicyV2.CANDIDATE_OVERRIDDEN in decision.all_reasons

    def test_budget_exhaustion_alone_never_labels(self):
        chain = [_token(0.55, 0.6)]  # weakly generative, far from confident
        decision = _decide(chain, turns=MAX_MODEL_TURNS)
        assert decision.action == "halt"
        assert decision.verdict == "Uncertain"
        assert decision.primary_reason == HaltingPolicyV2.BUDGET_EXHAUSTED

    def test_budget_exhaustion_may_still_report_a_confident_posterior(self):
        chain = [_token(0.98, 0.99), _token(0.96, 0.99, source="noise_expert")]
        decision = _decide(chain, turns=MAX_MODEL_TURNS)
        assert decision.verdict == "Fake"
        assert decision.primary_reason == HaltingPolicyV2.BUDGET_EXHAUSTED

    def test_exhausted_budget_with_open_conflict_abstains(self):
        chain = [_token(0.99, 1.0), _token(0.01, 1.0, source="noise_expert")]
        decision = _decide(chain, turns=MAX_MODEL_TURNS)
        assert decision.verdict == "Uncertain"
        assert HaltingPolicyV2.CONFLICT_UNRESOLVED in decision.all_reasons

    def test_no_expected_gain_with_a_confident_posterior_labels(self):
        chain = [_token(0.98, 0.99), _token(0.97, 0.99, source="noise_expert")]
        decision = _decide(chain, called=set(EXPERTS))
        assert decision.action == "halt"
        assert decision.primary_reason == HaltingPolicyV2.NO_EXPECTED_GAIN

    def test_continues_when_a_tool_would_still_pay(self):
        decision = _decide([_token(0.8, 0.8)])
        assert decision.action == "continue"
        assert decision.next_expert in EXPERTS
        assert decision.expected_net_utility > 0

    def test_a_repeated_candidate_stops_the_loop(self):
        """
        The policy recommends a tool but cannot compel the model to call it;
        a model that repeats the same rejected verdict must not run the budget
        down (measured: 7 turns before this rule, 5 after).
        """
        chain = [_token(0.4, 0.6)]
        decision = _decide(chain, candidate={"verdict": "Fake"},
                           turns=3, calls=2, called=set(EXPERTS))
        decision_stalled = HaltingPolicyV2.decide(
            model_turns=3, expert_calls=2, evidence_chain=chain,
            candidate_verdict={"verdict": "Fake"}, available_experts=EXPERTS,
            called_experts=set(EXPERTS), turns_without_new_evidence=2)

        assert decision_stalled.action == "halt"
        assert decision_stalled.primary_reason == HaltingPolicyV2.MODEL_STALLED
        assert decision_stalled.verdict == "Uncertain"

    def test_the_stall_rule_still_reports_a_confident_posterior(self):
        chain = [_token(0.99, 0.99), _token(0.98, 0.99, source="noise_expert")]
        decision = HaltingPolicyV2.decide(
            model_turns=3, expert_calls=2, evidence_chain=chain,
            candidate_verdict={"verdict": "Real"}, available_experts=EXPERTS,
            called_experts=set(EXPERTS), turns_without_new_evidence=2)

        assert decision.verdict == "Fake"  # the posterior outranks the candidate
        assert HaltingPolicyV2.MODEL_STALLED in decision.all_reasons

    def test_every_decision_explains_itself(self):
        for chain, candidate, turns in (
            ([], None, 1),
            ([_token(0.99, 1.0), _token(0.01, 1.0, source="noise_expert")], {"verdict": "Fake"}, 1),
            ([_token(0.5, 0.5)], None, MAX_MODEL_TURNS),
        ):
            decision = _decide(chain, candidate=candidate, turns=turns)
            assert decision.all_reasons, decision
            assert isinstance(decision, HaltingDecision)


class TestExpertWeights:
    def test_margin_conversion(self):
        weights = HaltingPolicyV2.expert_weights(
            {"a": 0.972, "b": 0.845, "c": 0.505, "d": None})
        assert weights["a"] == pytest.approx(0.944)
        assert weights["b"] == pytest.approx(0.69)
        assert weights["c"] == pytest.approx(0.01)
        assert "d" not in weights  # no measurement, no weight

    def test_chance_expert_gets_no_weight(self):
        weights = HaltingPolicyV2.expert_weights({"weak": 0.4})
        assert weights["weak"] == 0.0
