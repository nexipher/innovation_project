"""Unit tests for HaltingChecker."""

import pytest
from state_machine.halting import HaltingChecker


class TestVerdictDetection:
    def test_verdict_present(self):
        text = '<verdict>{"verdict": "Fake", "confidence": 0.9}</verdict>'
        assert HaltingChecker._verdict_detected(text) == "Fake"

    def test_verdict_absent(self):
        assert HaltingChecker._verdict_detected("no verdict here") is None


class TestBudget:
    """G1 (§4.8): budget counts expert calls and model turns separately."""

    def test_expert_budget_not_reached(self):
        assert not HaltingChecker._budget_exhausted(model_turns=2, expert_calls=3)

    def test_expert_budget_reached(self):
        assert HaltingChecker._budget_exhausted(model_turns=3, expert_calls=5)

    def test_expert_budget_exceeded(self):
        assert HaltingChecker._budget_exhausted(model_turns=4, expert_calls=10)

    def test_model_turn_budget_reached(self):
        assert HaltingChecker._budget_exhausted(model_turns=6, expert_calls=1)


class TestConflictDetection:
    def test_no_conflict_single_evidence(self):
        chain = [{"strength": 0.8, "support": "AI-generated"}]
        assert not HaltingChecker._conflict_detected(chain)

    def test_no_conflict_agreeing(self):
        chain = [
            {"strength": 0.8, "support": "AI-generated"},
            {"strength": 0.9, "support": "AI-generated"},
        ]
        assert not HaltingChecker._conflict_detected(chain)

    def test_conflict_detected(self):
        chain = [
            {"strength": 0.9, "support": "AI-generated"},
            {"strength": 0.1, "support": "Real"},
        ]
        assert HaltingChecker._conflict_detected(chain)


class TestInfoGain:
    def test_not_enough_evidence(self):
        assert not HaltingChecker._info_gain_converged([])
        assert not HaltingChecker._info_gain_converged([{"strength": 0.5}])

    def test_converged(self):
        chain = [
            {"strength": 0.80},
            {"strength": 0.81},  # very small delta
        ]
        assert HaltingChecker._info_gain_converged(chain)

    def test_not_converged(self):
        chain = [
            {"strength": 0.1},
            {"strength": 1.0},  # large delta
        ]
        assert not HaltingChecker._info_gain_converged(chain)


class TestCheck:
    def test_verdict_priority(self):
        """Verdict should halt regardless of other conditions."""
        should, reason = HaltingChecker.check(
            model_turns=0,
            expert_calls=0,
            evidence_chain=[],
            last_output='<verdict>{"verdict": "Real", "confidence": 0.9}</verdict>',
        )
        assert should
        assert reason == HaltingChecker.VERDICT_OUTPUT

    def test_budget_priority(self):
        should, reason = HaltingChecker.check(
            model_turns=3,
            expert_calls=5,
            evidence_chain=[],
            last_output="<call_freq>[1,2,3,4]</call_freq>",
        )
        assert should
        assert reason == HaltingChecker.BUDGET_EXHAUSTED

    def test_no_halt_normal(self):
        should, reason = HaltingChecker.check(
            model_turns=1,
            expert_calls=1,
            evidence_chain=[{"strength": 0.5}],
            last_output="<call_noise>[1,2,3,4]</call_noise>",
        )
        assert not should
