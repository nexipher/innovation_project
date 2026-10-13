"""
v3 inside a real session (G7 §4.17 R5).

The policy tests cover the decision function; these cover the wiring: which
label reaches `final_verdict`, what the reference is allowed to do to it, and
that the stop reasons are the ones the record claims.
"""

import json

import pytest

from state_machine import controller as controller_module
from state_machine.controller import ForensicStateMachine
from state_machine.halting_v2 import HaltingPolicyV2
from tests.test_controller import FAKE_PATH, ScriptedMLLM, _build_experts

CALL = ("<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n"
        "<call_noise>[200, 100, 300, 280]</call_noise>")
SECOND_CALL = ("<planning>\nSuspected Region: [400, 300, 700, 600]\n</planning>\n"
               "<call_noise>[400, 300, 700, 600]</call_noise>")
GARBAGE = "我认为这张图片看起来很正常，没有发现明显的异常。"


def _verdict(label, confidence=0.8, report="基于已获取的测量给出结论。"):
    return (f'<reasoning>根据现有测量。</reasoning>\n<verdict>'
            f'{{"verdict": "{label}", "confidence": {confidence},'
            f' "primary_evidence": [], "report": "{report}"}}</verdict>')


CLOSING = _verdict("Uncertain", 0.4, "现有测量不足以判定，如实说明。")


@pytest.fixture
def v3(monkeypatch, tmp_path):
    monkeypatch.setattr(controller_module, "HALTING_POLICY", "v3")

    def run(responses):
        from utils.logger import SessionLogger

        fsm = ForensicStateMachine(ScriptedMLLM(responses), _build_experts(),
                                   logger=SessionLogger(sft_dir=str(tmp_path)))
        return fsm.run(FAKE_PATH, "Fake")

    return run


def _trace(result):
    with open(result["sft_data_path"], encoding="utf-8") as handle:
        return json.load(handle)


class TestTheLabelIsTheModels:
    def test_a_conclusion_the_reference_disagrees_with_survives(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "posterior", classmethod(lambda cls, chain: 0.95))
        result = v3([CALL, _verdict("Real", 0.9)])
        assert result["final_verdict"]["verdict"] == "Real"
        assert result["halting_reason"] == "model_concluded"

    def test_the_reference_is_recorded_beside_it(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "posterior", classmethod(lambda cls, chain: 0.95))
        result = v3([CALL, _verdict("Real", 0.9)])
        reference = result["final_verdict"]["reference"]
        assert reference["recommendation"] == "Fake"
        assert "reference_disagrees" in reference["advisory"]
        counters = _trace(result)["metadata"]
        assert counters["model_label"] == "Real"
        assert counters["reference_recommendation"] == "Fake"

    def test_no_override_note_is_sent(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "posterior", classmethod(lambda cls, chain: 0.95))
        result = v3([CALL, _verdict("Real", 0.9)])
        notes = [turn["value"] for turn in result["conversation"]
                 if turn["from"] == "user"]
        assert not any("暂不能结案" in note for note in notes)
        assert not any("与当前校准后验不一致" in note for note in notes)

    def test_uncertain_is_kept_as_the_models_own_conclusion(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "posterior", classmethod(lambda cls, chain: 0.95))
        result = v3([CALL, _verdict("Uncertain", 0.4)])
        assert result["final_verdict"]["verdict"] == "Uncertain"
        advisory = result["final_verdict"]["reference"]["advisory"]
        assert "model_abstained" in advisory and "reference_disagrees" not in advisory


class TestLowGainIsOnlyAdvice:
    def test_a_call_with_no_expected_gain_still_runs(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "tool_utilities",
                            classmethod(lambda cls, *args, **kwargs: {}))
        result = v3([CALL, SECOND_CALL, _verdict("Fake", 0.8)])
        assert result["expert_call_count"] == 1     # the second call is the same expert
        assert result["skipped_duplicate_calls"] == 1
        assert result["halting_reason"] == "model_concluded"

    def test_the_advice_is_sent_once_and_says_it_is_advice(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "tool_utilities",
                            classmethod(lambda cls, *args, **kwargs: {}))
        result = v3([CALL, SECOND_CALL, _verdict("Fake", 0.8)])
        advice = [turn["value"] for turn in result["conversation"]
                  if turn["from"] == "user" and "参考估计" in turn["value"]]
        assert len(advice) == 1
        assert "只是参考" in advice[0]


class TestStopReasons:
    def test_the_same_action_three_times_is_loop_protection(self, v3):
        result = v3([CALL, CALL, CALL, CLOSING])
        assert result["halting_reason"] == "loop_protection"
        assert result["final_verdict"]["verdict"] == "Uncertain"

    def test_turns_without_a_legal_action_are_a_format_failure(self, v3):
        result = v3([GARBAGE, GARBAGE, GARBAGE, CLOSING])
        assert result["halting_reason"] == "format_failure"

    def test_the_budget_leaves_the_label_to_the_closing_turn(self, v3, monkeypatch):
        monkeypatch.setattr(controller_module, "MAX_MODEL_TURNS", 3)
        from state_machine import halting_v3

        monkeypatch.setattr(halting_v3, "MAX_MODEL_TURNS", 3)
        third_call = ("<planning>\nSuspected Region: [500, 400, 600, 500]\n</planning>\n"
                      "<call_noise>[500, 400, 600, 500]</call_noise>")
        # The third turn is still an action, not a conclusion: the budget is
        # what ends the session, and the closing turn supplies the label.
        result = v3([CALL, SECOND_CALL, third_call,
                     _verdict("Real", 0.7, "预算用尽前的结案。")])
        assert result["halting_reason"] == "budget_exhausted"
        assert result["final_verdict"]["verdict"] == "Real"
        conclusion = result["final_verdict"]["conclusion"]
        assert conclusion["source"] == "closing_turn"


class TestConfidenceIsTheModelsOwnNumber:
    def test_a_zero_confidence_is_kept(self, v3):
        result = v3([CALL, _verdict("Fake", 0.0)])
        assert result["final_verdict"]["confidence"] == 0.0

    def test_a_missing_confidence_uses_the_documented_fallback(self, v3):
        result = v3([CALL, '<verdict>{"verdict": "Fake", "primary_evidence": [],'
                           ' "report": "没有写置信度。"}</verdict>'])
        assert result["final_verdict"]["confidence"] == 0.5

    def test_an_impossible_confidence_is_not_believed(self, v3):
        result = v3([CALL, _verdict("Fake", 1.9)])
        assert result["final_verdict"]["confidence"] == 0.5


class TestAFailedConclusionIsNotAnAbstention:
    def test_it_carries_its_own_status(self, v3):
        result = v3([GARBAGE, GARBAGE, GARBAGE, GARBAGE, GARBAGE, GARBAGE])
        assert result["final_verdict"]["verdict_status"] == "no_parseable_conclusion"

    def test_the_abstention_counter_ignores_it(self, v3):
        result = v3([GARBAGE, GARBAGE, GARBAGE, GARBAGE, GARBAGE, GARBAGE])
        assert _trace(result)["metadata"]["model_abstained"] is False

    def test_a_legal_uncertain_does_count_as_abstention(self, v3):
        result = v3([CALL, _verdict("Uncertain", 0.4)])
        metadata = _trace(result)["metadata"]
        assert metadata["verdict_status"] == "model_concluded"
        assert metadata["model_abstained"] is True
