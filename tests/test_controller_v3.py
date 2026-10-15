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

pytestmark = pytest.mark.slow

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

    def test_no_reference_is_recorded_and_the_label_survives(self, v3, monkeypatch):
        """
        阶段 4.2：参考后验在 v3 里不再计算。把后验 patch 成极端反对方仍有意义——
        它证明**这个值根本不会被咨询**：标签不变，而且会话里没有参考块可查。
        """
        monkeypatch.setattr(HaltingPolicyV2, "posterior", classmethod(lambda cls, chain: 0.95))
        result = v3([CALL, _verdict("Real", 0.9)])
        verdict = result["final_verdict"]
        assert verdict["verdict"] == "Real"
        assert "reference" not in verdict
        assert "posterior" not in verdict
        counters = _trace(result)["metadata"]
        assert counters["model_label"] == "Real"
        assert "reference_recommendation" not in counters
        assert "posterior" not in counters

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
        verdict = result["final_verdict"]
        assert verdict["verdict"] == "Uncertain"
        # 合法弃权仍是"模型结案"：弃权与否由单独的计数器区分，不是另一个 verdict_status
        assert verdict["verdict_status"] == "model_concluded"
        assert _trace(result)["metadata"]["model_abstained"] is True
        assert "reference" not in verdict

class TestLowGainIsOnlyAdvice:
    def test_a_call_with_no_expected_gain_still_runs(self, v3, monkeypatch):
        monkeypatch.setattr(HaltingPolicyV2, "tool_utilities",
                            classmethod(lambda cls, *args, **kwargs: {}))
        result = v3([CALL, SECOND_CALL, _verdict("Fake", 0.8)])
        assert result["expert_call_count"] == 1     # the second call is the same expert
        assert result["skipped_duplicate_calls"] == 1
        assert result["halting_reason"] == "model_concluded"

    def test_no_reference_estimate_advice_is_sent(self, v3, monkeypatch):
        """
        阶段 4.2：那句"参考估计……只是参考"是**因为算了增益才存在**的。
        计算关掉之后它也不该出现——留着一段解释"某个没算的数只是参考"的文字，
        比删掉它更容易让人以为系统还在算。
        """
        monkeypatch.setattr(HaltingPolicyV2, "tool_utilities",
                            classmethod(lambda cls, *args, **kwargs: {}))
        result = v3([CALL, SECOND_CALL, _verdict("Fake", 0.8)])
        advice = [turn["value"] for turn in result["conversation"]
                  if turn["from"] == "user" and "参考估计" in turn["value"]]
        assert advice == []
        assert result["halting_reason"] == "model_concluded"

class TestStopReasons:
    def test_the_same_action_three_times_is_loop_protection(self, v3):
        result = v3([CALL, CALL, CALL, CLOSING])
        assert result["halting_reason"] == "loop_protection"
        assert result["final_verdict"]["verdict"] == "Uncertain"

    def test_a_re_request_is_counted_as_a_no_progress_action(self, v3):
        """The 41-of-56 pattern: ask again for an expert already measured."""
        again = ("<planning>\nSuspected Region: [500, 400, 600, 500]\n</planning>\n"
                 "<call_noise>[500, 400, 600, 500]</call_noise>")
        result = v3([CALL, again, again, CLOSING])
        assert result["skipped_duplicate_calls"] == 2
        assert result["halting_reason"] == "loop_protection"

    def test_turns_without_a_legal_action_are_a_format_failure(self, v3):
        result = v3([GARBAGE, GARBAGE, GARBAGE, CLOSING])
        assert result["halting_reason"] == "format_failure"

    def test_the_budget_leaves_the_label_to_the_closing_turn(self, v3, monkeypatch):
        monkeypatch.setattr(controller_module, "MAX_MODEL_TURNS", 3)
        from state_machine import halting_v3

        monkeypatch.setattr(halting_v3, "MAX_MODEL_TURNS", 3)
        # Three *distinct* actions: repeating one expert would be a loop, and
        # the loop guard fires before the budget does (by design).
        probe_call = ("<planning>\nSuspected Region: [400, 300, 700, 600]\n</planning>\n"
                      "<call_probe>[400, 300, 700, 600]</call_probe>")
        result = v3([CALL, probe_call, SECOND_CALL,
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

class TestTheSessionStateIsRestated:
    """§4.17: the model should not have to re-derive the state from a blob."""

    def test_every_measurement_is_followed_by_a_state_note(self, v3):
        result = v3([CALL, _verdict("Fake", 0.8)])
        notes = [turn["value"] for turn in result["conversation"]
                 if turn["from"] == "user" and "本会话状态" in turn["value"]]
        assert len(notes) == 1
        assert "已执行的专家" in notes[0] and "noise_expert" in notes[0]
        assert "尚可调用" in notes[0]

    def test_the_note_repeats_only_what_the_pack_delivered(self, v3):
        """
        这一行曾经无条件写 `P(Fake)/raw_metric/适用`：完整投递下没错，精简投递下
        却是把包刚收起来的后台字段又放回去，还和包的方向打架（实测：包说
        AI-generated，这行说"方向 —"）。所以它按投递形状说话。
        """
        result = v3([CALL, _verdict("Fake", 0.8)])
        note = next(turn["value"] for turn in result["conversation"]
                    if turn["from"] == "user" and "本会话状态" in turn["value"])
        assert "该值的方向：" in note
        for background in ("P(Fake)", "raw_metric", "适用 "):
            assert background not in note, background

    def test_the_state_note_lists_what_is_still_available(self, v3):
        result = v3([CALL, _verdict("Fake", 0.8)])
        note = next(turn["value"] for turn in result["conversation"]
                    if turn["from"] == "user" and "本会话状态" in turn["value"])
        assert "jpeg_expert" in note                # served but not yet called
        assert "noise_expert" not in note.split("尚可调用")[1]   # already measured
        assert "重复请求" in note
