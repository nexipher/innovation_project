"""
The single-action protocol (G7 §4.17 R9).

One tool request or one conclusion per turn.  The parser decides what a turn
did, the executor executes nothing when the answer is "more than one action",
and the model is told — the pipeline never quietly picks a half.  The bbox is
optional: every expert measures the whole image, so a region is a diagnostic
note, and a model with nothing to point at must not be made to invent one.
"""

import json

import pytest

from state_machine.controller import ForensicStateMachine
from tests.test_controller import FAKE_PATH, ScriptedMLLM, _build_experts
from utils.parser import Parser

CALL = ("<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n"
        "<call_noise>[200, 100, 300, 280]</call_noise>")
BARE_CALL = ("<planning>\nSuspected Region: none\n"
             "Visual Anomalies: 未观察到明确异常，需要统计证据\n"
             "Expert Target & Hypothesis: noise\n</planning>\n"
             "<call_noise></call_noise>")
VERDICT = ('<reasoning>只有噪声测量可用。</reasoning>\n<verdict>'
           '{"verdict": "Uncertain", "confidence": 0.4, "primary_evidence": [],'
           ' "report": "证据有限，如实说明。"}</verdict>')
MIXED = CALL + "\n" + VERDICT
TWO_CALLS = CALL + "\n" + ("<planning>\nSuspected Region: [10, 10, 200, 200]\n</planning>\n"
                           "<call_jpeg>[10, 10, 200, 200]</call_jpeg>")


class TestTheParserDecides:
    def test_a_call_without_a_bbox_is_a_call(self):
        assert Parser.extract_all_calls(BARE_CALL) == [("noise", None)]

    def test_a_call_with_a_bbox_keeps_it(self):
        assert Parser.extract_all_calls(CALL) == [("noise", [200, 100, 300, 280])]

    def test_a_call_and_a_verdict_is_a_violation(self):
        action = Parser.protocol_action(MIXED)
        assert action["violation"] == "call_and_verdict"
        assert len(action["calls"]) == 1 and action["verdict"] is True

    def test_two_calls_in_one_turn_is_a_violation(self):
        assert Parser.protocol_action(TWO_CALLS)["violation"] == "multiple_calls"

    def test_a_pure_conclusion_is_not_a_violation(self):
        action = Parser.protocol_action(VERDICT)
        assert action["violation"] is None and action["verdict"] is True
        assert action["calls"] == []


class TestAPlanningTagIsNotARequest:
    """The acceptance batch's real output: the tag named in the planning prose."""

    REAL_TURN = ("<planning>\n"
                 "Suspected Region: 未观察到明确异常，需要统计证据\n"
                 "Visual Anomalies: 未观察到明确异常，需要统计证据\n"
                 "Expert Target & Hypothesis: <call_probe> 用于检测整体图像的生成器特征\n"
                 "</planning>\n<call_probe>")

    def test_it_is_one_call_and_no_violation(self):
        action = Parser.protocol_action(self.REAL_TURN)
        assert action["violation"] is None
        assert action["calls"] == [("probe", None)]

    def test_a_tag_only_inside_planning_is_not_a_call(self):
        action = Parser.protocol_action(
            "<planning>Expert Target & Hypothesis: <call_noise> 噪声</planning>")
        assert action["calls"] == [] and action["violation"] is None

    def test_strip_planning_leaves_the_real_request(self):
        stripped = Parser.strip_planning(self.REAL_TURN)
        assert "call_probe" in stripped and "Expert Target" not in stripped

    def test_two_real_calls_outside_planning_are_still_refused(self):
        assert Parser.protocol_action(TWO_CALLS)["violation"] == "multiple_calls"

    def test_the_prompt_says_the_tag_appears_once_after_planning(self):
        from mllm.message_builder import build_forensic_prompt

        prompt = build_forensic_prompt(["probe", "noise"])
        assert "The call tag appears EXACTLY ONCE, AFTER </planning>" in prompt
        assert "a <call_x> tag here is read as a second request" in prompt


class TestTheExecutorRefuses:
    def _run(self, responses, tmp_path):
        from utils.logger import SessionLogger

        fsm = ForensicStateMachine(ScriptedMLLM(responses), _build_experts(),
                                   logger=SessionLogger(sft_dir=str(tmp_path)))
        return fsm.run(FAKE_PATH, "Fake")

    def test_a_mixed_turn_executes_nothing_and_says_so(self, tmp_path):
        result = self._run([MIXED, VERDICT], tmp_path)
        assert result["expert_call_count"] == 0
        assert result["final_verdict"]["conclusion"]["source"] != "candidate"
        notes = [turn["value"] for turn in result["conversation"]
                 if turn["from"] == "user" and "协议要求" in turn["value"]]
        assert len(notes) == 1
        with open(result["sft_data_path"], encoding="utf-8") as handle:
            assert json.load(handle)["metadata"]["protocol_violations"] == 1

    def test_two_calls_are_refused_rather_than_halved(self, tmp_path):
        result = self._run([TWO_CALLS, VERDICT], tmp_path)
        assert result["expert_call_count"] == 0
        assert result["unique_evidence_count"] == 0

    def test_the_refusal_is_not_silent(self, tmp_path):
        result = self._run([MIXED, VERDICT], tmp_path)
        assert any("都未执行" in turn["value"] for turn in result["conversation"]
                   if turn["from"] == "user")


class TestTheBboxIsOptional:
    def _run(self, responses, tmp_path):
        from utils.logger import SessionLogger

        fsm = ForensicStateMachine(ScriptedMLLM(responses), _build_experts(),
                                   logger=SessionLogger(sft_dir=str(tmp_path)))
        return fsm.run(FAKE_PATH, "Fake")

    def test_a_call_without_a_bbox_measures_the_whole_image(self, tmp_path):
        result = self._run([BARE_CALL, VERDICT], tmp_path)
        assert result["expert_call_count"] == 1
        token = result["evidence_chain"][0]
        with open(result["sft_data_path"], encoding="utf-8") as handle:
            height, width = json.load(handle)["metadata"]["image_size"]
        assert token["region_normalized_1000"] is None
        assert token["region_pixels"] == [0, 0, height, width]
        assert token["measurement_scope"] == "global"

    def test_nothing_is_invented_for_the_region(self, tmp_path):
        result = self._run([BARE_CALL, VERDICT], tmp_path)
        # the diagnostic region is recorded as absent, not as a made-up box
        assert "none" not in json.dumps(result["evidence_chain"][0]["region_normalized_1000"])


class TestThePromptStatesTheProtocol:
    def _prompt(self):
        from mllm.message_builder import build_forensic_prompt

        return build_forensic_prompt(["probe", "noise"])

    def test_one_action_per_response_is_stated(self):
        prompt = self._prompt()
        assert "ONE ACTION PER RESPONSE" in prompt
        assert "never two calls" in prompt

    def test_the_bbox_is_optional_and_invention_is_forbidden(self):
        prompt = self._prompt()
        assert "bbox is OPTIONAL" in prompt
        assert "do NOT invent a suspicious region" in prompt

    def test_a_quiet_first_turn_is_allowed(self):
        assert "未观察到明确异常" in self._prompt()

    def test_physical_claims_need_a_measurement_first(self):
        prompt = self._prompt()
        assert "SRM residuals" in prompt and "cannot see" in prompt


class TestAReasoningTagIsNotARequest:
    """The second real shape: the model *refers* to a tool inside <reasoning>."""

    REAL_TURN = ("<reasoning>\n综合分析两个专家的反馈：\n"
                 "1. <call_probe> 的结果表明图像高度可能是 AI 生成的。\n"
                 "2. <call_noise> 的结果表明图像中的微噪声水平较低。\n</reasoning>\n"
                 '<verdict>{"verdict": "Uncertain", "confidence": 0.4}</verdict>')

    def test_it_is_a_conclusion_and_not_two_calls(self):
        action = Parser.protocol_action(self.REAL_TURN)
        assert action["violation"] is None
        assert action["calls"] == [] and action["verdict"] is True

    def test_executable_calls_ignores_prose_references(self):
        assert Parser.executable_calls(self.REAL_TURN) == []

    def test_a_real_request_outside_the_blocks_is_still_seen(self):
        text = self.REAL_TURN + "\n<call_jpeg></call_jpeg>"
        assert Parser.executable_calls(text) == [("jpeg", None)]
