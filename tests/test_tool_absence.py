"""
Tool absence and tool failure (stage 4.4b).

Two things the proposal asks for by name — "工具失败不得变成中性测量" and
"工具缺席要如实呈现" — and both were false before this stage: an unregistered
tag was dropped with a bare `continue` (so "this session has no such tool" and
"the model did not call it" looked identical in the trace), and an expert that
raised would abort the whole session instead of leaving a failed record.
"""

import json
import os

import pytest
from PIL import Image

from config import PROJECT_ROOT
from state_machine.controller import ForensicStateMachine
from state_machine.halting_v2 import HaltingPolicyV2

pytestmark = pytest.mark.core


def _image(tmp_path):
    path = tmp_path / "input.png"
    Image.new("RGB", (64, 64), (90, 120, 60)).save(path)
    return str(path)


class _Scripted:
    """A client that says exactly what the test needs it to say."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def generate(self, image_path, history):
        value = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return value

    def reset(self):
        self._index = 0

    @property
    def name(self):
        return "scripted"

    @property
    def mode(self):
        return "scripted"


class _WorkingExpert:
    source_name = "probe_dino_expert"
    evidence_name = "aigi_discriminator_score"
    counter_explanation = ""

    def analyze(self, patch):
        from experts.base import ExpertResult

        return ExpertResult(source=self.source_name, evidence_name=self.evidence_name,
                            strength=0.9, raw_metric=0.9, support="AI-generated",
                            phenomenon="整图判别分数 0.9。", reasoning="分数偏高。",
                            interpretation_text="指向生成。")


class _BrokenExpert(_WorkingExpert):
    source_name = "npr_expert"
    evidence_name = "npr_upsampling_artefact_score"

    def analyze(self, patch):
        raise RuntimeError("权重文件不存在")


CALL_UNKNOWN = ("<planning>\nSuspected Region: none\n</planning>\n"
                "<call_jpeg></call_jpeg>")
CALL_BROKEN = ("<planning>\nSuspected Region: none\n</planning>\n"
               "<call_npr></call_npr>")
VERDICT = ('<reasoning>证据不足。</reasoning>\n<verdict>'
           '{"verdict": "Uncertain", "confidence": 0.4}</verdict>')


def _run(tmp_path, responses, experts):
    from utils.logger import SessionLogger

    fsm = ForensicStateMachine(_Scripted(responses), experts,
                               logger=SessionLogger(sft_dir=str(tmp_path)))
    return fsm.run(_image(tmp_path), "Fake")


class TestAnAbsentToolIsReportedNotSwallowed:
    def test_the_model_is_told_the_tool_is_not_served(self, tmp_path):
        result = _run(tmp_path, [CALL_UNKNOWN, VERDICT],
                      {"probe_dino_expert": _WorkingExpert()})
        notes = [turn["value"] for turn in result["conversation"]
                 if turn["from"] == "user"]
        assert any("本次会话不提供 jpeg" in note for note in notes), notes
        # 并且告诉它现在能用什么
        assert any("可用工具" in note and "probe" in note for note in notes), notes

    def test_the_counter_records_it(self, tmp_path):
        result = _run(tmp_path, [CALL_UNKNOWN, VERDICT],
                      {"probe_dino_expert": _WorkingExpert()})
        trace = json.load(open(result["sft_data_path"], encoding="utf-8"))
        assert trace["metadata"]["unavailable_tool_calls"] == ["jpeg"]

    def test_the_session_still_ends_legally(self, tmp_path):
        result = _run(tmp_path, [CALL_UNKNOWN, VERDICT],
                      {"probe_dino_expert": _WorkingExpert()})
        assert result["final_verdict"]["verdict"] == "Uncertain"
        assert result["halting_reason"] == "model_concluded"


class TestAFailedToolLeavesAFailedRecord:
    def test_a_raising_expert_does_not_kill_the_session(self, tmp_path):
        result = _run(tmp_path, [CALL_BROKEN, VERDICT],
                      {"npr_expert": _BrokenExpert()})
        assert result["final_verdict"]["verdict"] == "Uncertain"

    def test_the_chain_carries_a_failed_token_with_its_reason(self, tmp_path):
        result = _run(tmp_path, [CALL_BROKEN, VERDICT],
                      {"npr_expert": _BrokenExpert()})
        failed = [token for token in result["evidence_chain"]
                  if token.get("status") == "failed"]
        assert len(failed) == 1
        assert failed[0]["source"] == "npr_expert"
        assert "RuntimeError" in failed[0]["reason"]

    def test_the_model_is_told_it_is_not_a_neutral_measurement(self, tmp_path):
        result = _run(tmp_path, [CALL_BROKEN, VERDICT],
                      {"npr_expert": _BrokenExpert()})
        notes = [turn["value"] for turn in result["conversation"]
                 if turn["from"] == "user"]
        assert any("不是中性测量" in note for note in notes), notes

    def test_the_counter_records_the_failure(self, tmp_path):
        result = _run(tmp_path, [CALL_BROKEN, VERDICT],
                      {"npr_expert": _BrokenExpert()})
        trace = json.load(open(result["sft_data_path"], encoding="utf-8"))
        assert trace["metadata"]["failed_tool_calls"]
        assert "RuntimeError" in trace["metadata"]["failed_tool_calls"][0]


class TestAFailedTokenDeliversNothing:
    def test_any_number_quoted_about_it_is_fabrication(self):
        """失败 token 没有数值：引用它给出的任何数字都是编造。"""
        from utils.evidence_ledger import Ledger

        ledger = Ledger.from_session([{
            "source": "npr_expert", "evidence_id": "E-failed-npr_expert",
            "status": "failed", "reason": "RuntimeError: 权重不存在",
            "measurement_scope": "global", "model_delivered": "slim"}])
        record = ledger.measured["npr_expert"]
        assert record.delivered == {}
        finding = ledger._check_number(0.93, 2, "npr_expert", ["npr_expert"],
                                       strict=False, label=None, system_cue=False,
                                       clause="NPR 给出 0.93")
        assert finding is not None and finding["kind"] == "fabricated_value"
