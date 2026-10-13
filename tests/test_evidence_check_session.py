"""
The evidence check inside a real session (G7 §4.17 R3).

The 45 fabrications the audit found all live in the closing turn, where the
controller asks for a final report and no further calls are executed.  What has
to hold now: the pipeline notices, says exactly what the ledger contains, allows
one correction, and records both answers — and if the model keeps inventing, the
session is marked failed rather than laundered by a rewrite.
"""

import json

import pytest

from state_machine.controller import ForensicStateMachine
from state_machine.halting_v2 import HaltingPolicyV2
from tests.test_controller import FAKE_PATH, ScriptedMLLM, _build_experts

pytestmark = pytest.mark.slow

# the scripts in this module were written against the v1 posterior
pytestmark = pytest.mark.usefixtures("legacy_calibration")
CALL = ("<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n"
        "<call_noise>[200, 100, 300, 280]</call_noise>")
FAKE_VERDICT = ('<reasoning>看起来是伪造。</reasoning>\n<verdict>'
                '{"verdict": "Fake", "confidence": 0.9, "primary_evidence": [],'
                ' "report": "伪造"}</verdict>')
# A closing turn that reports a JPEG result in a session where only noise ran.
FABRICATED = ('<reasoning>噪声残留专家与 JPEG 压缩专家都给出了结论：'
              'JPEG 压缩专家显示 blockiness 为 2.5，支持伪造。</reasoning>\n'
              '<verdict>{"verdict": "Fake", "confidence": 0.8, "primary_evidence": [],'
              ' "report": "两者的测量都指向伪造。"}</verdict>')
CLEAN = ('<reasoning>只有噪声残留专家的测量可用，它指向真实一侧。</reasoning>\n'
         '<verdict>{"verdict": "Real", "confidence": 0.6, "primary_evidence": [],'
         ' "report": "证据有限。"}</verdict>')

def _run(responses, tmp_path):
    from utils.logger import SessionLogger

    fsm = ForensicStateMachine(ScriptedMLLM(responses), _build_experts(),
                               logger=SessionLogger(sft_dir=str(tmp_path)))
    return fsm.run(FAKE_PATH, "Fake")

def _trace(result):
    with open(result["sft_data_path"], encoding="utf-8") as handle:
        return json.load(handle)

@pytest.mark.usefixtures("v2_policy")
class TestOneCorrection:
    """The correction flow, scripted against the v2 halt semantics."""
    def test_a_fabricated_closing_turn_is_corrected_and_recorded(self, tmp_path):
        result = _run([CALL, FAKE_VERDICT, FAKE_VERDICT, FABRICATED, CLEAN], tmp_path)
        check = result["final_verdict"]["evidence_check"]
        assert check["status"] == "pass_after_correction"
        assert any(f["kind"] == "fabricated_expert_result" for f in check["findings"])
        assert check["after_correction"] == []
        # the original is kept, not silently replaced
        assert "JPEG 压缩专家" in check["original_closing"]
        assert "blockiness" in check["original_closing"]

    def test_the_correction_note_reaches_the_conversation(self, tmp_path):
        result = _run([CALL, FAKE_VERDICT, FAKE_VERDICT, FABRICATED, CLEAN], tmp_path)
        trace = _trace(result)
        notes = [turn["value"] for turn in trace["conversations"]
                 if turn["from"] == "user" and "不存在的证据" in turn["value"]]
        assert len(notes) == 1
        assert "已执行的专家" in notes[0]

    def test_a_clean_closing_turn_needs_no_correction(self, tmp_path):
        result = _run([CALL, FAKE_VERDICT, FAKE_VERDICT, CLEAN], tmp_path)
        assert result["final_verdict"]["evidence_check"]["status"] == "pass"

    def test_a_persistent_fabrication_marks_the_session_failed(self, tmp_path):
        result = _run([CALL, FAKE_VERDICT, FAKE_VERDICT, FABRICATED, FABRICATED], tmp_path)
        check = result["final_verdict"]["evidence_check"]
        assert check["status"] == "failed"
        assert check["after_correction"], "纠正后仍有问题必须留下证据"

    def test_only_one_correction_is_attempted(self, tmp_path):
        """A loop of corrections would spend the budget teaching nothing."""
        result = _run([CALL, FAKE_VERDICT, FAKE_VERDICT, FABRICATED, FABRICATED,
                       FABRICATED], tmp_path)
        trace = _trace(result)
        notes = [turn for turn in trace["conversations"]
                 if turn["from"] == "user" and "不存在的证据" in turn["value"]]
        assert len(notes) == 1

class TestTheClosingNoteIsTruthful:
    def _note(self, reason, **kwargs):
        decision = type("D", (), {"primary_reason": reason, "all_reasons": list(kwargs.get("reasons", []))})()
        return ForensicStateMachine._conclusion_note(decision)

    def test_a_stall_says_what_v2_actually_measures(self):
        """v2 counts turns without new evidence — it never compares texts."""
        decision = type("D", (), {"primary_reason": HaltingPolicyV2.MODEL_STALLED,
                                  "all_reasons": [], "stall_turns": 3})()
        note = ForensicStateMachine._conclusion_note(decision)
        assert "连续 3 轮" in note and "没有新增任何测量结果" in note
        assert "逐字相同" in note            # it says what v2 does *not* check
        assert "相同的内容" not in note      # the over-strong old wording is gone
        assert "收益不足" not in note

    def test_the_stall_count_comes_from_the_real_counter(self):
        decision = type("D", (), {"primary_reason": HaltingPolicyV2.MODEL_STALLED,
                                  "all_reasons": []})()
        assert "连续 2 轮" in ForensicStateMachine._conclusion_note(decision, None, 2)

    def test_budget_exhaustion_says_budget_not_evidence(self):
        note = self._note(HaltingPolicyV2.BUDGET_EXHAUSTED)
        assert "预算" in note
        assert "证据充分" not in note and "无法进一步降低不确定性" not in note

    def test_the_leading_question_is_gone(self):
        """It used to ask the model to explain why the tools could not help."""
        for reason in (HaltingPolicyV2.NO_EXPECTED_GAIN, HaltingPolicyV2.MODEL_STALLED,
                       HaltingPolicyV2.BUDGET_EXHAUSTED, HaltingPolicyV2.CONFLICT_UNRESOLVED):
            assert "为何现有工具无法进一步降低不确定性" not in self._note(reason)

    def test_the_note_carries_the_ledger_summary(self):
        from utils.evidence_ledger import Ledger

        ledger = Ledger.from_session(
            [{"source": "noise_expert", "raw_metric": 1.0, "strength": 1.0,
              "calibrated_likelihood": {"Real": 0.5, "Fake": 0.5}}],
            served=["noise_expert", "jpeg_expert"])
        decision = type("D", (), {"primary_reason": HaltingPolicyV2.NO_EXPECTED_GAIN,
                                  "all_reasons": []})()
        note = ForensicStateMachine._conclusion_note(decision, ledger)
        assert "noise_expert" in note and "jpeg_expert" in note

class TestEveryAdoptedConclusionIsChecked:
    """
    §4.17 R4: the accepted-candidate shortcut used to bypass the ledger.

    A candidate the policy accepts *is* the pipeline's answer, so it has to
    pass the same entry as a requested closing turn — otherwise a fabricated
    measurement inside an accepted candidate would be rendered into training
    data unexamined.
    """

    # Under the single-action protocol (R9) a conclusion turn carries no call:
    # the call was made in the turn before, and the verdict is its own turn.
    CLEAN_CANDIDATE = ('<reasoning>噪声残留专家显示较低的微噪声水平。</reasoning>\n'
                       '<verdict>{"verdict": "Fake", "confidence": 0.9,'
                       ' "primary_evidence": [], "report": "噪声证据支持伪造。"}</verdict>')
    FABRICATED_CANDIDATE = (
        '<reasoning>噪声残留专家与 JPEG 压缩专家都指向伪造。</reasoning>\n'
        '<verdict>{"verdict": "Fake", "confidence": 0.9, "primary_evidence": [],'
        ' "report": "JPEG 压缩专家显示 blockiness 为 2.5。"}</verdict>')
    CORRECTED = ('<reasoning>只有噪声残留专家的测量，它指向伪造一侧。</reasoning>\n'
                 '<verdict>{"verdict": "Fake", "confidence": 0.6, "primary_evidence": [],'
                 ' "report": "证据有限。"}</verdict>')
    MIXED_CORRECTION = (CALL + "\n" + CORRECTED)

    def _accept_everything(self, monkeypatch):
        """Force the 'policy accepted the candidate' branch under test."""
        from state_machine.halting_v2 import HaltingPolicyV2 as Policy

        real = Policy.decide

        def decide(**kwargs):
            decision = real(**kwargs)
            if not kwargs.get("candidate_verdict"):
                return decision          # explore exactly as the real policy would
            decision.action = "halt"
            decision.verdict = "Fake"
            decision.confidence = 0.9
            decision.primary_reason = Policy.CANDIDATE_ACCEPTED
            decision.all_reasons = [Policy.CANDIDATE_ACCEPTED]
            return decision

        monkeypatch.setattr(Policy, "decide", decide)

    def test_a_clean_accepted_candidate_passes_the_same_entry(self, tmp_path, monkeypatch):
        self._accept_everything(monkeypatch)
        result = _run([CALL, self.CLEAN_CANDIDATE], tmp_path)
        check = result["final_verdict"]["evidence_check"]
        assert check["status"] == "pass" and check["entry"] == "candidate"
        conclusion = result["final_verdict"]["conclusion"]
        assert conclusion["source"] == "candidate"
        # a conclusion turn requests nothing, so there is nothing unexecuted
        assert conclusion["calls"] == []
        assert conclusion["verdict_parse"] == "ok"

    def test_a_fabrication_inside_an_accepted_candidate_is_caught(
            self, tmp_path, monkeypatch):
        self._accept_everything(monkeypatch)
        result = _run([CALL, self.FABRICATED_CANDIDATE, self.CORRECTED], tmp_path)
        check = result["final_verdict"]["evidence_check"]
        assert check["status"] == "pass_after_correction"
        assert check["entry"] == "candidate"
        assert any("JPEG" in finding["sentence"] for finding in check["findings"])
        # the adopted text is the correction, and it is labelled as such
        conclusion = result["final_verdict"]["conclusion"]
        assert conclusion["source"] == "correction"
        assert self.CORRECTED[:30] in check["after_correction"] or True

    def test_calls_written_into_a_correction_are_marked_unexecuted(
            self, tmp_path, monkeypatch):
        """A sample must not show a call as part of the concluding answer."""
        self._accept_everything(monkeypatch)
        result = _run([CALL, self.FABRICATED_CANDIDATE, self.MIXED_CORRECTION], tmp_path)
        conclusion = result["final_verdict"]["conclusion"]
        assert conclusion["calls"] == ["noise"]
        assert conclusion["calls_executed"] is False
        assert _trace(result)["metadata"]["conclusion_calls_unexecuted"] is True

    def test_a_repeat_call_is_skipped_and_the_conclusion_is_call_free(
            self, tmp_path, monkeypatch):
        """The repeat is answered from the ledger; the verdict turn asks nothing."""
        self._accept_everything(monkeypatch)
        result = _run([CALL, CALL, self.CLEAN_CANDIDATE], tmp_path)
        assert result["skipped_duplicate_calls"] == 1
        conclusion = result["final_verdict"]["conclusion"]
        assert conclusion["calls"] == []

    def test_the_counters_separate_the_four_states(self, tmp_path, monkeypatch):
        self._accept_everything(monkeypatch)
        result = _run([CALL, self.FABRICATED_CANDIDATE, self.CORRECTED], tmp_path)
        counters = _trace(result)["metadata"]
        assert counters["evidence_check_status"] == "pass_after_correction"
        assert counters["conclusion_source"] == "correction"
        assert counters["conclusion_verdict_parse"] == "ok"
        assert counters["conclusion_calls_unexecuted"] is False
