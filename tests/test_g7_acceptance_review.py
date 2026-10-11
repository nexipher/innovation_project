"""
Tests for the acceptance review aid (CPU).

The review sheet is what a person reads instead of 120 raw traces, so the two
things that matter are pinned here: the flags fire on the failure they name,
and they do not fire on the behaviour we want (a verdict that cites a
probability as a probability).
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from scripts.g7_acceptance_review import (
    markdown,
    probe_artifact_sentences,
    review_session,
    session_text,
)


def _trace(turns, tokens=(), metadata=None, verdict=None):
    return {
        "conversations": [{"from": role, "value": value} for role, value in turns],
        "evidence_chain": list(tokens),
        "metadata": metadata or {},
        "final_verdict": verdict or {"verdict": "Fake", "posterior": 0.9},
    }


def _token(source="probe_dino_expert", p_fake=0.97, direction="AI-generated"):
    return {"source": source, "direction": direction,
            "calibrated_likelihood": {"Real": 1 - p_fake, "Fake": p_fake},
            "applicability": "strong", "consistency": "pass"}


class TestExtraction:
    def test_calls_are_read_from_the_assistant_turns_only(self):
        trace = _trace([
            ("system", "用户 <call_probe>[0,0,1,1]</call_probe>"),
            ("assistant", "…<call_probe>[10, 20, 30, 40]</call_probe>"),
        ])
        _, _, calls = session_text(trace)
        assert calls == ["probe"]

    def test_the_full_text_is_available_for_reading(self):
        trace = _trace([("assistant", "<reasoning>证据一致</reasoning>")])
        assistant, _, _ = session_text(trace)
        assert "证据一致" in assistant


class TestScoreAsArtifact:
    def test_a_score_attributed_to_a_physical_finding_is_flagged(self):
        text = "PROBE 判别分数检测到图像右下的纹理异常。"
        assert probe_artifact_sentences(text) == [text.rstrip("。")]

    def test_the_models_own_disclaimer_is_not_flagged(self):
        """This sentence is the behaviour we asked for, not the failure."""
        text = "该分数仅表示图像作为一个整体的生成性，并没有提供具体的物理区域信息。"
        assert probe_artifact_sentences(text) == []

    def test_talking_about_another_expert_is_not_flagged(self):
        text = "噪声残留专家的分析结果也表明存在噪声差异。"
        assert probe_artifact_sentences(text) == []

    def test_listing_two_experts_evidence_is_not_an_attribution(self):
        text = "AI生成鉴别器的分数和噪声残留水平的测量结果都指向图像可能是真实的。"
        assert probe_artifact_sentences(text) == []

    def test_a_score_cited_as_a_probability_is_not_flagged(self):
        text = "PROBE 判别分数的校准概率为 0.97，指向 AI 生成。"
        assert probe_artifact_sentences(text) == []

    def test_physics_alone_is_not_flagged(self):
        """The flag is about the *attribution*, not about mentioning textures."""
        assert probe_artifact_sentences("画面右下区域的纹理较平滑。") == []

    def test_a_prose_mention_of_the_tag_is_not_a_call(self):
        """The controller's parser wants coordinates; so does this counter."""
        trace = _trace([("assistant", "首先调用 <call_probe> 来评估整体真实性。")])
        _, _, calls = session_text(trace)
        assert calls == []

    def test_a_call_with_coordinates_is_counted(self):
        trace = _trace([("assistant", "<planning>…</planning><call_probe>[0, 0, 100, 100]</call_probe>")])
        _, _, calls = session_text(trace)
        assert calls == ["probe"]


class TestReviewSession:
    def _record(self, **overrides):
        record = {"sample_id": "s1", "label": "Fake", "treatment": "png",
                  "origin": "B", "verdict": "Fake", "posterior": 0.9,
                  "halting_reason": "candidate_matches_posterior",
                  "call_tags": ["probe"], "probe_token_p_fake": 0.97,
                  "sft_data_path": "traces/x.json"}
        record.update(overrides)
        return record

    def test_a_clean_session_has_no_findings(self):
        trace = _trace([("assistant", "<call_probe>[0,0,1,1]</call_probe> PROBE 的校准概率为 0.97。")],
                       [_token()])
        assert review_session(self._record(), trace)["findings"] == []

    def test_an_executed_call_that_never_landed_is_flagged(self):
        """Two calls executed, one token: the identity is what catches a drop."""
        trace = _trace([("assistant", "<call_probe>[0,0,1,1]</call_probe>")],
                       [_token()], metadata={"expert_call_count": 2,
                                             "suppressed_duplicate_count": 0})
        findings = review_session(self._record(), trace)["findings"]
        assert any(f["kind"] == "call_accounting_mismatch" for f in findings)

    def test_a_suppressed_duplicate_keeps_the_identity(self):
        trace = _trace([("assistant", "<call_probe>[0,0,1,1]</call_probe>")],
                       [_token()], metadata={"expert_call_count": 2,
                                             "suppressed_duplicate_count": 1})
        kinds = {f["kind"] for f in review_session(self._record(), trace)["findings"]}
        assert "call_accounting_mismatch" not in kinds

    def test_a_tag_outside_the_toolkit_is_flagged(self):
        trace = _trace([("assistant", "<call_oracle>[0,0,1,1]</call_oracle>")], [_token()])
        findings = review_session(self._record(), trace)["findings"]
        assert any(f["kind"] == "unknown_call_tag" for f in findings)

    def test_a_missing_trace_is_reported_rather_than_hidden(self):
        findings = review_session(self._record(), None)["findings"]
        assert findings[0]["kind"] == "missing_trace"

    def test_a_label_against_its_own_posterior_is_flagged(self):
        trace = _trace([("assistant", "结论")], [_token()],
                       verdict={"verdict": "Real", "posterior": 0.9})
        findings = review_session(self._record(verdict="Real"), trace)["findings"]
        assert any(f["kind"] == "probability_label_mismatch" for f in findings)

    def test_an_auxiliary_token_never_mentioned_is_flagged(self):
        trace = _trace([("assistant", "PROBE 的校准概率为 0.97。")],
                       [_token(), _token(source="noise_expert", p_fake=0.11, direction="Real")])
        findings = review_session(self._record(), trace)["findings"]
        assert any(f["kind"] == "aux_token_unread" and "noise" in f["detail"]
                   for f in findings)

    def test_a_chinese_mention_counts_as_reading_it(self):
        """The expert reported as 频率域专家 was read, whatever the language."""
        trace = _trace([("assistant", "PROBE 的校准概率为 0.97；频率域专家的频谱分析无周期结构。")],
                       [_token(), _token(source="frequency_expert_v2", p_fake=0.3)])
        kinds = {f["kind"] for f in review_session(self._record(), trace)["findings"]}
        assert "aux_token_unread" not in kinds

    def test_stopping_with_positive_utility_left_is_flagged(self):
        trace = _trace([("assistant", "结论")], [_token()],
                       metadata={"halting_reason": "no_expected_gain",
                                 "tool_utilities": {"jpeg_expert": 0.12}},
                       verdict={"verdict": "Uncertain", "posterior": 0.55})
        findings = review_session(self._record(verdict="Uncertain"), trace)["findings"]
        assert any(f["kind"] == "stopped_with_gain_left" for f in findings)

    def test_a_duplicate_is_reported_not_ignored(self):
        trace = _trace([("assistant", "<call_probe>[0,0,1,1]</call_probe><call_probe>[0,0,2,2]</call_probe>")],
                       [_token()], metadata={"suppressed_duplicate_count": 1})
        findings = review_session(self._record(), trace)["findings"]
        assert any(f["kind"] == "duplicate_call" for f in findings)


class TestMarkdown:
    def test_the_sheet_says_the_flags_are_not_verdicts(self):
        report = {"mode": "gpu", "samples": 1, "sources": 1,
                  "summary": {"probe_called_in": 1}}
        sessions = [review_session(
            {"sample_id": "s1", "label": "Fake", "treatment": "png", "origin": "B",
             "verdict": "Fake", "posterior": 0.9, "halting_reason": "x",
             "call_tags": ["probe"], "sft_data_path": None}, None)]
        text = markdown(report, sessions)
        assert "不构成结论" in text
        assert "missing_trace" in text
