"""
Tests for the raw-trace audit (G7 §4.17 A1).

The lesson this file exists for: a keyword counter reported 35 sessions of
"ghost experts" and a second pass reported 45 sentences — neither number is a
count of defects until a criterion is stated and the cases are read.  What the
tests pin is the criterion's *boundary*: it must fire on a claim of a result
and stay silent on advice, absence and hypotheses.
"""

import json

import pytest

from scripts.g7_audit_traces import (
    _mentions_raw,
    coordinate_out_of_range,
    fabricated_expert,
    label_report_conflict,
    raw_score_as_direction,
    skipped_request_as_evidence,
)


def _trace(*assistant_turns):
    return {"conversations": [{"from": "user", "value": "x"}]
            + [{"from": "gpt", "value": turn} for turn in assistant_turns]}


def _record(*tokens, **overrides):
    record = {"evidence": list(tokens), "final_verdict": "Real",
              "model_candidate": "Real", "policy": "probe+noise"}
    record.update(overrides)
    return record


def _token(source="probe_dino_expert", raw=0.9, p_fake=1.0):
    return {"source": source, "raw_metric": raw,
            "calibrated_likelihood": {"Real": round(1 - p_fake, 3), "Fake": p_fake},
            "strength": raw}


class TestFabricatedExpert:
    def test_a_claimed_result_for_an_unrun_expert_is_flagged(self):
        trace = _trace("微噪声残留专家也显示了较高水平的微噪声，支持这一结论。")
        findings = fabricated_expert(trace, _record(_token()))
        assert findings and findings[0]["class"] == "fabricated_expert_result"
        assert findings[0]["status"] == "candidate"

    def test_a_result_for_an_expert_that_ran_is_not_flagged(self):
        trace = _trace("微噪声残留专家显示了较高水平的微噪声。")
        tokens = [_token(), _token("noise_expert", 4.2, 0.11)]
        assert fabricated_expert(trace, _record(*tokens)) == []

    def test_advice_to_call_is_not_a_claim(self):
        trace = _trace("建议调用噪声残留专家以进一步验证。")
        assert fabricated_expert(trace, _record(_token())) == []

    def test_saying_it_was_not_called_is_not_a_claim(self):
        trace = _trace("未调用噪声残留专家，无法进一步降低不确定性。")
        assert fabricated_expert(trace, _record(_token())) == []

    def test_a_hypothesis_is_not_a_claim(self):
        trace = _trace("若调用噪声残留专家，可能会得到较低的水平。")
        assert fabricated_expert(trace, _record(_token())) == []


class TestRawScoreAsDirection:
    def _weak(self, raw=0.3431):
        return _token(raw=raw, p_fake=0.52)

    def test_a_raw_score_carrying_direction_in_the_mid_band_is_flagged(self):
        trace = _trace("AI生成概率为0.3431，表明图像更可能是由真实相机拍摄。")
        assert raw_score_as_direction(trace, _record(self._weak()))

    def test_the_correct_reading_is_not_flagged(self):
        """Naming the near-chance calibration and withholding direction."""
        trace = _trace("得分为0.3431，表明图像更可能是真实拍摄，但其得分接近于随机"
                       "（P(Fake)=0.52），因此不足以确定图像的真实性。")
        assert raw_score_as_direction(trace, _record(self._weak())) == []

    def test_a_clear_measurement_is_out_of_scope(self):
        trace = _trace("AI生成概率为0.99，表明图像更可能是AI生成的。")
        assert raw_score_as_direction(trace, _record(_token(raw=0.99, p_fake=1.0))) == []

    def test_the_value_must_actually_appear(self):
        trace = _trace("探针专家表明图像更可能是AI生成的。")
        assert raw_score_as_direction(trace, _record(self._weak())) == []

    def test_值匹配容忍四舍五入(self):
        assert _mentions_raw("得分 0.3431 表明", 0.343105)
        assert _mentions_raw("得分 0.343 表明", 0.343105)
        assert not _mentions_raw("得分 0.9999 表明", 0.343105)


class TestCoordinateRange:
    def test_a_pixel_coordinate_in_the_normalized_space_is_confirmed(self):
        trace = _trace("<call_probe>[200, 130, 780, 1036]</call_probe>")
        findings = coordinate_out_of_range(trace, _record())
        assert findings and findings[0]["status"] == "confirmed"

    def test_in_range_calls_pass(self):
        trace = _trace("<call_probe>[0, 0, 1000, 1000]</call_probe>")
        assert coordinate_out_of_range(trace, _record()) == []


class TestLabelConflict:
    def test_a_silent_override_is_recorded(self):
        record = _record(final_verdict="Fake", model_candidate="Uncertain")
        findings = label_report_conflict(_trace("x"), record)
        assert findings[0]["class"] == "label_overrode_model"

    def test_agreement_is_not_a_finding(self):
        assert label_report_conflict(_trace("x"),
                                     _record(final_verdict="Real", model_candidate="Real")) == []


class TestSkippedRepeat:
    def test_a_skip_later_spoken_of_as_new_measurement_is_flagged(self):
        trace = {"conversations": [
            {"from": "user", "value": "[System: ... 已跳过执行 ...]"},
            {"from": "gpt", "value": "噪声残留专家再次测量显示微噪声水平较低。"}]}
        assert skipped_request_as_evidence(trace, _record())

    def test_without_a_skip_event_nothing_is_flagged(self):
        trace = _trace("噪声残留专家再次测量显示微噪声水平较低。")
        assert skipped_request_as_evidence(trace, _record()) == []


class TestTheAdjudicationFile:
    def test_the_adjudication_covers_every_candidate(self):
        """每个机器候选都要有裁决，否则数字仍不可追溯。"""
        from config import PROJECT_ROOT
        import os

        audit_path = os.path.join(PROJECT_ROOT, "calibration",
                                  "g7_audit_raw_traces.json")
        adj_path = os.path.join(PROJECT_ROOT, "calibration",
                                "g7_audit_raw_traces_adjudication.json")
        if not (os.path.exists(audit_path) and os.path.exists(adj_path)):
            pytest.skip("审计尚未运行")
        audit = json.load(open(audit_path, encoding="utf-8"))
        adj = json.load(open(adj_path, encoding="utf-8"))

        candidates = {(tid, f["sentence"]) for tid, fs in audit["findings"].items()
                      for f in fs}
        adjudicated = {(x["trajectory_id"], x["sentence"]) for entries in
                       adj["adjudication"].values()
                       for status in entries.values() for x in status}
        assert candidates == adjudicated, candidates ^ adjudicated

    def test_the_summary_matches_the_entries(self):
        from config import PROJECT_ROOT
        import os

        path = os.path.join(PROJECT_ROOT, "calibration",
                            "g7_audit_raw_traces_adjudication.json")
        if not os.path.exists(path):
            pytest.skip("裁决尚未生成")
        adj = json.load(open(path, encoding="utf-8"))
        for cls, entries in adj["adjudication"].items():
            for status, rows in entries.items():
                assert adj["summary"][cls][status] == len(rows), (cls, status)


class TestAbstentionDiagnosis:
    """G7 §4.17 A3: the four moments stay apart, and reasons are candidates."""

    def test_the_verdicts_of_every_turn_are_collected_in_order(self):
        from scripts.g7_abstention_diagnosis import parse_verdicts

        trace = {"conversations": [
            {"from": "gpt", "value": '<verdict>{"verdict": "Uncertain"}</verdict>'},
            {"from": "user", "value": "x"},
            {"from": "gpt", "value": '<reasoning>…</reasoning>\n'
                                    '<verdict>{"verdict": "Fake"}</verdict>'}]}
        verdicts, failures = parse_verdicts(trace)
        assert verdicts == ["Uncertain", "Fake"] and failures == 0

    def test_an_unparseable_verdict_is_counted_not_swallowed(self):
        from scripts.g7_abstention_diagnosis import parse_verdicts

        trace = {"conversations": [{"from": "gpt", "value": "<verdict>{broken</verdict>"}]}
        verdicts, failures = parse_verdicts(trace)
        assert verdicts == [] and failures == 1

    def test_a_near_neutral_session_is_attributed_to_the_calibration(self):
        from scripts.g7_abstention_diagnosis import classify

        trace = _trace("该分数接近随机，不足以支撑判定。")
        findings = classify(trace, _record(_token(raw=0.5, p_fake=0.52)))
        assert "near_chance_calibration" in findings

    def test_demanding_another_expert_with_a_decisive_signal_is_suspicious(self):
        from scripts.g7_abstention_diagnosis import classify

        trace = _trace("由于没有其他专家可以提供额外验证，我们无法进一步降低不确定性。")
        findings = classify(trace, _record(_token(raw=0.98, p_fake=1.0)))
        assert "demands_second_expert" in findings

    def test_a_colour_mismatch_is_not_an_evidentiary_conflict(self):
        from scripts.g7_abstention_diagnosis import classify

        trace = _trace("眼睛周围的毛发颜色不一致，可能存在合成痕迹。")
        findings = classify(trace, _record(_token(raw=0.9, p_fake=0.95)))
        assert "real_conflict_or_limit" not in findings

    def test_denying_a_conflict_is_not_a_conflict(self):
        from scripts.g7_abstention_diagnosis import classify

        trace = _trace("两个专家的测量在各自领域内都合理，且没有直接冲突。")
        findings = classify(trace, _record(_token(raw=0.9, p_fake=0.95)))
        assert "real_conflict_or_limit" not in findings

    def test_a_real_applicability_limit_is_recognised(self):
        from scripts.g7_abstention_diagnosis import classify

        trace = _trace("单独的 JPEG 压缩痕迹不能确定真实性，重压缩后该证据近乎无效，"
                       "不得作为独立判据。")
        findings = classify(trace, _record(_token("jpeg_expert", 2.2, 0.449)))
        assert "real_conflict_or_limit" in findings

    def test_global_task_refused_for_want_of_a_region(self):
        from scripts.g7_abstention_diagnosis import classify

        trace = _trace("虽然证据指向真实，但测量覆盖整个图像，没有提供具体的可疑区域。")
        findings = classify(trace, _record(_token(raw=0.1, p_fake=0.02)))
        assert "global_local_confusion" in findings

    def test_the_rationale_quote_is_not_a_verdict_fragment(self):
        """The fallback used to quote `</verdict>`, which tells a reader nothing."""
        from scripts.g7_abstention_diagnosis import rationale_sentence

        trace = _trace('<reasoning>证据不足，无法确定方向。</reasoning>\n'
                       '<verdict>{"verdict": "Uncertain"}</verdict>')
        sentence = rationale_sentence(trace)
        assert "无法确定" in sentence
        assert "</verdict>" not in sentence
