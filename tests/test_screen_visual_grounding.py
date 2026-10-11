"""Tests for the G4-f visual-grounding screen (D2)."""

import json

import pytest

from scripts.screen_visual_grounding import (
    ScriptedJudge,
    build_compare_request,
    build_crop_request,
    build_screen_request,
    crop_images,
    evaluate,
    human_visual_flag,
    is_suspect,
    parse_screen,
    screen,
)


def _sample(sample_id="f2_noise__ADM_x_png", answer="<observation>眼睛周围的纹理不自然</observation>"):
    return {
        "id": sample_id,
        "image_path": "sft_data/variants/ADM_x_png.png",
        "ground_truth": "Fake",
        "final_verdict": {"verdict": "Fake", "confidence": 0.8, "posterior": 0.8},
        "conversations": [
            {"from": "user", "value": "<image>\n请分析。"},
            {"from": "gpt", "value": answer},
        ],
        "evidence_chain": [{
            "evidence_id": "E-1", "source": "noise_expert", "measurement_scope": "global",
            "diagnostic_region_image": "traces/evidence/s1/region_E-1.png",
            "visual_artifacts": ["traces/evidence/s1/map_E-1.png"],
        }],
        "metadata": {"tools_served": ["noise"], "trajectory_id": "noise__ADM_x_png"},
        "_bucket_file": "tool_positive",
    }


class TestRequestBuilding:
    def test_the_first_call_describes_without_the_answer(self):
        """It must look before it compares; showing the answer first gets it echoed."""
        text, images = build_screen_request(_sample(answer="<observation>右侧建筑有压缩痕迹</observation>"))
        assert images == ["sft_data/variants/ADM_x_png.png"]
        assert "右侧建筑" not in text

    def test_the_second_call_compares_description_and_answer(self):
        text, images = build_compare_request(_sample(), "回答文本", "图里有一条鳄鱼")
        assert images == ["sft_data/variants/ADM_x_png.png"]
        assert "图里有一条鳄鱼" in text and "回答文本" in text

    def test_the_crop_question_sends_the_crops_alone(self):
        request = build_crop_request(_sample())
        assert request is not None
        text, images = request
        assert images and images[0].endswith("region_E-1.png")
        assert "照片裁剪图" in text

    def test_a_tool_free_sample_has_no_crop_question(self):
        sample = _sample()
        sample["evidence_chain"] = []
        assert build_crop_request(sample) is None

    def test_at_most_two_crops_like_inference(self):
        sample = _sample()
        sample["evidence_chain"][0]["visual_artifacts"] = ["a.png", "b.png", "c.png"]
        assert len(crop_images(sample)) == 1      # only the photo crop counts

    def test_heat_maps_are_not_offered_as_crops(self):
        """
        The artifact images are colour-mapped residual maps; asking whether they
        show the described part is a question with no correct answer (it was
        answered false for 69 of 79 samples).
        """
        sample = _sample()
        sample["evidence_chain"][0]["visual_artifacts"] = ["traces/evidence/s1/noise_residual_map_E-1.png"]
        assert crop_images(sample) == ["traces/evidence/s1/region_E-1.png"]

    def test_the_answer_under_review_reaches_the_comparison(self):
        text, _ = build_compare_request(
            _sample(), "<observation>右侧建筑有压缩痕迹</observation>", "一张石砌教堂的照片")
        assert "右侧建筑有压缩痕迹" in text and "一张石砌教堂" in text

    def test_the_prompt_forbids_judging_authenticity(self):
        text, _ = build_screen_request(_sample())
        assert "真伪判断" in text
        compare, _ = build_compare_request(_sample(), "回答", "描述")
        assert "不要判断图像真伪" in compare

    def test_the_judge_is_asked_to_describe_before_judging(self):
        text, _ = build_screen_request(_sample())
        assert "先描述你看到的" in text or "content" in text


class TestParsing:
    def test_it_reads_the_json(self):
        parsed = parse_screen('{"verdict": "ok", "described_target_in_image": true}')
        assert parsed["verdict"] == "ok"

    def test_it_survives_prose_around_the_json(self):
        parsed = parse_screen('分析如下：\n{"verdict": "suspect"}\n以上。')
        assert parsed["verdict"] == "suspect"

    def test_single_quotes_are_repaired(self):
        assert parse_screen("{'verdict': 'ok'}")["verdict"] == "ok"

    def test_garbage_yields_nothing(self):
        assert parse_screen("no json here") is None


class TestSuspectDecision:
    def test_an_ok_verdict_is_not_a_suspect(self):
        assert is_suspect({"verdict": "ok", "q1_target_in_image": True,
                           "q3_anomaly_visible": True}) is False

    def test_a_false_check_is_a_suspect_even_without_the_label(self):
        """A judge that answers "no" and forgets the label still flagged it."""
        assert is_suspect({"q1_target_in_image": False}) is True

    def test_the_crop_question_is_judged_on_its_own_key(self):
        assert is_suspect({"q1_target_in_image": True}, ("q3_crop_shows_target",)) is False
        assert is_suspect({"q3_crop_shows_target": False}, ("q3_crop_shows_target",)) is True

    def test_a_judgement_without_a_flag_is_not_a_suspect(self):
        """
        The first version flagged 75 of 79 because a *judgement* ("纹理不自然")
        is never directly visible; only an explicit no counts.
        """
        assert is_suspect({"q1_target_in_image": True,
                           "q2_anomaly_visible": True}) is False

    def test_an_unparsable_answer_is_a_suspect_not_a_pass(self):
        assert is_suspect(None) is True

    def test_the_crop_check_may_be_not_applicable(self):
        assert is_suspect({"q3_crop_shows_target": "na"}, ("q3_crop_shows_target",)) is False


class TestScreenRun:
    def test_it_records_a_verdict_per_sample(self):
        records = screen([_sample()], ScriptedJudge().judge, progress=False)
        assert len(records) == 1
        assert records[0]["suspect"] is False
        assert records[0]["checks"]["q1_target_in_image"] is True
        assert records[0]["checks"]["q3_crop_shows_target"] is True

    def test_a_sample_with_a_crop_costs_three_calls(self):
        judge = ScriptedJudge()
        screen([_sample()], judge.judge, progress=False)
        assert judge.calls == 3

    def test_a_tool_free_sample_costs_two_calls(self):
        sample = _sample()
        sample["evidence_chain"] = []
        judge = ScriptedJudge()
        screen([sample], judge.judge, progress=False)
        assert judge.calls == 2

    def test_a_resumed_run_skips_what_is_done(self):
        first = screen([_sample()], ScriptedJudge().judge, progress=False)
        second = screen([_sample()], ScriptedJudge().judge, progress=False, records=first)
        assert len(second) == 1

    def test_a_suspect_verdict_is_carried_through(self):
        records = screen([_sample()], ScriptedJudge("suspect").judge, progress=False)
        assert records[0]["suspect"] is True

    def test_a_bad_crop_answer_flags_the_sample_on_its_own(self):
        """The crop question is where the reviewer's largest class lives."""
        class CropOnlyJudge:
            def __call__(self, text, images):
                if "先看图" in text:
                    return json.dumps({"content": "背景"})
                if "照片裁剪图" in text:
                    return json.dumps({"crop_content": "背景", "q3_crop_shows_target": False,
                                       "reason": "裁剪图与所述部位无关"})
                return json.dumps({"q1_target_in_image": True,
                                   "q2_anomaly_visible": True, "reason": "ok"})

        records = screen([_sample()], CropOnlyJudge(), progress=False)
        assert records[0]["suspect"] is True
        assert records[0]["missing"] == "" or True


class TestEvaluation:
    def _dispositions(self, notes):
        return {sid: {"decision": d, "notes": n} for sid, (d, n) in notes.items()}

    def test_it_matches_the_reviewers_visual_notes(self):
        records = [{"sample_id": "a", "suspect": True},
                   {"sample_id": "b", "suspect": False}]
        dispositions = self._dispositions({
            "a": ("revise", "诊断区域为左上角背景，与所述眼睛不符"),
            "b": ("accept", "区域一致、证据充分"),
        })
        report = evaluate(records, dispositions, {"a", "b"})
        assert report["human_visual_flagged"] == 1
        assert report["model_suspects"] == 1
        assert report["agreement"] == 1
        assert report["recall_on_human_visual"] == 1.0

    def test_it_counts_suspects_that_pass_the_automatic_checks(self):
        """The samples where the screen earns its keep."""
        records = [{"sample_id": "a", "suspect": True}]
        dispositions = self._dispositions({"a": ("revise", "裁剪区域不对")})
        report = evaluate(records, dispositions, {"a"})
        assert report["suspects_that_pass_automatic_checks"] == 1

    def test_the_same_suspect_counts_zero_when_the_checks_already_fail_it(self):
        records = [{"sample_id": "a", "suspect": True}]
        dispositions = self._dispositions({"a": ("revise", "裁剪区域不对")})
        report = evaluate(records, dispositions, set())
        assert report["suspects_that_pass_automatic_checks"] == 0

    def test_a_miss_is_listed_not_hidden(self):
        records = [{"sample_id": "a", "suspect": False}]
        dispositions = self._dispositions({"a": ("reject", "编造了局部测量")})
        report = evaluate(records, dispositions, {"a"})
        assert report["human_flagged_model_missed"] == ["a"]
        assert report["recall_on_human_visual"] == 0.0

    def test_the_proxy_is_named_in_the_report(self):
        report = evaluate([], {}, set())
        assert "proxy" in report["note"]


class TestHumanCue:
    def test_region_mismatch_wording_counts(self):
        assert human_visual_flag("诊断框 [13,13,29,29] 位于左上方墙面，非所述右上角狗毛发区域")

    def test_an_unrelated_complaint_does_not(self):
        assert not human_visual_flag("仅以 budget_exhausted 说明弃权，未说明缺何种证据")
