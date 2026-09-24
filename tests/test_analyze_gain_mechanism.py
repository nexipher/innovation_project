"""Tests for the G2-d mechanism analysis (arm reconstruction + mechanism)."""

import json

import pytest

from scripts.analyze_gain_mechanism import (
    _support_versus_calibration,
    classify_arm,
    load_traces,
    mcnemar_vs_baseline,
    per_cell_table,
    visual_channel_leakage,
)


def _turn(value, image_paths=None):
    turn = {"from": "user", "value": value}
    if image_paths:
        turn["image_paths"] = image_paths
    return turn


class TestClassifyArm:
    def test_no_evidence_turn_is_rgb(self):
        conversation = [
            _turn("<image> 请分析这张图像的真实性"),
            {"from": "gpt", "value": "<verdict>{}</verdict>"},
        ]
        assert classify_arm(conversation) == "rgb"

    def test_json_turn_without_images_is_text(self):
        assert classify_arm([_turn('{"evidence_name": "x"}')]) == "text"

    def test_json_turn_with_images_is_both(self):
        assert classify_arm([_turn('{"evidence_name": "x"}', ["a.png"])]) == "both"

    def test_neutral_marker_is_image(self):
        marker = "[诊断证据图已附：noise_expert 对区域 [1, 2, 3, 4] 的分析产物，请结合图像自行判断。]"
        assert classify_arm([_turn(marker, ["a.png"])]) == "image"

    def test_initial_prompt_is_not_evidence(self):
        assert classify_arm([_turn("<image> 提示")]) == "rgb"


class TestLoadTraces:
    def _write(self, directory, name, payload):
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_matches_sample_id_and_arm(self, tmp_path):
        self._write(tmp_path, "forensic_sft_session_20260924_120000_real_0000_png.json", {
            "image_path": "calibration/set/images/real_0000_png.png",
            "conversations": [_turn('{"evidence_name": "x"}')],
        })
        traces = load_traces(str(tmp_path))
        assert ("text", "real_0000_png") in traces

    def test_latest_trace_wins(self, tmp_path):
        base = {
            "image_path": "calibration/set/images/real_0000_png.png",
            "conversations": [_turn('{"evidence_name": "x"}')],
            "final_verdict": {"verdict": "Real"},
        }
        self._write(tmp_path, "forensic_sft_session_20260924_120000_real_0000_png.json",
                    {**base, "final_verdict": {"verdict": "Real"}})
        self._write(tmp_path, "forensic_sft_session_20260924_130000_real_0000_png.json",
                    {**base, "final_verdict": {"verdict": "Fake"}})

        traces = load_traces(str(tmp_path))
        assert traces[("text", "real_0000_png")]["final_verdict"]["verdict"] == "Fake"

    def test_ignores_non_calibration_and_malformed_files(self, tmp_path):
        self._write(tmp_path, "forensic_sft_session_20260924_120000_other.png", {
            "image_path": "dataset/Real/x.jpg",
            "conversations": [_turn('{"evidence_name": "x"}')],
        })
        (tmp_path / "forensic_sft_session_20260924_120001_bad.png.json").write_text(
            "{not json", encoding="utf-8")
        (tmp_path / "unrelated.json").write_text("{}", encoding="utf-8")

        assert load_traces(str(tmp_path)) == {}


def _record(gt, verdict, cell="real_png"):
    return {"sample_id": f"{cell}_{gt}_{verdict}", "gt": gt, "verdict": verdict,
            "confidence": 0.9, "cell": cell, "model_turns": 1, "expert_calls": 0}


class TestPerCellTable:
    def test_reports_hits_per_arm(self):
        records = {
            "rgb": {"a": {**_record("Real", "Real", "real_png"), "sample_id": "a"}},
            "text": {"a": {**_record("Real", "Fake", "real_png"), "sample_id": "a"}},
            "image": {}, "both": {},
        }
        table = per_cell_table(records)
        assert table["real_png"]["rgb"] == "1/1"
        assert table["real_png"]["text"] == "0/1"


class TestMcnemar:
    def _records(self, rgb_verdicts, text_verdicts):
        records = {"rgb": {}, "text": {}, "image": {}, "both": {}}
        for index, (rgb_v, text_v) in enumerate(zip(rgb_verdicts, text_verdicts)):
            truth = "Real" if index % 2 else "Fake"
            records["rgb"][str(index)] = {**_record(truth, rgb_v), "sample_id": str(index)}
            records["text"][str(index)] = {**_record(truth, text_v), "sample_id": str(index)}
        return records

    def test_detects_a_significant_drop(self):
        # rgb (["Fake", "Real", ...]) matches the alternating truth and is
        # right everywhere; text is wrong everywhere → one-sided discordance.
        records = self._records(["Fake", "Real"] * 10, ["Real", "Fake"] * 10)
        entry = mcnemar_vs_baseline(records)["text"]
        assert entry["baseline_only_correct"] == 20
        assert entry["arm_only_correct"] == 0
        assert entry["significant"] is True
        assert isinstance(entry["significant"], bool)

    def test_identical_arms_are_not_significant(self):
        records = self._records(["Fake", "Real"] * 5, ["Fake", "Real"] * 5)
        entry = mcnemar_vs_baseline(records)["text"]
        assert entry["p_value"] == 1.0
        assert entry["significant"] is False


class TestSupportVersusCalibration:
    def test_reports_mean_likelihood_per_claim(self):
        traces = {
            ("text", "a"): {"evidence_chain": [
                {"support": "AI-generated", "calibrated_likelihood": {"Fake": 0.2}},
            ]},
            ("text", "b"): {"evidence_chain": [
                {"support": "Real", "calibrated_likelihood": {"Fake": 0.8}},
                {"support": "Real", "calibrated_likelihood": {"Fake": 0.9}},
            ]},
            ("image", "c"): {"evidence_chain": [
                {"support": "Real", "calibrated_likelihood": {"Fake": 1.0}},
            ]},
        }
        result = _support_versus_calibration(traces, "text")
        assert result["AI-generated"]["mean_calibrated_p_fake"] == pytest.approx(0.2)
        assert result["Real"]["mean_calibrated_p_fake"] == pytest.approx(0.85)
        assert result["Real"]["n"] == 2  # the image-arm token is excluded

    def test_ignores_tokens_without_calibration(self):
        traces = {("both", "a"): {"evidence_chain": [{"support": "Real"}]}}
        assert _support_versus_calibration(traces, "both") == {}


class TestVisualChannelLeakage:
    def test_detects_direction_from_metric_means(self):
        traces = {
            ("image", "a"): {"final_verdict": {"verdict": "Fake"}, "evidence_chain": [
                {"source": "noise_expert", "raw_metric": 2.0}]},
            ("image", "b"): {"final_verdict": {"verdict": "Real"}, "evidence_chain": [
                {"source": "noise_expert", "raw_metric": 1.0}]},
        }
        entry = visual_channel_leakage(traces)["noise_expert"]
        assert entry["direction"] == "fake_calls_have_higher_metric"
        assert entry["mean_metric_fake"] == pytest.approx(2.0)

    def test_uncertain_verdicts_are_excluded(self):
        """A source with no decidable verdict carries no leakage evidence."""
        traces = {
            ("image", "a"): {"final_verdict": {"verdict": "Uncertain"}, "evidence_chain": [
                {"source": "noise_expert", "raw_metric": 2.0}]},
        }
        assert visual_channel_leakage(traces) == {}

    def test_one_sided_verdicts_are_marked_insufficient(self):
        traces = {
            ("image", "a"): {"final_verdict": {"verdict": "Fake"}, "evidence_chain": [
                {"source": "noise_expert", "raw_metric": 2.0}]},
        }
        entry = visual_channel_leakage(traces)["noise_expert"]
        assert entry["direction"] == "insufficient"
        assert entry["n_fake"] == 1
