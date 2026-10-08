"""Tests for the G5 acceptance checker."""

import json

import pytest

from scripts.check_acceptance_g5 import (
    JPEG_CELL_FLOOR,
    UNCERTAIN_CEILING,
    _container_reason_lines,
    cell_auroc,
    check,
    container_offenders,
)

CELLS = ("real_png", "fake_png", "real_jpeg_q70", "fake_jpeg_q70")


def _record(sample_id, cell, gt, verdict, confidence):
    return {"sample_id": sample_id, "cell": cell, "gt": gt,
            "verdict": verdict, "confidence": confidence,
            "model_turns": 2, "expert_calls": 1, "elapsed_s": 1.0}


def _report(arm_records):
    return {"mode": "gpu", "conditions": {
        arm: {"records": rows} for arm, rows in arm_records.items()}}


def _good_arm(errors=0):
    """A tool arm that separates within every format and commits often."""
    rows = []
    index = 0
    for _ in range(12):
        for cell, truth in (("real_png", "Real"), ("fake_png", "Fake"),
                            ("real_jpeg_q70", "Real"), ("fake_jpeg_q70", "Fake")):
            wrong = index < errors
            verdict = truth if not wrong else ("Fake" if truth == "Real" else "Real")
            rows.append(_record(f"{cell}_{index}", cell, truth, verdict, 0.85))
            index += 1
    return rows


def _baseline_arm(confidence=0.9):
    """
    Always Real at one confidence: no ranking signal at all.

    A spread version would be a *harder* baseline than the real one — the
    G3-e baseline ranked at 0.654 precisely because it was less sure on fakes
    — so the fixture uses the degenerate form to keep the comparison honest.
    """
    rows = []
    for index in range(12):
        for cell, truth in (("real_png", "Real"), ("fake_png", "Fake"),
                            ("real_jpeg_q70", "Real"), ("fake_jpeg_q70", "Fake")):
            rows.append(_record(f"{cell}_{index}", cell, truth, "Real", confidence))
    return rows


class TestCellAuroc:
    def test_it_pools_a_formats_two_cells(self):
        rows = {r["sample_id"]: r for r in _good_arm()}
        assert cell_auroc(rows, "png") == pytest.approx(1.0)

    def test_too_few_samples_yields_none(self):
        rows = {r["sample_id"]: r for r in _good_arm()[:4]}
        assert cell_auroc(rows, "png") is None


class TestContainerReason:
    def test_a_reason_drawn_from_the_container_is_flagged(self):
        assert _container_reason_lines("这张图是 PNG 格式，因此判定为 AI 生成。")

    def test_a_denial_is_not(self):
        assert _container_reason_lines("容器格式与真伪无关，不得据此判断。") == []

    def test_a_bare_mention_is_not(self):
        assert _container_reason_lines("图像为 JPEG 容器。") == []

    def test_an_english_reason_is_flagged(self):
        assert _container_reason_lines("The file is PNG, therefore it is fake.")

    def test_offenders_are_counted_from_real_traces_only(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps({
            "metadata": {"mock_mode": "qwen_real"},
            "conversations": [{"from": "gpt", "value": "因为是 PNG，所以是假图。"},
                              {"from": "gpt", "value": "证据不足，输出 Uncertain。"}]}),
            encoding="utf-8")
        (tmp_path / "b.json").write_text(json.dumps({
            "metadata": {"mock_mode": "two_calls"},
            "conversations": [{"from": "gpt", "value": "因为是 PNG，所以是假图。"}]}),
            encoding="utf-8")

        offenders = container_offenders(str(tmp_path))
        assert list(offenders) == ["a.json"]
        assert offenders["a.json"] == 1

    def test_a_missing_directory_is_not_an_error(self):
        assert container_offenders(None) == {}
        assert container_offenders("/nonexistent") == {}


class TestCheck:
    def _passing(self):
        return _report({"rgb": _baseline_arm(), "text": _good_arm(),
                        "image": _good_arm(), "both": _good_arm()})

    def test_a_strong_result_passes_every_criterion(self):
        result = check(self._passing())
        failed = [c["name"] for c in result["criteria"] if not c["passed"]]
        assert result["passed"], failed

    def test_the_untuned_shape_fails(self):
        """Tools that score below the baseline must fail, not squeak through."""
        report = _report({"rgb": _baseline_arm(), "text": _baseline_arm(),
                          "image": _baseline_arm(), "both": _baseline_arm()})
        result = check(report)
        assert not result["passed"]
        names = {c["name"] for c in result["criteria"] if not c["passed"]}
        # The core criterion: tools that do not beat the baseline must fail.
        assert "tool_arms_beat_baseline_auroc" in names
        # Note the baseline's png-cell ranking can still clear the floor on
        # confidence alone — which is exactly what G3-e measured (AUROC 0.61
        # while never committing to Fake), so it is not asserted here.

    def test_jpeg_cells_below_the_floor_fail(self):
        rows = []
        for index in range(12):
            rows.append(_record(f"r{index}", "real_jpeg_q70", "Real", "Fake", 0.9))
            rows.append(_record(f"f{index}", "fake_jpeg_q70", "Fake", "Real", 0.9))
        report = self._passing()
        report["conditions"]["image"]["records"] = rows
        result = check(report)
        criterion = next(c for c in result["criteria"] if c["name"] == "jpeg_cells_above_chance")
        assert not criterion["passed"]

    def test_the_floor_appears_in_the_detail(self):
        report = self._passing()
        result = check(report)
        criterion = next(c for c in result["criteria"] if c["name"] == "jpeg_cells_above_chance")
        assert str(JPEG_CELL_FLOOR) in criterion["detail"]

    def test_abstention_ceiling_is_reported(self):
        report = self._passing()
        for row in report["conditions"]["text"]["records"]:
            row["verdict"], row["confidence"] = "Uncertain", 0.5
        criterion = next(c for c in check(report)["criteria"]
                         if c["name"] == "abstention_falls")
        assert not criterion["passed"]
        assert str(UNCERTAIN_CEILING) in criterion["detail"]

    def test_the_baseline_report_supplies_the_jpeg_reference(self):
        result = check(self._passing(), _report({"rgb": _baseline_arm()}))
        criterion = next(c for c in result["criteria"] if c["name"] == "jpeg_cells_above_chance")
        assert "untuned worst" in criterion["detail"]

    def test_container_offenders_fail_the_run(self, tmp_path):
        (tmp_path / "t.json").write_text(json.dumps({
            "metadata": {"mock_mode": "qwen_real"},
            "conversations": [{"from": "gpt", "value": "因为格式是 PNG，所以是伪造。"}]}),
            encoding="utf-8")
        result = check(self._passing(), traces_dir=str(tmp_path))
        assert not result["passed"]
        criterion = next(c for c in result["criteria"]
                         if c["name"] == "no_container_as_reason")
        assert "1 model turns" in criterion["detail"]
