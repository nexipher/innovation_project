"""
The g8 evaluation entry, exercised on CPU with mock experts.

What the tests pin is the *bookkeeping* the user asked for — every measurement
records identity, preprocessing, raw score with its direction, and status — and
the two rules that are easy to lose later: mocks never enter performance
results, and thresholds are never searched on the test partition.
"""

import json
import os

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core


@pytest.fixture(scope="module")
def mock_report(tmp_path_factory):
    from scripts.g8_evaluate_experts import run

    out = tmp_path_factory.mktemp("g8_eval")
    return run("mock-fixed", "val", 2, ("png", "q70"), "cpu", out_dir=str(out))


class TestThePipelineRunsEndToEnd:
    def test_it_scores_every_image_of_every_source(self, mock_report):
        # 2 sources × 2 formats
        assert mock_report["sources"] == 2
        assert mock_report["images"] == 4
        assert mock_report["failures"] == []

    def test_the_report_and_the_per_image_scores_are_written(self, mock_report):
        out = os.path.join(PROJECT_ROOT, mock_report["out_dir"])
        assert os.path.exists(os.path.join(out, "report.json"))
        lines = open(os.path.join(out, "scores.jsonl"), encoding="utf-8").read().strip().split("\n")
        assert len(lines) == 4
        row = json.loads(lines[0])
        for key in ("variant_id", "source_id", "label", "treatment", "raw_score",
                    "elapsed_ms", "status", "error"):
            assert key in row, key


class TestEveryMeasurementCarriesItsIdentity:
    def test_identity_names_the_weights_and_the_preprocessing(self, mock_report):
        identity = mock_report["identity"]
        assert identity["expert"] == "probe_dino_expert"
        assert "preprocess" in identity and "measurement_scope" in identity

    def test_the_direction_comes_from_the_registry(self, mock_report):
        assert mock_report["semantics"]["direction"] == "higher_is_fake"
        assert mock_report["semantics"]["is_calibrated_probability"] is False

    def test_the_threshold_policy_is_recorded(self, mock_report):
        policy = mock_report["threshold_policy"]
        assert policy["value"] == 0.5
        assert policy["searched_on_val"] is False


class TestMocksStayOutOfTheResults:
    def test_a_mock_run_says_so(self, mock_report):
        assert mock_report["mock"] is True
        assert "不计入任何性能结果" in mock_report["note"]

    def test_a_real_expert_run_is_not_flagged_as_mock(self):
        from scripts.g8_evaluate_experts import EXPERTS

        assert EXPERTS["noise"][1] is False
        assert EXPERTS["npr-a"][1] is False
        assert EXPERTS["univfd"][1] is False


class TestTheThresholdRule:
    def test_searching_a_threshold_on_test_is_refused(self, tmp_path):
        from scripts.g8_evaluate_experts import run

        with pytest.raises(SystemExit) as excinfo:
            run("mock-fixed", "test", 1, ("png",), "cpu",
                out_dir=str(tmp_path / "out"), threshold_search=True)
        assert "val" in str(excinfo.value)

    def test_searching_on_val_is_allowed_and_recorded(self, tmp_path):
        from scripts.g8_evaluate_experts import run

        report = run("mock-random", "val", 4, ("png",), "cpu",
                     out_dir=str(tmp_path / "out2"), threshold_search=True)
        assert report["threshold_policy"]["searched_on_val"] is True
        assert "threshold_search_on_val" in report["metrics"]


class TestFailuresAreRecordedNotDropped:
    def test_a_missing_image_becomes_an_error_row(self, tmp_path):
        from scripts.g8_evaluate_experts import score_all

        class Exploding:
            name = "exploding"

            def score(self, path):
                raise FileNotFoundError(path)

        rows = score_all(Exploding(), [{"variant_id": "v", "source_id": "s",
                                        "label": "Fake", "treatment": "png",
                                        "path": "/nonexistent.png"}])
        assert rows[0]["status"] == "error"
        assert "FileNotFoundError" in rows[0]["error"]
        assert rows[0]["raw_score"] is None

    def test_the_error_count_reaches_the_metrics(self):
        from scripts.g8_evaluate_experts import metrics_for

        rows = [{"variant_id": "v", "source_id": "s", "label": "Fake",
                 "treatment": "png", "raw_score": None, "elapsed_ms": 1.0,
                 "status": "error", "error": "boom"}]
        metrics = metrics_for(rows, "probe_dino_expert", 0.5)
        assert metrics["n"] == 0
        assert metrics["note"] == "没有可用的测量"


class TestTheFormatVocabulary:
    def test_the_short_names_map_to_the_manifest_names(self):
        from scripts.g8_evaluate_experts import _normalise_formats

        assert _normalise_formats(["png", "q70", "q85", "q95"]) == {
            "png", "jpeg_q70", "jpeg_q85", "jpeg_q95"}

    def test_the_manifest_names_pass_through(self):
        from scripts.g8_evaluate_experts import _normalise_formats

        assert _normalise_formats(["jpeg_q70"]) == {"jpeg_q70"}


class TestTheDirectionIsAppliedConsistently:
    def test_a_reversed_expert_still_scores_high_for_fake(self):
        """noise is higher_is_real: its AUROC must be computed on 1−score, or a
        good expert reads as a bad one."""
        from scripts.g8_evaluate_experts import metrics_for

        rows = [{"variant_id": "a", "source_id": "s1", "label": "Fake", "treatment": "png",
                 "raw_score": 0.2, "elapsed_ms": 1.0, "status": "ok", "error": None},
                {"variant_id": "b", "source_id": "s2", "label": "Real", "treatment": "png",
                 "raw_score": 0.8, "elapsed_ms": 1.0, "status": "ok", "error": None}]
        metrics = metrics_for(rows, "noise_expert", 0.5)
        assert metrics["direction"] == "higher_is_real"
        assert metrics["auroc"] == 1.0
