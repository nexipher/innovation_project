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


class TestTheFrozenCandidateScheme:
    """
    阶段三冻结的方案必须把用户四条方法学指正一起冻结，否则下一轮又会
    把图当独立样本、又会说"从不伤害"、又会拿 test 当选型集。
    """

    @pytest.fixture(scope="class")
    def scheme(self):
        path = os.path.join(PROJECT_ROOT, "sft_data/g8_eval/FROZEN_CANDIDATE_SCHEME.json")
        if not os.path.exists(path):
            pytest.skip("候选方案尚未冻结")
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def test_the_roles_are_the_decided_ones(self, scheme):
        assert "主检测器" in scheme["experts"]["probe"]["role"]
        assert "尚不认定" in scheme["experts"]["univfd"]["role"]
        assert "退出" in scheme["experts"]["noise"]["role"]
        assert "PNG 条件下的补充候选" in scheme["experts"]["npr-a"]["role"]

    def test_the_resampling_unit_is_the_source(self, scheme):
        assert scheme["statistics"]["unit"].startswith("源")
        assert "√4" in scheme["statistics"]["unit"]

    def test_the_test_partition_stays_closed_until_three_things_are_fixed(self, scheme):
        conditions = scheme["partition_rule"]["opening_conditions"]
        assert any("证据接口" in item for item in conditions)
        assert any("调用规则" in item for item in conditions)
        assert any("提示词" in item for item in conditions)
        assert "未打开" in scheme["partition_rule"]["test"]

    def test_the_png_container_caveat_is_recorded(self, scheme):
        conditions = scheme["experts"]["npr-a"]["conditions"]
        assert any("容器" in item and "JPEG 压缩历史" in item for item in conditions)
        assert any("不得解释成强 Real 证据" in item for item in conditions)

    def test_no_claim_says_an_expert_never_harms(self, scheme):
        finding = scheme["experts"]["univfd"]["finding"]
        assert "未证明" in finding and "未排除损害" in finding
        assert "从不伤害" not in json.dumps(scheme, ensure_ascii=False)

    def test_the_combination_scope_limit_is_recorded(self, scheme):
        assert "等权组合" in scheme["experts"]["noise"]["scope_limit"]

    def test_the_cost_comparison_is_device_matched(self, scheme):
        for name in ("probe", "npr-a", "univfd"):
            assert scheme["experts"][name]["cost_gpu_ms"] > 0

