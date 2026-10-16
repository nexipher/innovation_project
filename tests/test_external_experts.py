"""
Tests for the G7 external-expert adapters (PROBE-DINOv2, TruFor).

Everything here runs on CPU: the adapters' job is to be a well-behaved
boundary, and the parts that are testable without their weights — availability,
preprocessing arithmetic, statistic summarisation, artifact painting, the CLI
they would run — are exactly the parts that break silently when the model is
finally loaded (docs/development/agent.md §3.2.2).
"""

import os

import numpy as np
import pytest

from experts.external_base import (
    UNCALIBRATED_SUPPORT,
    MockExternalScoreExpert,
    sha256_file,
    tile_positions,
)
from experts.probe_dino import MockProbeExpert, ProbeDinoExpert
from experts.trufor import (
    HIGH_RESPONSE_THRESHOLD,
    MockTruForExpert,
    TruForExpert,
    _entropy,
)


@pytest.fixture(autouse=True)
def _clear_external_env(monkeypatch):
    for name in ("PROBE_WEIGHTS", "TRUFOR_HOME", "TRUFOR_PYTHON", "HF_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)


class TestTiling:
    def test_a_small_image_is_one_tile(self):
        assert tile_positions(256, 256, 336) == [(0, 0, 336, 336)]

    def test_tiles_cover_the_image_without_gaps(self):
        tiles = tile_positions(700, 500, 336)
        assert len(tiles) == 6
        xs = sorted({x for x, _, _, _ in tiles})
        ys = sorted({y for _, y, _, _ in tiles})
        assert xs == [0, 164]                      # last column pulled to the edge
        assert ys == [0, 336, 364]                 # last row likewise

    def test_an_exact_multiple_needs_no_extra_tile(self):
        assert len(tile_positions(672, 672, 336)) == 4


class TestProvenance:
    def test_a_hash_is_computed_and_cached(self, tmp_path):
        weights = tmp_path / "model.pth"
        weights.write_bytes(b"weights" * 100)
        first = sha256_file(str(weights))
        assert len(first) == 64
        assert (tmp_path / "model.pth.sha256").exists()
        assert sha256_file(str(weights)) == first     # served from the sidecar

    def test_missing_weights_are_reported_not_crashed(self):
        expert = ProbeDinoExpert(weights_path="/nonexistent/model.pth")
        assert expert.available() is False
        assert "权重文件不存在" in expert.unavailable_reason()

    def test_no_weights_path_configured(self):
        expert = ProbeDinoExpert()
        assert "未配置权重路径" in expert.unavailable_reason()

    def test_analyse_without_weights_raises_with_the_reason(self):
        expert = ProbeDinoExpert()
        with pytest.raises(RuntimeError, match="未配置权重路径"):
            expert.analyze(np.zeros((32, 32, 3), dtype=np.uint8))


class TestTheUnivfdScoringPathRunsWithoutItsWeights:
    """
    UnivFD 的**打分路径本身**要在没有权重的情况下也能被执行到。

    真机一跑就抓到：`_score` 里 `np` 没导入（`NameError`），也就是说这个专家
    的真实路径**一调用就崩**——在小批里它会被 4.4 的失败包装记成一次 `failed`
    工具调用，会话照跑、数据照收（"看起来跑通"）。这正是本文件开头那句话说的
    那类：没权重也能测的部分，恰恰是加载权重时才会暴露的部分。

    这里用一个假 evaluator 顶替 CLIP：路径真的走一遍，不加载 1.7 GB 主干。
    """

    def _expert(self, tmp_path, monkeypatch):
        from experts import univfd as module

        head = tmp_path / "fc_weights.pth"
        head.write_bytes(b"head" * 64)
        backbone = tmp_path / "ViT-L-14.pt"
        backbone.write_bytes(b"backbone" * 64)

        class _FakeEvaluator:
            def __init__(self, **kwargs):
                pass

            def score(self, image):
                from PIL import Image as _Image
                assert isinstance(image, _Image.Image)
                return 0.77

            def identity(self):
                return {"preprocess_detail": "centre crop（假 evaluator）"}

        monkeypatch.setattr(module, "DEFAULT_HEAD", str(head))
        monkeypatch.setattr(module, "DEFAULT_BACKBONE", str(backbone))
        monkeypatch.setattr(module, "UnivFDEvaluator", _FakeEvaluator)
        return module

    # 注意：`UnivFDExpert()` 的 device 是**自动解析**的（有 CUDA 就是 cuda）。
    # 在无 GPU 的机器上 `unavailable_reason()` 会如实说"官方主干不能在 CPU 上跑"，
    # 于是 analyze 在到达 _score 前就抛错——那样这条测试会变成依赖机器。显式给
    # device="cuda"：这里测的是打分路径本身，不测设备可用性。
    _DEVICE = "cuda"

    def test_analyze_returns_a_score_through_the_real_code_path(self, tmp_path,
                                                                monkeypatch):
        import numpy as np

        module = self._expert(tmp_path, monkeypatch)
        expert = module.UnivFDExpert(device=self._DEVICE)
        result = expert.analyze(np.zeros((240, 240, 3), dtype=np.uint8))
        assert result.raw_metric == pytest.approx(0.77)
        assert result.source == "univfd_expert"

    def test_the_module_actually_imports_numpy(self, tmp_path, monkeypatch):
        """那次真机崩溃的直接原因（`_score` 用了 `np`，模块里没有它）。"""
        module = self._expert(tmp_path, monkeypatch)
        assert hasattr(module, "np"), "univfd 的打分路径用 np，模块必须导入它"


class TestLicencePosture:
    """G7-5: the adapter may exist before the terms do."""

    def test_probe_declares_its_licence_as_unconfirmed(self):
        note = ProbeDinoExpert.licence_note
        assert "待确认" in note and "不再分发" in note
        assert "不构成使用授权" in note

    def test_trufor_declares_non_profit_only(self):
        assert "非营利" in TruForExpert.licence_note

    def test_the_weights_environment_variable_has_no_default(self):
        assert ProbeDinoExpert(weights_path=None).weights_path == ""
        assert TruForExpert(weights_path=None).weights_path == ""


class TestProbeScoring:
    def test_the_score_is_the_raw_metric_and_the_strength(self):
        expert = MockProbeExpert(score=0.8123)
        result = expert.analyze(np.zeros((64, 64, 3), dtype=np.uint8))
        assert result.raw_metric == pytest.approx(0.8123)
        assert result.strength == pytest.approx(0.8123)

    def test_the_mock_is_deterministic(self):
        image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
        first = MockProbeExpert().analyze(image).raw_metric
        second = MockProbeExpert().analyze(image).raw_metric
        assert first == second
        assert 0.0 <= first <= 1.0

    def test_the_adapter_still_claims_no_direction_of_its_own(self):
        """Even with a mapping in place, the adapter is not the one reading it.

        The mapping lives in the calibration table and is applied downstream;
        the adapter's job is to report a measurement.  Its support label stays
        `Uncertain` until the pipeline attaches `calibrated_likelihood` — the
        rectifier is what turns a probability into a direction.
        """
        result = MockProbeExpert(score=0.99).analyze(np.zeros((16, 16, 3), np.uint8))
        assert result.support == UNCALIBRATED_SUPPORT
        assert "calibrated_likelihood" in result.interpretation_text

    def test_an_unmapped_expert_says_so(self):
        """TruFor has no mapping, and its text must not imply one."""
        result = MockTruForExpert(score=0.9).analyze(np.zeros((16, 16, 3), np.uint8))
        assert result.metadata["calibration_status"] == "unmapped"
        assert "尚未在本地拟合组上建立概率映射" in result.interpretation_text

    def test_the_text_does_not_invent_physics(self):
        """A discriminator's number is not a measurement of a physical trace."""
        result = MockProbeExpert(score=0.7).analyze(np.zeros((16, 16, 3), np.uint8))
        text = result.phenomenon + result.reasoning + result.counter_explanation
        for forbidden in ("PRNU", "噪声指纹", "频谱异常", "物理痕迹"):
            assert forbidden not in text

    def test_the_third_layer_says_what_it_cannot_show(self):
        result = MockProbeExpert(score=0.7).analyze(np.zeros((16, 16, 3), np.uint8))
        assert "只有在本地校准集上" in result.counter_explanation

    def test_provenance_travels_with_the_token(self):
        result = MockProbeExpert(score=0.5).analyze(np.zeros((16, 16, 3), np.uint8))
        assert result.metadata["calibration_status"] == "mapped"
        assert "preprocess" in result.metadata

    def test_a_cached_backbone_counts_as_available(self, tmp_path, monkeypatch):
        """Once downloaded, the mirror setting is no longer needed."""
        from experts import probe_dino

        cache = tmp_path / "hub" / "models--facebook--dinov2-with-registers-large"
        cache.mkdir(parents=True)
        monkeypatch.setattr(probe_dino, "_HF_CACHE", str(tmp_path / "hub"))
        expert = ProbeDinoExpert(weights_path=str(tmp_path / "w.pth"))
        assert expert.backbone_cached() is True

    def test_a_missing_hf_endpoint_is_explained_before_transformers_fails(self, tmp_path):
        weights = tmp_path / "model.pth"
        weights.write_bytes(b"x")
        expert = ProbeDinoExpert(weights_path=str(weights))
        if expert._missing_dependency():               # torch absent: skip
            pytest.skip("torch/transformers not installed")
        reason = expert.unavailable_reason()
        assert reason == "" or "hf-mirror" in reason


class TestTruForSummaries:
    def _arrays(self, anomaly=None, confidence=None, score=0.42):
        arrays = {"score": np.array(score), "imgsize": np.array([64, 64])}
        if anomaly is not None:
            arrays["map"] = np.asarray(anomaly, dtype=np.float32)
        if confidence is not None:
            arrays["conf"] = np.asarray(confidence, dtype=np.float32)
        return arrays

    def test_it_reports_the_detection_score(self):
        stats = TruForExpert.summarise(self._arrays())
        assert stats["score"] == pytest.approx(0.42)

    def test_high_response_area_is_measured_against_the_recorded_threshold(self):
        anomaly = np.zeros((10, 10))
        anomaly[:3, :] = 0.9                     # 30 of 100 pixels
        stats = TruForExpert.summarise(self._arrays(anomaly=anomaly, confidence=np.ones((10, 10))))
        assert stats["high_response_fraction"] == pytest.approx(0.30)
        assert stats["anomaly_peak"] == pytest.approx(0.9)

    def test_confidence_weighting_uses_only_trusted_pixels(self):
        anomaly = np.ones((4, 4))
        confidence = np.zeros((4, 4))            # TruFor trusts nothing here
        stats = TruForExpert.summarise(self._arrays(anomaly=anomaly, confidence=confidence))
        assert stats["confidence_weighted_anomaly"] == pytest.approx(0.0)
        assert stats["low_confidence_fraction"] == 1.0

    def test_a_flat_map_has_no_entropy(self):
        stats = TruForExpert.summarise(self._arrays(anomaly=np.full((8, 8), 0.5)))
        assert stats["anomaly_entropy"] == pytest.approx(0.0)

    def test_a_bimodal_map_has_more_entropy_than_a_flat_one(self):
        flat = _entropy(np.full((16, 16), 0.5))
        half = np.zeros((16, 16))
        half[:8, :] = 1.0
        assert _entropy(half) > flat

    def test_missing_maps_do_not_crash_the_summary(self):
        assert TruForExpert.summarise({"score": np.array(0.1)}) == {"score": 0.1}

    def test_the_metadata_records_the_threshold_used(self):
        expert = MockTruForExpert(score=0.3)
        result = expert.analyze(np.zeros((32, 32, 3), dtype=np.uint8))
        assert result.metadata["high_response_threshold"] == HIGH_RESPONSE_THRESHOLD

    def test_the_token_is_global_scope_and_uncalibrated(self):
        """G7-4: the task stays global even though TruFor has maps."""
        expert = MockTruForExpert(score=0.3)
        result = expert.analyze(np.zeros((32, 32, 3), dtype=np.uint8))
        assert result.region == "full_image"
        assert result.support == UNCALIBRATED_SUPPORT
        assert "区域" not in result.interpretation_text or "待 G7-1" in result.interpretation_text


class TestTruForArtifacts:
    def test_artifacts_are_painted_from_the_analysis_not_recomputed(self):
        expert = MockTruForExpert(score=0.5)
        expert.analyze(np.zeros((32, 32, 3), dtype=np.uint8))
        artifacts = expert.render_artifacts(np.zeros((32, 32, 3), dtype=np.uint8))
        assert set(artifacts) == {"anomaly_map", "confidence_map"}
        for image in artifacts.values():
            assert image.dtype == np.uint8 and image.ndim == 3

    def test_no_analysis_means_no_artifacts(self):
        assert MockTruForExpert(score=0.5).render_artifacts(np.zeros((8, 8, 3), np.uint8)) == {}


class TestRawProductsAndCache:
    """G7-2: the raw products are the evidence; they are kept, not discarded."""

    def _expert(self, **kwargs):
        return MockTruForExpert(score=0.42, **kwargs)

    def test_the_result_is_persisted_where_it_can_be_re_read(self, tmp_path):
        expert = self._expert(results_dir=str(tmp_path))
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        expert.analyze(image)
        key = expert.cache_key(image)
        assert os.path.exists(expert.result_path(key))
        assert os.path.exists(expert.meta_path(key))

    def test_the_metadata_carries_weights_source_licence_and_timings(self, tmp_path):
        import json as _json

        expert = self._expert(results_dir=str(tmp_path))
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        expert.analyze(image)
        with open(expert.meta_path(expert.cache_key(image)), encoding="utf-8") as handle:
            meta = _json.load(handle)
        assert meta["weights"]["source"] == "mock"
        assert "mock" in meta["weights"]["licence"]
        assert meta["cold_start"] is True and meta["elapsed_s"] >= 0
        assert meta["stats"]["score"] == pytest.approx(0.42)
        assert meta["npz_keys"] == ["conf", "imgsize", "map", "score"]
        assert meta["recipe_version"]

    def test_a_second_identical_image_is_served_from_the_cache(self, tmp_path):
        expert = self._expert(results_dir=str(tmp_path))
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        first = expert.analyze(image)
        second = expert.analyze(image)
        assert expert.tool_calls == 1                 # the model ran once
        assert second.raw_metric == first.raw_metric
        assert expert.timing_summary()["cache_hits"] == 1

    def test_a_different_image_is_not_served_from_the_cache(self, tmp_path):
        expert = self._expert(results_dir=str(tmp_path))
        expert.analyze(np.zeros((32, 32, 3), dtype=np.uint8))
        expert.analyze(np.full((32, 32, 3), 7, dtype=np.uint8))
        assert expert.tool_calls == 2

    def test_different_weights_invalidate_the_cache(self, tmp_path):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        first = self._expert(results_dir=str(tmp_path))
        first.analyze(image)
        other_weights = tmp_path / "other.pth"
        other_weights.write_bytes(b"different")
        second = self._expert(results_dir=str(tmp_path), weights_path=str(other_weights))
        second.analyze(image)
        assert second.tool_calls == 1                 # nothing was reused

    def test_turning_the_cache_off_still_keeps_the_products(self, tmp_path):
        """
        Reusing a result and recording it are different decisions: the raw
        products are the run's evidence and are always written; `cache=False`
        only stops the adapter from serving a previous run.
        """
        expert = self._expert(results_dir=str(tmp_path), cache=False)
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        expert.analyze(image)
        expert.analyze(image)
        assert expert.tool_calls == 2                 # recomputed, not reused
        assert os.path.exists(expert.result_path(expert.cache_key(image)))

    def test_timing_summary_separates_cold_from_warm(self, tmp_path):
        expert = self._expert(results_dir=str(tmp_path))
        expert.analyze(np.zeros((16, 16, 3), dtype=np.uint8))
        expert.analyze(np.full((16, 16, 3), 3, dtype=np.uint8))
        summary = expert.timing_summary()
        assert summary["runs"] == 2 and summary["mode"] == "per_image"
        assert summary["cold_start_s"] is not None
        assert summary["warm_mean_s"] is not None

    def test_a_batch_timing_is_reported_per_image_not_per_batch(self, tmp_path):
        """
        Reporting a batch's wall time in the per-image slot makes a fast
        pipeline look slow — the review caught exactly that in the A-group
        report, where whole-batch seconds sat next to per-image ones.
        """
        expert = self._expert(results_dir=str(tmp_path))
        expert.analyze_many({f"i{i}": np.full((16, 16, 3), i, dtype=np.uint8)
                             for i in range(6)})
        summary = expert.timing_summary()
        assert summary["mode"] == "batch"
        assert summary["images_in_batches"] == 6
        assert summary["per_image_mean_s"] < 1.0        # not the 6-image wall time
        assert summary["total_s"] >= summary["per_image_mean_s"]

    def test_the_persisted_map_can_be_re_summarised_without_the_model(self, tmp_path):
        """The point of keeping the npz: a new threshold is an offline change."""
        expert = self._expert(results_dir=str(tmp_path))
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        expert.analyze(image)
        stored = dict(np.load(expert.result_path(expert.cache_key(image))))
        assert TruForExpert.summarise(stored)["score"] == pytest.approx(0.42)


class TestBatchInvocation:
    """
    The measured reason batch exists: 12.4 s cold and 12.2 s warm for one
    image means almost all of it is the interpreter, the imports and the
    checkpoint.  One process for many images pays that once.
    """

    def test_many_images_cost_one_tool_call(self, tmp_path):
        expert = MockTruForExpert(score=0.4, results_dir=str(tmp_path))
        images = {f"img_{i}": np.full((16, 16, 3), i, dtype=np.uint8) for i in range(5)}
        results = expert.analyze_many(images)
        assert len(results) == 5
        assert expert.tool_calls == 1                 # one process, five images

    def test_every_batch_result_is_persisted_individually(self, tmp_path):
        expert = MockTruForExpert(score=0.4, results_dir=str(tmp_path))
        images = {f"img_{i}": np.full((16, 16, 3), i, dtype=np.uint8) for i in range(3)}
        results = expert.analyze_many(images)
        for image_id, image in images.items():
            key = expert.cache_key(image)
            assert os.path.exists(expert.result_path(key))
            assert results[image_id].raw_metric == pytest.approx(0.4)

    def test_the_batch_timing_is_reported_per_image(self, tmp_path):
        expert = MockTruForExpert(score=0.4, results_dir=str(tmp_path))
        expert.analyze_many({f"i{i}": np.full((16, 16, 3), i, dtype=np.uint8)
                             for i in range(4)})
        summary = expert.timing_summary()
        assert summary["runs"] == 4
        assert expert.timings[-1]["batch_size"] == 4
        assert expert.timings[-1]["per_image_s"] >= 0

    def test_a_second_batch_is_served_from_the_cache(self, tmp_path):
        expert = MockTruForExpert(score=0.4, results_dir=str(tmp_path))
        images = {"a": np.zeros((16, 16, 3), dtype=np.uint8)}
        expert.analyze_many(images)
        expert.analyze_many(images)
        assert expert.tool_calls == 1
        assert expert.timing_summary()["cache_hits"] == 1

    def test_a_batch_mixed_with_cached_images_only_runs_the_missing_ones(self, tmp_path):
        expert = MockTruForExpert(score=0.4, results_dir=str(tmp_path))
        expert.analyze_many({"a": np.zeros((16, 16, 3), dtype=np.uint8)})
        results = expert.analyze_many({
            "a": np.zeros((16, 16, 3), dtype=np.uint8),
            "b": np.full((16, 16, 3), 9, dtype=np.uint8)})
        assert expert.tool_calls == 2                 # second call was the batch
        assert len(results) == 2
        assert expert.timings[-1]["batch_size"] == 1  # only "b" was missing

    def test_a_missing_result_is_reported_rather_than_silently_dropped(self, tmp_path):
        class ForgetfulMock(MockTruForExpert):
            def _run_tool_directory(self, in_dir, out_dir):
                return 0, "", ""                      # writes nothing

        expert = ForgetfulMock(score=0.4, results_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match="批量运行缺少"):
            expert.analyze_many({"a": np.zeros((16, 16, 3), dtype=np.uint8)})


class TestTruForCommand:
    def test_the_cli_matches_the_upstream_entry_point(self, tmp_path):
        home = tmp_path / "TruFor"
        (home / "TruFor_train_test").mkdir(parents=True)
        (home / "pretrained_models").mkdir()
        weights = home / "pretrained_models" / "trufor.pth.tar"
        weights.write_bytes(b"w")
        expert = TruForExpert(weights_path=str(weights), device="cpu")
        command = expert.command("/tmp/in.png", "/tmp/out")
        assert command[1] == "test.py"
        assert "-exp" in command and "trufor_ph3" in command
        assert command[command.index("-g") + 1] == "-1"        # CPU mode
        assert "TEST.MODEL_FILE" in command
        assert command[command.index("TEST.MODEL_FILE") + 1] == str(weights)

    def test_saving_noiseprint_is_opt_in(self, tmp_path):
        home = tmp_path / "TruFor"
        (home / "TruFor_train_test").mkdir(parents=True)
        weights = home / "pretrained_models" / "trufor.pth.tar"
        weights.parent.mkdir()
        weights.write_bytes(b"w")
        assert "--save_np" not in TruForExpert(weights_path=str(weights)).command("i", "o")
        assert "--save_np" in TruForExpert(
            weights_path=str(weights), save_noiseprint=True).command("i", "o")

    def test_home_is_derived_from_the_weights_path(self, tmp_path):
        home = tmp_path / "TruFor"
        weights = home / "pretrained_models" / "trufor.pth.tar"
        weights.parent.mkdir(parents=True)
        weights.write_bytes(b"w")
        expert = TruForExpert(weights_path=str(weights))
        assert expert.home == str(home)
        assert expert.script().endswith("TruFor_train_test/test.py")

    def test_a_missing_checkout_says_which_script_is_missing(self, tmp_path):
        weights = tmp_path / "trufor.pth.tar"
        weights.write_bytes(b"w")
        reason = TruForExpert(weights_path=str(weights)).unavailable_reason()
        assert "TruFor_train_test/test.py" in reason
