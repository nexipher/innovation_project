"""
Tests for the G7 external-expert adapters (PROBE-DINOv2, TruFor).

Everything here runs on CPU: the adapters' job is to be a well-behaved
boundary, and the parts that are testable without their weights — availability,
preprocessing arithmetic, statistic summarisation, artifact painting, the CLI
they would run — are exactly the parts that break silently when the model is
finally loaded (agent.md §3.2.2).
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

    def test_an_uncalibrated_score_claims_no_direction(self):
        result = MockProbeExpert(score=0.99).analyze(np.zeros((16, 16, 3), np.uint8))
        assert result.support == UNCALIBRATED_SUPPORT
        assert "不作为方向性证据" in result.interpretation_text

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
        assert result.metadata["calibrated"] is False
        assert "preprocess" in result.metadata

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
        assert summary["runs"] == 2
        assert summary["cold_start_s"] is not None
        assert summary["warm_mean_s"] is not None

    def test_the_persisted_map_can_be_re_summarised_without_the_model(self, tmp_path):
        """The point of keeping the npz: a new threshold is an offline change."""
        expert = self._expert(results_dir=str(tmp_path))
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        expert.analyze(image)
        stored = dict(np.load(expert.result_path(expert.cache_key(image))))
        assert TruForExpert.summarise(stored)["score"] == pytest.approx(0.42)


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
