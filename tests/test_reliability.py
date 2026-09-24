"""Tests for the G2 reliability table and the Evidence Bundle fields."""

import json

import pytest

from state_machine.evidence_tokenizer import EvidenceTokenizer
from utils.reliability import ReliabilityTable


def _write_table(tmp_path, experts):
    path = tmp_path / "reliability_table.json"
    path.write_text(json.dumps({"experts": experts}), encoding="utf-8")
    return str(path)


class TestReliabilityTableLoading:
    def test_missing_file_returns_none(self, tmp_path):
        assert ReliabilityTable.load(str(tmp_path / "nope.json")) is None

    def test_corrupt_file_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        assert ReliabilityTable.load(str(path)) is None

    def test_loads_experts(self, tmp_path):
        path = _write_table(tmp_path, {"noise_expert": {"bins": []}})
        table = ReliabilityTable.load(path)
        assert table is not None
        assert table.expert_entry("noise_expert") is not None


class TestReliabilityLookup:
    def _table(self, tmp_path):
        return ReliabilityTable.load(_write_table(tmp_path, {
            "noise_expert": {
                "semantics_aligned": False,
                "separation_polarity_corrected": 0.845,
                "confound_drop_png_to_q70": -0.073,
                "applicability": "inverted:high-metric-means-real",
                "bins": [
                    {"lo": None, "hi": 2.0, "n": 100, "p_fake": 0.8},
                    {"lo": 2.0, "hi": 4.0, "n": 120, "p_fake": 0.5},
                    {"lo": 4.0, "hi": None, "n": 90, "p_fake": 0.2},
                ],
            },
        }))

    def test_lookup_first_bin(self, tmp_path):
        result = self._table(tmp_path).lookup("noise_expert", 1.0)
        assert result["calibrated_likelihood"] == {"Real": pytest.approx(0.2), "Fake": pytest.approx(0.8)}
        assert result["semantics_aligned"] is False
        assert result["reliability"] == pytest.approx(0.845)

    def test_lookup_open_ended_high_bin(self, tmp_path):
        result = self._table(tmp_path).lookup("noise_expert", 99.0)
        assert result["calibrated_likelihood"]["Fake"] == pytest.approx(0.2)

    def test_lookup_unknown_expert(self, tmp_path):
        assert self._table(tmp_path).lookup("ghost_expert", 1.0) is None

    def test_condition_metadata_carries_applicability_context(self, tmp_path):
        result = self._table(tmp_path).lookup("noise_expert", 3.0)
        assert result["condition_metadata"]["bin_samples"] == 120
        assert result["condition_metadata"]["separation_polarity_corrected"] == pytest.approx(0.845)

    def test_lookup_passes_applicability_conditions(self, tmp_path):
        """G2-c: usage conditions must travel with the measurement."""
        path = _write_table(tmp_path, {
            "noise_expert": {
                "semantics_aligned": False,
                "applicability": "inverted:high-metric-means-real",
                "applicability_conditions": "高 strength 统计上对应 Real；不得按 AI-generated 解读。",
                "bins": [{"lo": None, "hi": None, "n": 10, "p_fake": 0.5}],
            },
        })
        table = ReliabilityTable.load(path)
        result = table.lookup("noise_expert", 1.0)
        assert result["applicability"] == "inverted:high-metric-means-real"
        assert "对应 Real" in result["applicability_conditions"]
        assert result["semantics_aligned"] is False


class TestBundleFields:
    class FakeResult:
        evidence_name = "noise_residual_inconsistency"
        phenomenon = "phenomenon"
        reasoning = "reasoning"
        strength = 0.76
        source = "noise_expert"
        support = "AI-generated"
        interpretation_text = "severe anomaly"
        raw_metric = 3.21
        counter_explanation = "降噪与重压缩也会改变局部噪声一致性。"

    def test_bundle_fields_included_when_provided(self):
        token = EvidenceTokenizer.tokenize(
            self.FakeResult(), [10, 20, 30, 40], (100, 200),
            reliability=0.845,
            calibrated_likelihood={"Real": 0.2, "Fake": 0.8},
            condition_metadata={"bin_samples": 120},
            visual_artifacts=["traces/evidence/s1/noise_residual_map_E-abc.png"],
            semantics_aligned=False,
            applicability="inverted:high-metric-means-real",
            applicability_conditions="高 strength 统计上对应 Real。",
        )
        assert token["raw_metric"] == pytest.approx(3.21)
        assert token["counter_explanation"].startswith("降噪")
        assert token["reliability"] == pytest.approx(0.845)
        assert token["calibrated_likelihood"]["Fake"] == pytest.approx(0.8)
        assert token["condition_metadata"]["bin_samples"] == 120
        assert token["visual_artifacts"][0].endswith(".png")
        # G2-c semantic transmission
        assert token["semantics_aligned"] is False
        assert token["applicability"] == "inverted:high-metric-means-real"
        assert "对应 Real" in token["applicability_conditions"]

    def test_bundle_fields_absent_when_unavailable(self):
        class BareResult:
            evidence_name = "x"
            phenomenon = "p"
            reasoning = "r"
            strength = 0.1
            source = "s"
            support = "Real"
            interpretation_text = "t"

        token = EvidenceTokenizer.tokenize(BareResult(), [0, 0, 10, 10], (100, 100))
        assert "reliability" not in token
        assert "calibrated_likelihood" not in token
        assert "visual_artifacts" not in token
        assert "semantics_aligned" not in token
        assert "applicability" not in token
        assert token["raw_metric"] == 0.0
