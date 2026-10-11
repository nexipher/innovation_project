"""
Tests for the G7 baselines (CPU, cached scores only).

This script produces a *decision*, so the parts that decide are the parts that
need pinning: the metrics, the paired interval, the rule, and the gate that
keeps the validation group unread until the scheme is frozen.
"""

import json
import os

import pytest

from scripts.g7_baselines import (
    FUSION_LOSES_MARGIN,
    FUSION_WINS_MARGIN,
    _acc_f1,
    auroc,
    best_threshold,
    brier,
    decide,
    paired_delta,
    validate_c,
)


class TestMetrics:
    def test_auroc_of_a_perfect_ranking_is_one(self):
        assert auroc([0.9, 0.8, 0.7, 0.2, 0.1, 0.05], [1, 1, 1, 0, 0, 0]) == 1.0

    def test_too_few_per_class_reports_nothing(self):
        assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) is None

    def test_accuracy_and_f1_at_a_threshold(self):
        probabilities = [0.9, 0.8, 0.4, 0.1]
        labels = [1, 1, 0, 0]
        acc, f1 = _acc_f1(probabilities, labels, 0.5)
        assert acc == 1.0 and f1 == 1.0
        acc_low, _ = _acc_f1(probabilities, labels, 0.85)
        assert acc_low == 0.75                       # one positive falls below

    def test_brier_is_a_mean_squared_error(self):
        assert brier([1.0, 0.0], [1, 0]) == 0.0
        assert brier([0.0, 1.0], [1, 0]) == 1.0

    def test_the_threshold_is_chosen_by_f1_on_that_group(self):
        probabilities = [0.9, 0.85, 0.6, 0.55, 0.4, 0.1]
        labels = [1, 1, 1, 0, 0, 0]
        threshold, f1 = best_threshold(probabilities, labels)
        assert 0.55 < threshold <= 0.6 and f1 == 1.0


def _delta_of(rows: list) -> float | None:
    """AUROC(left) − AUROC(right) on one resample, or None if a class thinned out."""
    left = auroc([row["left"] for row in rows], [row["label"] for row in rows])
    right = auroc([row["right"] for row in rows], [row["label"] for row in rows])
    return None if left is None or right is None else left - right


class TestPairedDelta:
    def _rows(self, left, right, labels):
        return [{"source_id": f"s{i}", "label": y, "left": a, "right": b}
                for i, (a, b, y) in enumerate(zip(left, right, labels))]

    def test_identical_conditions_have_a_null_delta(self):
        """Same numbers on both sides: zero by construction, whatever they are."""
        left, labels = [], []
        for source in range(16):
            left.append(0.9 if source % 2 else 0.1)
            labels.append(source % 2)
        rows = self._rows(left, list(left), labels)
        entry = paired_delta(rows, _delta_of, rounds=200)
        assert entry["delta"] == 0.0
        assert entry["ci"][0] <= 0 <= entry["ci"][1]

    def test_the_interval_is_built_by_resampling_sources(self):
        """Two rows from one source must move together, or the CI is too tight."""
        rows = []
        for source in range(8):
            for _ in range(2):
                rows.append({"source_id": f"s{source}", "label": 1 if source % 2 else 0,
                             "left": 0.9 if source % 2 else 0.1,
                             "right": 0.5})
        entry = paired_delta(rows, _delta_of, rounds=200)
        assert entry["delta"] == 0.5 and entry["ci"][0] > 0     # never crosses zero

    def test_a_single_source_cannot_be_resampled(self):
        rows = self._rows([0.9, 0.1], [0.5, 0.5], [1, 0])
        assert paired_delta(rows, _delta_of, rounds=50) is None


class TestDecisionRule:
    """Step 3's rule, written before the numbers arrived."""

    def test_a_clear_fusion_gain_keeps_the_combination(self):
        assert "保留多专家组合" in decide(FUSION_WINS_MARGIN)

    def test_a_loss_hands_the_job_to_probe(self):
        assert "PROBE 作为主检测器" in decide(FUSION_LOSES_MARGIN)

    def test_a_marginal_gain_is_still_no_gain(self):
        """The band is symmetric and everything inside it is a null result."""
        assert "PROBE 作为主检测器" in decide(0.0)
        assert "PROBE 作为主检测器" in decide(FUSION_WINS_MARGIN - 1e-9)

    def test_no_numbers_is_not_a_verdict(self):
        assert "无法判定" in decide(None)


class TestValidationGate:
    def test_c_cannot_be_read_before_freezing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("scripts.g7_baselines.FREEZE_PATH",
                            str(tmp_path / "missing.json"))
        with pytest.raises(SystemExit, match="尚未冻结"):
            validate_c()

    def test_a_changed_split_invalidates_the_freeze(self, tmp_path, monkeypatch):
        freeze = tmp_path / "frozen.json"
        freeze.write_text(json.dumps({"split_hash": "different", "generated_at": "x",
                                      "conditions": {}, "verdict": "y"}),
                          encoding="utf-8")
        monkeypatch.setattr("scripts.g7_baselines.FREEZE_PATH", str(freeze))
        with pytest.raises(SystemExit, match="划分哈希"):
            validate_c()


class TestTheCommittedArtifacts:
    def test_the_freeze_file_records_the_rule_and_the_verdict(self):
        from config import PROJECT_ROOT
        path = os.path.join(PROJECT_ROOT, "calibration", "g7_frozen_scheme.json")
        with open(path, encoding="utf-8") as handle:
            freeze = json.load(handle)
        assert freeze["fitted_on"] == "A" and freeze["selected_on"] == "B"
        assert "fusion_wins_margin" in freeze["decision_rule"]
        assert set(freeze["conditions"]) == {"existing", "probe", "existing+probe"}
        for entry in freeze["conditions"].values():
            assert entry["coefficients"] and entry["threshold_b"] is not None

    def test_the_validation_report_carries_both_verdicts(self):
        from config import PROJECT_ROOT
        path = os.path.join(PROJECT_ROOT, "calibration", "g7_validation_C.json")
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["group"] == "C" and report["n"] > 0
        assert "verdict_on_b" in report and "verdict_on_c" in report
        assert "以 C 为准" in report["verdict_note"]

    def test_the_sealed_c_report_contains_no_metric(self):
        """The point of sealing: scores in, nothing that could steer a choice."""
        from config import PROJECT_ROOT
        path = os.path.join(PROJECT_ROOT, "calibration",
                            "g7_external_probe_dino_C_sealed.json")
        with open(path, encoding="utf-8") as handle:
            sealed = json.load(handle)
        assert sealed["sealed"] is True
        assert "summary" not in sealed and "correlation_with_existing" not in sealed
        assert sealed["rows"] and "score" in sealed["rows"][0]
        assert sealed["weights"]["sha256"]
