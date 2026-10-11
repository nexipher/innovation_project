"""Tests for the G7 external-expert evaluation (CPU; no external models)."""

import json

import numpy as np
import pytest

from scripts.evaluate_external_expert import (
    auroc,
    bootstrap_ci,
    correlation_report,
    evaluate,
    _finite_tree,
    _round,
    load_group,
    orientation_corrected,
    summarise,
)


def _rows(spec):
    """[(source, label, treatment, score)] → the evaluator's row shape."""
    return [{"sample_id": f"{source}_{treatment}", "source_id": source,
             "label": label, "treatment": treatment, "cell": f"{label.lower()}_{treatment}",
             "score": score}
            for source, label, treatment, score in spec]


class TestAuroc:
    def test_a_perfect_separation_is_one(self):
        assert auroc([(0.9, "Fake"), (0.85, "Fake"), (0.8, "Fake"),
                      (0.2, "Real"), (0.15, "Real"), (0.1, "Real")]) == 1.0

    def test_an_inverted_score_is_zero(self):
        assert auroc([(0.1, "Fake"), (0.15, "Fake"), (0.2, "Fake"),
                      (0.8, "Real"), (0.85, "Real"), (0.9, "Real")]) == 0.0

    def test_too_few_samples_reports_nothing(self):
        """Three per class, as everywhere else in the project — below that a
        rank statistic is noise pretending to be a measurement."""
        assert auroc([(0.9, "Fake"), (0.8, "Fake"), (0.2, "Real"), (0.1, "Real")]) is None


class TestOrientation:
    def test_the_polarity_is_measured_not_assumed(self):
        inverted = _rows([(f"s{i}", "Fake", "png", 0.10 + i * 0.01) for i in range(5)] +
                         [(f"r{i}", "Real", "png", 0.80 + i * 0.01) for i in range(5)])
        report = orientation_corrected(inverted)
        assert report["polarity"] == -1
        assert report["separated"] > 0.9

    def test_a_constant_score_has_no_direction(self):
        flat = _rows([(f"s{i}", "Fake", "png", 0.5) for i in range(5)] +
                     [(f"r{i}", "Real", "png", 0.5) for i in range(5)])
        assert orientation_corrected(flat)["auroc"] == 0.5


class TestBootstrap:
    def test_the_interval_is_computed_from_sources(self):
        rows = _rows([(f"s{i}", "Fake", "png", 0.9) for i in range(8)] +
                     [(f"r{i}", "Real", "png", 0.1) for i in range(8)])
        ci = bootstrap_ci(rows, lambda rs: auroc([(r["score"], r["label"]) for r in rs]),
                          rounds=200)
        assert ci is not None and ci[0] <= ci[1]

    def test_too_few_sources_reports_nothing(self):
        rows = _rows([("s", "Fake", "png", 0.9), ("r", "Real", "png", 0.1)])
        assert bootstrap_ci(rows, lambda rs: 1.0, rounds=50) is None


class TestSummarise:
    def test_it_reports_each_treatment_separately(self):
        rows = _rows(
            [(f"s{i}", "Fake", "png", 0.9) for i in range(5)] +
            [(f"r{i}", "Real", "png", 0.1) for i in range(5)] +
            [(f"s{i}", "Fake", "jpeg_q70", 0.4) for i in range(5)] +
            [(f"r{i}", "Real", "jpeg_q70", 0.6) for i in range(5)])
        report = summarise(rows, "A")
        assert report["per_treatment"]["png"]["auroc"] == 1.0
        assert report["per_treatment"]["jpeg_q70"]["auroc"] == 0.0
        assert report["group"] == "A"

    def test_a_broken_map_statistic_does_not_break_the_report(self):
        rows = _rows([(f"s{i}", "Fake", "png", 0.9) for i in range(5)] +
                     [(f"r{i}", "Real", "png", 0.1) for i in range(5)])
        for row in rows:
            row["map_anomaly_peak"] = float("nan")
        report = summarise(rows, "A")
        assert json.dumps(report, allow_nan=False)      # no NaN survives


class TestCorrelation:
    def test_a_constant_score_reports_no_correlation_rather_than_nan(self):
        rows = _rows([(f"s{i}", "Fake", "png", 0.5) for i in range(12)])
        cached = {"noise": {row["sample_id"]: float(i) for i, row in enumerate(rows)}}
        report = correlation_report(rows, cached)
        assert report["noise"]["spearman"] is None
        assert "恒定" in report["noise"]["note"]
        assert json.dumps(report, allow_nan=False)

    def test_a_real_correlation_is_reported(self):
        rows = _rows([(f"s{i}", "Fake", "png", float(i)) for i in range(12)])
        cached = {"noise": {row["sample_id"]: float(i) for i, row in enumerate(rows)}}
        report = correlation_report(rows, cached)
        assert report["noise"]["spearman"] == pytest.approx(1.0, abs=1e-6)

    def test_too_few_overlapping_images_is_skipped(self):
        rows = _rows([("s", "Fake", "png", 0.5)])
        assert correlation_report(rows, {"noise": {"s": 0.4}}) == {}


class TestSanitising:
    def test_non_finite_floats_become_none(self):
        tree = {"a": float("nan"), "b": [float("inf"), 1.0], "c": {"d": float("-inf")}}
        cleaned = _finite_tree(tree)
        assert cleaned == {"a": None, "b": [None, 1.0], "c": {"d": None}}
        json.dumps(cleaned, allow_nan=False)

    def test_round_is_none_safe(self):
        assert _round(None) is None and _round(float("nan")) is None
        assert _round(0.123456) == 0.1235


class TestGroupLoading:
    def test_group_b_is_the_development_set(self):
        samples, split = load_group("B")
        assert len({s["source_id"] for s in samples}) == split["counts"]["B"]
        assert {s["source_id"] for s in samples} == set(split["groups"]["B"])

    def test_an_unknown_group_is_refused(self):
        with pytest.raises(SystemExit, match="unknown group"):
            load_group("test")
