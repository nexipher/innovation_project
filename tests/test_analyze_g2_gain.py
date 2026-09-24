"""Tests for the G2-d paired analysis (McNemar + paired bootstrap)."""

import pytest

from scripts.analyze_g2_gain import (
    align_records,
    analyze,
    class_breakdown,
    mcnemar,
    paired_bootstrap_auroc,
)


class TestMcNemar:
    def test_known_example(self):
        """b=16, c=1 → (|16-1|-1)²/17 with 1 df."""
        baseline = [True] * 16 + [False]
        other = [False] * 16 + [True]
        result = mcnemar(baseline, other)
        assert result["b"] == 16 and result["c"] == 1
        assert result["statistic"] == pytest.approx(14 ** 2 / 17)
        assert result["p_value"] == pytest.approx(0.00069, abs=1e-5)

    def test_no_disagreement_is_not_significant(self):
        result = mcnemar([True, False], [True, False])
        assert result["b"] + result["c"] == 0
        assert result["p_value"] == 1.0

    def test_symmetric_swaps_are_not_significant(self):
        baseline = [True] * 10 + [False] * 10
        other = [False] * 10 + [True] * 10
        assert mcnemar(baseline, other)["p_value"] > 0.5

    def test_one_sided_loss_is_significant(self):
        baseline = [True] * 30 + [False] * 10
        other = [False] * 30 + [False] * 10
        assert mcnemar(baseline, other)["p_value"] < 0.001


class TestAlignRecords:
    def test_keeps_only_shared_samples(self):
        conditions = {
            "rgb": [{"sample_id": "a"}, {"sample_id": "b"}],
            "text": [{"sample_id": "b"}, {"sample_id": "c"}],
        }
        order, by_condition = align_records(conditions)
        assert order == ["b"]
        assert by_condition["rgb"]["b"]["sample_id"] == "b"

    def test_no_overlap_raises(self):
        with pytest.raises(SystemExit):
            align_records({"rgb": [{"sample_id": "a"}],
                           "text": [{"sample_id": "b"}]})


def _record(sample_id, gt, verdict, confidence):
    return {"sample_id": sample_id, "gt": gt, "verdict": verdict,
            "confidence": confidence, "cell": "x"}


class TestClassBreakdown:
    def test_counts_the_bias(self):
        records = [
            _record("1", "Real", "Real", 0.9),
            _record("2", "Fake", "Real", 0.9),
            _record("3", "Fake", "Uncertain", 0.5),
        ]
        breakdown = class_breakdown(records)
        assert breakdown["real_recall"] == pytest.approx(1.0)
        assert breakdown["fake_recall"] == pytest.approx(0.0)
        assert breakdown["uncertain_rate"] == pytest.approx(1 / 3)


class TestPairedBootstrap:
    def _conditions(self, worse: bool):
        """Six samples; the `worse` arm inverts the baseline's correct answers."""
        pairs = [("1", "Fake"), ("2", "Fake"), ("3", "Fake"),
                 ("4", "Real"), ("5", "Real"), ("6", "Real")]
        rgb, text = {}, {}
        for sample_id, gt in pairs:
            rgb[sample_id] = _record(sample_id, gt, gt, 0.9)
            wrong = "Fake" if gt == "Real" else "Real"
            text[sample_id] = _record(sample_id, gt, wrong if worse else gt, 0.9)
        return {"rgb": rgb, "text": text}, [p[0] for p in pairs]

    def test_deterministic_under_seed(self):
        by_condition, order = self._conditions(worse=False)
        first = paired_bootstrap_auroc(by_condition, order, iterations=200)
        second = paired_bootstrap_auroc(by_condition, order, iterations=200)
        assert first["text"]["auroc_ci"] == second["text"]["auroc_ci"]

    def test_identical_arm_has_zero_delta(self):
        by_condition, order = self._conditions(worse=False)
        result = paired_bootstrap_auroc(by_condition, order, iterations=200)
        assert result["text"]["delta_auroc"] == pytest.approx(0.0)

    def test_worse_arm_delta_interval_is_negative(self):
        by_condition, order = self._conditions(worse=True)
        result = paired_bootstrap_auroc(by_condition, order, iterations=200)
        low, high = result["text"]["delta_ci"]
        assert high < 0


class TestAnalyze:
    def test_reports_every_arm_against_the_baseline(self):
        report = {"conditions": {
            "rgb": {"records": [
                _record("1", "Fake", "Real", 0.9),
                _record("2", "Real", "Real", 0.9),
                _record("3", "Fake", "Fake", 0.8),
            ]},
            "text": {"records": [
                _record("1", "Fake", "Fake", 0.9),
                _record("2", "Real", "Fake", 0.9),
                _record("3", "Fake", "Fake", 0.8),
            ]},
        }}
        analysis = analyze(report, iterations=100)
        assert analysis["paired_samples"] == 3
        assert analysis["arms"]["rgb"]["mcnemar"] is None
        assert analysis["arms"]["text"]["mcnemar"]["b"] == 1
        assert analysis["arms"]["text"]["mcnemar"]["c"] == 1
