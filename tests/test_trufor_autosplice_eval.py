"""Pure aggregation checks for the expanded TruFor localization script."""

import pytest

from scripts.trufor_autosplice_eval import _source_aggregate

pytestmark = pytest.mark.core


def _check(source_id, label, variant_id, score, iou=None, high=0.0):
    return {
        "source_id": source_id,
        "label": label,
        "variant_id": variant_id,
        "score": score,
        "iou": iou,
        "f1": iou,
        "precision": iou,
        "recall": iou,
        "map_high_fraction": high,
        "map_inside_mean": iou,
        "map_outside_mean": 0.0,
    }


def test_variants_are_aggregated_by_source_not_counted_as_independent():
    checks = [
        _check("a", "Real", "a__authentic", 0.2, high=0.0),
        _check("a", "Fake", "a__0", 0.8, iou=0.6, high=0.2),
        _check("a", "Fake", "a__1", 0.4, iou=0.2, high=0.1),
        _check("b", "Real", "b__authentic", 0.9, high=0.3),
    ]
    out = _source_aggregate(checks)
    assert out["n_sources"] == 2
    assert out["n_forged_sources"] == 1
    assert out["n_real_sources"] == 2
    assert out["forged_variant_iou"]["mean"] == pytest.approx(0.4)
    assert out["forged_source_iou_mean"]["mean"] == pytest.approx(0.4)
    assert out["real_false_positive_sources"][0]["source_id"] == "b"
    source = next(row for row in out["sources"] if row["source_id"] == "a")
    assert source["variant_count"] == 2
    assert source["forged_summary"]["score_mean"] == pytest.approx(0.6)
