"""Review flags for predicted suspicious regions (never bbox replacement)."""

import pytest

from utils.bbox_review import review_flags

pytestmark = pytest.mark.core


def test_pixel_dimension_bbox_is_flagged_not_changed():
    regions = [{"bbox": [0, 0, 341, 256], "reason": "badge"}]
    flags = review_flags(regions, (256, 341), "image size is 341x256")
    assert flags[0]["bbox"] == [0, 0, 341, 256]
    assert "pixel_dimension_bbox" in flags[0]["flags"]
    assert "pixel width/height" in flags[0]["detail"]


def test_location_text_mismatch_marks_adm_style_box():
    regions = [{"bbox": [100, 200, 300, 400],
                "reason": "下方黑色物体形态异常"}]
    flags = review_flags(regions, (500, 500), "下方")
    assert "location_text_mismatch" in flags[0]["flags"]


def test_anomaly_peak_outside_bbox_marks_57501_style_box():
    regions = [{"bbox": [100, 100, 500, 500],
                "reason": "局部异常"}]
    flags = review_flags(regions, (500, 500), "",
                         {"anomaly_peak_normalized_1000": [60, 920]})
    assert "anomaly_peak_outside_bbox" in flags[0]["flags"]
    assert "anomaly peak" in flags[0]["detail"]


def test_global_box_is_flagged_as_global_range():
    flags = review_flags([{"bbox": [0, 0, 1000, 1000], "reason": "whole"}],
                         (500, 500))
    assert "global_range" in flags[0]["flags"]
    assert "pixel_dimension_bbox" not in flags[0]["flags"]


def test_matching_box_and_peak_has_no_flags():
    flags = review_flags([{"bbox": [100, 100, 900, 900], "reason": "object"}],
                         (500, 500), "",
                         {"anomaly_peak_normalized_1000": [500, 500]})
    assert flags == []
