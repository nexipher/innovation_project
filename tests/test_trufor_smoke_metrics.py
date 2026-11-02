"""Geometry checks for the TruFor AutoSplice smoke (plan §10.5)."""

import numpy as np
import pytest

from experts.trufor import HIGH_RESPONSE_THRESHOLD
from scripts.trufor_autosplice_smoke import _metrics

pytestmark = pytest.mark.core


def test_authentic_ignores_inherited_base_mask():
    """Authentic rows may carry a base mask; it must not truncate FP stats."""
    map_array = np.array([[0.9, 0.1], [0.2, 0.05]])
    inherited = np.array([[1, 1], [0, 0]], dtype=bool)

    out = _metrics(0.9, map_array, None, inherited, "Real")

    assert out["mask_area_fraction"] == 0.0
    assert out["map_outside_mean"] == pytest.approx(float(map_array.mean()))
    assert out["map_high_fraction"] == pytest.approx(
        float((map_array >= HIGH_RESPONSE_THRESHOLD).mean()))
    assert out["iou"] is None
    assert out["f1"] is None
    assert out["precision"] is None
    assert out["recall"] is None


def test_forged_still_uses_the_mask_for_geometry():
    map_array = np.array([[0.9, 0.1], [0.2, 0.05]])
    mask = np.array([[1, 1], [0, 0]], dtype=bool)

    out = _metrics(0.8, map_array, None, mask, "Fake")

    assert out["mask_area_fraction"] == pytest.approx(0.5)
    assert out["iou"] == pytest.approx(0.5)
    assert out["precision"] == pytest.approx(1.0)
    assert out["recall"] == pytest.approx(0.5)
    assert out["map_inside_mean"] == pytest.approx(0.5)
    assert out["map_outside_mean"] == pytest.approx(0.125)
