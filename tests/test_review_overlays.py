"""CPU checks for review-only prediction-box overlays."""

import os

import cv2
import numpy as np
import pytest

from utils.review_overlays import overlay_artifact, overlay_original, render_record_overlays

pytestmark = pytest.mark.core


def _write_image(path, width=120, height=80, colour=(40, 80, 120)):
    image = np.full((height, width, 3), colour, dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (30, 30), (255, 255, 255), -1)
    cv2.imwrite(str(path), image)
    return str(path)


def test_original_overlay_changes_the_image(tmp_path):
    source = _write_image(tmp_path / "original.png")
    target = str(tmp_path / "overlay.png")
    overlay_original(source, [
        {"bbox": [100, 100, 300, 300], "reason": "seam", "evidence": ["E-1"]},
        {"bbox": [0, 0, 1000, 1000], "reason": "global", "evidence": ["E-2"]},
    ], target)
    before = cv2.imread(source)
    after = cv2.imread(target)
    assert after.shape == before.shape
    assert np.any(after != before)


def test_artifact_overlay_maps_content_box_and_keeps_canvas(tmp_path):
    content = np.full((80, 120, 3), (40, 80, 120), dtype=np.uint8)
    canvas = np.full((160, 300, 3), 245, dtype=np.uint8)
    canvas[38:118, :120] = content
    artifact = str(tmp_path / "artifact.png")
    cv2.imwrite(artifact, canvas)
    target = str(tmp_path / "artifact_overlay.png")
    overlay_artifact(artifact, [0, 38, 120, 118], [
        {"bbox": [100, 100, 300, 300], "reason": "seam", "evidence": ["E-1"]},
    ], target)
    out = cv2.imread(target)
    assert out.shape == canvas.shape
    # The overlay is inside the content box, not in the legend margin.
    assert np.any(out[38:118, :120] != canvas[38:118, :120])
    assert np.all(out[:, 122:] == canvas[:, 122:])


def test_record_overlay_writes_original_and_anomaly_outputs(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    original = _write_image(project / "image.png")
    anomaly = _write_image(project / "anomaly.png")
    record = {
        "trajectory_id": "session-x",
        "image_path": "image.png",
        "suspicious_regions": [
            {"bbox": [100, 100, 300, 300], "reason": "seam", "evidence": ["E-1"]},
        ],
        "evidence": [{
            "source": "trufor_expert",
            "visual_artifacts": ["anomaly.png"],
            "artifact_layout": {"anomaly_map": {"content_box": [0, 0, 120, 80]}},
        }],
    }
    paths = render_record_overlays(record, str(project / "overlays"),
                                   project_root=str(project))
    assert len(paths) == 2
    assert all(os.path.exists(project / p) for p in paths)
    assert any("original_predictions" in p for p in paths)
    assert any("anomaly_predictions" in p for p in paths)


def test_record_without_regions_writes_nothing(tmp_path):
    assert render_record_overlays(
        {"trajectory_id": "x", "image_path": "image.png",
         "suspicious_regions": []},
        str(tmp_path / "out"), project_root=str(tmp_path)) == []
