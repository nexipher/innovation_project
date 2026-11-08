"""Review-only overlays: put the model's predicted boxes on the images.

The workbench must let a person check three things together: the predicted
bbox, the written description, and the anomaly response at that location.
These images are review products, not model inputs, and they never alter the
record or the trace.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional

import cv2
import numpy as np

from config import PROJECT_ROOT

RED = (0, 0, 255)


def _pixels(bbox, width: int, height: int):
    """[ymin,xmin,ymax,xmax] in [0,1000] → pixel [x0,y0,x1,y1]."""
    ymin, xmin, ymax, xmax = [int(round(float(v))) for v in bbox]
    ymin = max(0, min(1000, ymin))
    xmin = max(0, min(1000, xmin))
    ymax = max(0, min(1000, ymax))
    xmax = max(0, min(1000, xmax))
    x0 = int(round(xmin / 1000.0 * width))
    x1 = int(round(xmax / 1000.0 * width))
    y0 = int(round(ymin / 1000.0 * height))
    y1 = int(round(ymax / 1000.0 * height))
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def _is_global(bbox) -> bool:
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return False
    return ((ymax - ymin) * (xmax - xmin)) >= 0.98 * 1000 * 1000


def _draw_regions(image: np.ndarray, regions: Iterable[dict],
                  x_offset: int = 0, y_offset: int = 0,
                  content_width: Optional[int] = None,
                  content_height: Optional[int] = None) -> np.ndarray:
    out = image.copy()
    height, width = out.shape[:2]
    content_width = content_width or width
    content_height = content_height or height
    for index, region in enumerate(regions or [], 1):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = _pixels(bbox, content_width, content_height)
        x0 += x_offset
        x1 += x_offset
        y0 += y_offset
        y1 += y_offset
        global_box = _is_global(bbox)
        colour = (0, 128, 255) if global_box else RED
        cv2.rectangle(out, (x0, y0), (x1, y1), colour, 3)
        label = (f"GLOBAL RANGE (not localized) [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]"
                 if global_box else
                 f"predicted bbox [{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]")
        text_y = max(18, y0 - 8)
        cv2.putText(out, label, (max(4, x0), text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
    return out


def overlay_original(image_path: str, regions, out_path: str):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    cv2.imwrite(out_path, _draw_regions(image, regions))
    return out_path


def overlay_artifact(artifact_path: str, content_box, regions, out_path: str):
    image = cv2.imread(artifact_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(artifact_path)
    if not (isinstance(content_box, (list, tuple)) and len(content_box) == 4):
        content_box = [0, 0, image.shape[1], image.shape[0]]
    x0, y0, x1, y1 = [int(v) for v in content_box]
    content_width = max(1, x1 - x0)
    content_height = max(1, y1 - y0)
    # Draw on the content crop, then paste it back: prediction labels are
    # clipped to the content area and cannot spill into the title/colour bar.
    content = image[y0:y1, x0:x1].copy()
    content = _draw_regions(content, regions, content_width=content_width,
                            content_height=content_height)
    out = image.copy()
    out[y0:y1, x0:x1] = content
    cv2.rectangle(out, (x0, y0), (x1, y1), (70, 70, 70), 1)
    cv2.imwrite(out_path, out)
    return out_path


def _first_trufor_token(record: dict) -> Optional[dict]:
    for token in record.get("evidence") or []:
        if isinstance(token, dict) and token.get("source") == "trufor_expert":
            return token
    return None


def _anomaly_path(token: Optional[dict]) -> Optional[str]:
    if not token:
        return None
    for path in token.get("visual_artifacts") or []:
        if "anomaly" in os.path.basename(str(path)):
            return str(path)
    return None


def render_record_overlays(record: dict, out_dir: str,
                           project_root: str = PROJECT_ROOT) -> List[str]:
    """Write original/anomaly prediction overlays; return project-relative paths.

    Missing inputs are skipped individually: a failed TruFor measurement has no
    anomaly map, but the original-image overlay can still be reviewed.
    """
    regions = record.get("suspicious_regions") or []
    if not regions:
        return []
    os.makedirs(out_dir, exist_ok=True)
    session_id = str(record.get("trajectory_id") or "session")
    original = record.get("image_path")
    outputs: List[str] = []

    if original and os.path.exists(os.path.join(project_root, original)):
        target = os.path.join(out_dir, f"{session_id}__original_predictions.png")
        overlay_original(os.path.join(project_root, original), regions, target)
        outputs.append(os.path.relpath(target, project_root).replace(os.sep, "/"))

    token = _first_trufor_token(record)
    anomaly = _anomaly_path(token)
    if anomaly:
        absolute = os.path.join(project_root, anomaly)
        if os.path.exists(absolute):
            layout = (token.get("artifact_layout") or {}).get("anomaly_map") or {}
            content_box = layout.get("content_box")
            target = os.path.join(out_dir, f"{session_id}__anomaly_predictions.png")
            overlay_artifact(absolute, content_box, regions, target)
            outputs.append(os.path.relpath(target, project_root).replace(os.sep, "/"))
    return outputs
