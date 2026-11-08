"""Review flags for model-predicted suspicious regions.

These flags never change a model bbox.  They record why a person should look
at the overlay: a bbox that is exactly the original pixel size (the
41187-style pixel/normalized mix-up), a bbox that does not contain the anomaly
peak, a text/geometry contradiction (ADM's "below" box in the upper left), or
a full-image box presented as a local finding.  The review overlay remains the
authoritative picture.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

LOCATION_RULES = (
    ("头部", "top", "头"),
    ("顶部", "top", "顶"),
    ("上方", "top", "上"),
    ("上部", "top", "上"),
    ("天空", "top", "天"),
    ("底部", "bottom", "底"),
    ("下部", "bottom", "下"),
    ("下方", "bottom", "下"),
    ("左侧", "left", "左"),
    ("左边", "left", "左"),
    ("右侧", "right", "右"),
    ("右边", "right", "右"),
    ("中央", "center", "中"),
    ("中间", "center", "中"),
)


def _centre(bbox: Sequence[float]):
    ymin, xmin, ymax, xmax = [float(v) for v in bbox]
    return (ymin + ymax) / 2.0, (xmin + xmax) / 2.0


def _contains_peak(bbox, peak, margin: float = 20.0) -> bool:
    ymin, xmin, ymax, xmax = [float(v) for v in bbox]
    py, px = [float(v) for v in peak]
    return (ymin - margin <= py <= ymax + margin
            and xmin - margin <= px <= xmax + margin)


def review_flags(regions: Iterable[dict],
                 image_shape: Optional[Sequence[int]] = None,
                 text: str = "",
                 map_stats: Optional[dict] = None) -> List[dict]:
    """Return per-region review flags; the bboxes themselves are untouched."""
    height = width = None
    if image_shape and len(image_shape) >= 2:
        height, width = int(image_shape[0]), int(image_shape[1])
    peak = (map_stats or {}).get("anomaly_peak_normalized_1000")
    if not (isinstance(peak, (list, tuple)) and len(peak) == 2):
        peak = None
    found: List[dict] = []
    for index, region in enumerate(regions or [], 1):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        values = [int(round(float(v))) for v in bbox]
        ymin, xmin, ymax, xmax = values
        flags: List[str] = []
        details: List[str] = []
        area = ((ymax - ymin) * (xmax - xmin)) / (1000.0 * 1000.0)
        if area >= 0.98:
            flags.append("global_range")
            details.append("bbox covers ~the whole image; it is a global range, "
                           "not a completed local localisation")
        if (height and width and values[:2] == [0, 0]
                and ([values[2], values[3]] == [width, height]
                     or [values[2], values[3]] == [height, width])):
            flags.append("pixel_dimension_bbox")
            details.append(
                f"bbox [0,0,{values[2]},{values[3]}] equals the original pixel "
                f"width/height ({width}x{height}); this looks like a pixel-size "
                "value copied into the normalized [0,1000] field")
        if peak is not None and not _contains_peak(values, peak):
            flags.append("anomaly_peak_outside_bbox")
            details.append(
                f"anomaly peak is at normalized [y={peak[0]},x={peak[1]}], "
                f"outside bbox {values}")
        # Text/geometry contradiction: only a candidate, because the words
        # may describe another object.  The overlay is the authority.
        combined = f"{region.get('reason') or ''}\n{text or ''}"
        cy, cx = _centre(values)
        for word, side, _short in LOCATION_RULES:
            if word not in combined:
                continue
            bad = (
                (side == "top" and cy > 500)
                or (side == "bottom" and cy < 500)
                or (side == "left" and cx > 500)
                or (side == "right" and cx < 500)
                or (side == "center"
                    and (abs(cy - 500) > 250 or abs(cx - 500) > 250))
            )
            if bad:
                flags.append("location_text_mismatch")
                details.append(
                    f"text mentions {word} but the bbox centre is "
                    f"[y={cy:.0f},x={cx:.0f}]")
                break
        if flags:
            found.append({
                "region_index": index,
                "bbox": values,
                "flags": flags,
                "detail": "；".join(details),
            })
    return found
