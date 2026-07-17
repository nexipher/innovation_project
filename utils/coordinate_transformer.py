"""
Coordinate transformer for converting between MLLM relative coordinates [0, 1000]
and OpenCV absolute pixel coordinates.

Qwen2.5-VL convention: bbox = [ymin, xmin, ymax, xmax] in range [0, 1000].
Our internal convention matches OpenCV / NumPy slicing: img[ymin:ymax, xmin:xmax].
"""

from typing import List

from config import NORMALIZATION_SCALE, BBOX_MIN_SIZE


class CoordinateTransformer:
    """Stateless coordinate conversion utilities."""

    @staticmethod
    def relative_to_absolute(rel_bbox: List[int], width: int, height: int) -> List[int]:
        """
        Convert a normalised bbox [ymin, xmin, ymax, xmax] in [0, NORMALIZATION_SCALE]
        to absolute pixel coordinates.

        Args:
            rel_bbox: [ymin, xmin, ymax, xmax] each in [0, 1000].
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            [ymin_px, xmin_px, ymax_px, xmax_px] as integers.
        """
        ymin, xmin, ymax, xmax = rel_bbox
        scale_x = width / NORMALIZATION_SCALE
        scale_y = height / NORMALIZATION_SCALE

        abs_ymin = int(round(ymin * scale_y))
        abs_xmin = int(round(xmin * scale_x))
        abs_ymax = int(round(ymax * scale_y))
        abs_xmax = int(round(xmax * scale_x))

        return [abs_ymin, abs_xmin, abs_ymax, abs_xmax]

    @staticmethod
    def absolute_to_relative(abs_bbox: List[int], width: int, height: int) -> List[int]:
        """
        Convert absolute pixel coordinates back to normalised [0, NORMALIZATION_SCALE].

        Args:
            abs_bbox: [ymin, xmin, ymax, xmax] in pixel coordinates.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            [ymin, xmin, ymax, xmax] each in [0, 1000].
        """
        ymin, xmin, ymax, xmax = abs_bbox
        scale_x = NORMALIZATION_SCALE / width
        scale_y = NORMALIZATION_SCALE / height

        rel_ymin = int(round(ymin * scale_y))
        rel_xmin = int(round(xmin * scale_x))
        rel_ymax = int(round(ymax * scale_y))
        rel_xmax = int(round(xmax * scale_x))

        # Clamp to [0, NORMALIZATION_SCALE]
        rel_ymin = max(0, min(NORMALIZATION_SCALE, rel_ymin))
        rel_xmin = max(0, min(NORMALIZATION_SCALE, rel_xmin))
        rel_ymax = max(0, min(NORMALIZATION_SCALE, rel_ymax))
        rel_xmax = max(0, min(NORMALIZATION_SCALE, rel_xmax))

        return [rel_ymin, rel_xmin, rel_ymax, rel_xmax]

    @staticmethod
    def clip_bbox(bbox: List[int], width: int, height: int) -> List[int]:
        """
        Clip a bbox to image bounds and enforce minimum size.

        Args:
            bbox: [ymin, xmin, ymax, xmax] in absolute pixel coordinates.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Clipped bbox guaranteed within [0, width] × [0, height]
            and at least BBOX_MIN_SIZE in each dimension.
        """
        ymin, xmin, ymax, xmax = bbox

        # Clamp to image bounds
        ymin = max(0, min(height, ymin))
        xmin = max(0, min(width, xmin))
        ymax = max(0, min(height, ymax))
        xmax = max(0, min(width, xmax))

        # Ensure ymin < ymax, xmin < xmax
        if ymin >= ymax:
            ymax = min(height, ymin + BBOX_MIN_SIZE)
        if xmin >= xmax:
            xmax = min(width, xmin + BBOX_MIN_SIZE)

        # Enforce minimum size
        if (ymax - ymin) < BBOX_MIN_SIZE:
            ymax = min(height, ymin + BBOX_MIN_SIZE)
            ymin = max(0, ymax - BBOX_MIN_SIZE)
        if (xmax - xmin) < BBOX_MIN_SIZE:
            xmax = min(width, xmin + BBOX_MIN_SIZE)
            xmin = max(0, xmax - BBOX_MIN_SIZE)

        return [int(ymin), int(xmin), int(ymax), int(xmax)]
