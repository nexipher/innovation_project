"""
Error Level Analysis (ELA) candidate expert — G2 evaluation tool.

ELA re-saves the image (JPEG at a fixed quality), takes the per-pixel
difference against the original, and uses the amplified difference as a
forensic signal: regions that were previously JPEG-compressed respond
differently from regions that were not, and synthetic/edited regions often
show anomalous error levels.

This module provides:
  - `render_ela(image, quality)` — the classical ELA visualization;
  - `ELAExpert` — a BaseExpert-compatible wrapper whose raw metric is the
    95th-percentile ELA response (robust to single-pixel outliers), so G2
    can measure its discrimination before deciding whether to adopt it.
"""

import cv2
import numpy as np

from .base import BaseExpert, ExpertResult
from config import STRENGTH_THRESHOLD_LOW, STRENGTH_THRESHOLD_HIGH, STRENGTH_TEXT_MAP


def render_ela(image: np.ndarray, quality: int = 90) -> np.ndarray:
    """Return the ELA visualization as a BGR uint8 image."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    ok, buffer = cv2.imencode(".jpg", image, encode_params)
    if not ok:
        return np.zeros_like(image)

    recompressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if recompressed is None:
        return np.zeros_like(image)

    difference = cv2.absdiff(image, recompressed).astype(np.float32)
    # Amplify so the error level is visible; 20x matches common ELA practice.
    amplified = np.clip(difference * 20.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(cv2.cvtColor(amplified, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_INFERNO)


class ELAExpert(BaseExpert):
    """Candidate expert wrapping the ELA response."""

    source_name = "ela_expert"

    counter_explanation = (
        "ELA 对压缩历史敏感：所有图像经过统一重压缩后分离度降至随机水平（G2 q70 校准）。"
        "其高分主要指示‘存在压缩历史差异’，不等于生成模型伪迹，仅适用于未压缩或高质量来源。"
    )

    def __init__(self, quality: int = 90, sigmoid_midpoint: float = 18.0,
                 sigmoid_steepness: float = 0.08):
        self.quality = quality
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

    def analyze(self, img_patch: np.ndarray) -> ExpertResult:
        raw = self._ela_energy(img_patch)
        strength = 1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (raw - self.sigmoid_midpoint)))

        if strength < STRENGTH_THRESHOLD_LOW:
            support, interp = "Real", STRENGTH_TEXT_MAP["low"]
            direction = "Normal error-level profile; no compressed-region discontinuity detected."
        elif strength < STRENGTH_THRESHOLD_HIGH:
            support, interp = "Uncertain", STRENGTH_TEXT_MAP["medium"]
            direction = "Mild error-level spread; could reflect ordinary re-saving or texture complexity."
        else:
            support, interp = "AI-generated", STRENGTH_TEXT_MAP["high"]
            direction = "Strong error-level discontinuity consistent with localised re-encoding or synthesis."

        return self._build_result(
            evidence_name="ela_error_level",
            phenomenon=(
                f"Error-level analysis (q={self.quality}) shows a 95th-percentile response of "
                f"{raw:.2f}. {direction}"
            ),
            reasoning=(
                "Error Level Analysis re-encodes the image and measures the residual: regions "
                "that were compressed at a different quality than their surroundings respond "
                "anomalously. It is sensitive to re-saving history rather than to sensor physics."
            ),
            strength=float(strength),
            support=support,
            interpretation_text=interp,
            raw_metric=float(raw),
        )

    def _ela_energy(self, img_patch: np.ndarray) -> float:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.quality)]
        ok, buffer = cv2.imencode(".jpg", img_patch, encode_params)
        if not ok:
            return 0.0
        recompressed = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if recompressed is None:
            return 0.0
        difference = cv2.absdiff(img_patch, recompressed).astype(np.float32)
        return float(np.percentile(difference.mean(axis=2), 95))

    def render_artifacts(self, img_patch: np.ndarray) -> dict:
        return {"ela_map": render_ela(img_patch, self.quality)}
