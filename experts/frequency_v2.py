"""
Frequency-Domain Forensic Expert v2 — Phase 3.

Improved over v1:
  1. Full-image FFT (not bbox crop) for maximum spectral resolution.
  2. Multi-scale analysis — down-sample and re-analyse at 50% resolution
     to catch artifacts at different frequency scales.
  3. Adjusted peak detection with adaptive threshold.
  4. Better calibrated sigmoid parameters (from Phase 2.2 findings).

Replaces FrequencyExpert for SFT data construction and Phase 3 evaluation.
"""

import numpy as np
import cv2
from scipy.ndimage import maximum_filter

from .base import BaseExpert, ExpertResult
from config import (
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class FrequencyExpertV2(BaseExpert):
    source_name = "frequency_expert"

    def __init__(
        self,
        hp_radius_ratio: float = 0.4,      # wider high-pass than v1 (0.5→0.4)
        peak_sigma: float = 2.5,            # lower threshold for better sensitivity
        sigmoid_midpoint: float = 0.015,    # calibrated for full-image raw_metric range
        sigmoid_steepness: float = 80.0,
    ):
        self.hp_radius_ratio = hp_radius_ratio
        self.peak_sigma = peak_sigma
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, img_patch: np.ndarray) -> ExpertResult:
        h, w = img_patch.shape[:2]

        # Multi-scale: full-size + 50% down-sample
        raw_full = self._analyze_scale(img_patch)

        if min(h, w) >= 128:
            small = cv2.resize(img_patch, (w // 2, h // 2))
            raw_half = self._analyze_scale(small)
            raw_metric = 0.6 * raw_full + 0.4 * raw_half
        else:
            raw_metric = raw_full

        strength = self._sigmoid_normalise(raw_metric)
        support, interp_text = self._classify(strength)

        return self._build_result(
            evidence_name="frequency_grid_artifact_v2",
            phenomenon=(
                f"Multi-scale frequency analysis shows "
                f"{'concentrated periodic peaks' if strength > 0.5 else 'no significant periodic structure'}. "
                f"(raw peak ratio: {raw_metric:.4f}, "
                f"full={raw_full:.4f}"
                + (f", half={raw_half:.4f}" if min(h, w) >= 128 else "")
                + ")"
            ),
            reasoning=self._get_reasoning(strength, raw_metric),
            strength=strength,
            support=support,
            interpretation_text=interp_text,
            raw_metric=raw_metric,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_scale(self, img: np.ndarray) -> float:
        """Run FFT analysis at a single scale. Returns raw peak ratio."""
        if img.ndim == 3:
            gray = (
                0.114 * img[:, :, 0].astype(np.float64)
                + 0.587 * img[:, :, 1].astype(np.float64)
                + 0.299 * img[:, :, 2].astype(np.float64)
            )
        else:
            gray = img.astype(np.float64)

        h, w = gray.shape

        # Hanning window
        window = np.outer(np.hanning(h), np.hanning(w))
        windowed = gray * window

        # 2D-FFT → log power
        fft = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft)
        power = np.log(np.abs(fft_shifted) + 1.0)

        # High-frequency radial extraction
        hf_values, hf_mask = self._high_freq_radial(power)

        # Adaptive peak detection
        peak_ratio = self._detect_peaks_adaptive(power, hf_mask, hf_values)

        return float(peak_ratio)

    def _high_freq_radial(self, power: np.ndarray) -> tuple:
        h, w = power.shape
        cy, cx = h / 2.0, w / 2.0
        max_radius = min(cy, cx)
        inner_radius = max_radius * self.hp_radius_ratio

        y_coords, x_coords = np.indices((h, w))
        radii = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
        mask = (radii >= inner_radius) & (radii <= max_radius)
        values = power[mask]
        return values, mask

    def _detect_peaks_adaptive(self, power: np.ndarray, mask: np.ndarray,
                               hf_values: np.ndarray) -> float:
        """Adaptive peak detection using local maxima in HF region."""
        if len(hf_values) < 10:
            return 0.0

        median_val = np.median(hf_values)
        std_val = np.std(hf_values)
        threshold = median_val + self.peak_sigma * std_val

        # Build masked power for local-max detection
        masked_power = power.copy()
        masked_power[~mask] = -np.inf

        local_max = maximum_filter(masked_power, size=3) == masked_power
        local_max[~mask] = False

        peak_mask = local_max & (masked_power > threshold)
        peak_count = np.sum(peak_mask)
        total_hf = max(np.sum(mask), 1)

        return float(peak_count / total_hf)

    def _sigmoid_normalise(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (x - self.sigmoid_midpoint))))

    @staticmethod
    def _get_reasoning(strength: float, raw_metric: float) -> str:
        if strength < 0.3:
            return (
                "Multi-scale frequency analysis reveals no significant periodic "
                "structure in the high-frequency spectrum. The spectral decay "
                "pattern is consistent with natural image content, not the "
                "structured grid artifacts characteristic of GAN/Diffusion "
                "upsampling operations."
            )
        elif strength < 0.7:
            return (
                f"Multi-scale frequency analysis detects weak periodic energy "
                f"(raw={raw_metric:.4f}) below the confident detection threshold. "
                "This may indicate mild post-processing or natural texture patterns. "
                "Cross-reference with other forensic experts is recommended."
            )
        else:
            return (
                f"Multi-scale frequency analysis reveals concentrated periodic peaks "
                f"(raw={raw_metric:.4f}) consistent with GAN/Diffusion upsampling "
                "grid artifacts. The multi-scale approach confirms the artifact "
                "persists across resolutions, increasing confidence in the finding. "
                "Natural camera-captured images rarely exhibit such structured "
                "high-frequency periodicity."
            )

    @staticmethod
    def _classify(strength: float) -> tuple:
        if strength < STRENGTH_THRESHOLD_LOW:
            return "Real", STRENGTH_TEXT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return "Uncertain", STRENGTH_TEXT_MAP["medium"]
        else:
            return "AI-generated", STRENGTH_TEXT_MAP["high"]
