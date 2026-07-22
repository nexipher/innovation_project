"""
Frequency-Domain Forensic Expert.

Detects periodic grid artifacts in the 2D-FFT power spectrum that are
characteristic of GAN / Diffusion upsampling and deconvolution operations.

Algorithm (per the specification):
  1. Convert patch to grayscale.
  2. Apply Hanning window to reduce spectral leakage.
  3. Compute 2D-FFT → shift low frequencies to centre → log power spectrum.
  4. Radial average over the outer (high-frequency) annulus.
  5. Detect periodic peaks exceeding median + N*σ.
  6. Sigmoid-normalise the peak ratio to [0, 1].
"""

import numpy as np
from scipy.ndimage import map_coordinates

from .base import BaseExpert, ExpertResult
from config import (
    FREQ_HP_RADIUS_RATIO,
    FREQ_PEAK_SIGMA,
    FREQ_SIGMOID_MIDPOINT,
    FREQ_SIGMOID_STEEPNESS,
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class FrequencyExpert(BaseExpert):
    source_name = "frequency_expert"

    def __init__(
        self,
        hp_radius_ratio: float = FREQ_HP_RADIUS_RATIO,
        peak_sigma: float = FREQ_PEAK_SIGMA,
        sigmoid_midpoint: float = FREQ_SIGMOID_MIDPOINT,
        sigmoid_steepness: float = FREQ_SIGMOID_STEEPNESS,
    ):
        self.hp_radius_ratio = hp_radius_ratio
        self.peak_sigma = peak_sigma
        self.sigmoid_midpoint = sigmoid_midpoint
        self.sigmoid_steepness = sigmoid_steepness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, img_patch: np.ndarray) -> ExpertResult:
        """
        Args:
            img_patch: BGR uint8 numpy array (H, W, 3) — pre-cropped region.

        Returns:
            ExpertResult with frequency-domain anomaly assessment.
        """
        # 1. Grayscale
        if img_patch.ndim == 3:
            # BGR → Gray using standard luminance weights
            gray = (
                0.114 * img_patch[:, :, 0].astype(np.float64)
                + 0.587 * img_patch[:, :, 1].astype(np.float64)
                + 0.299 * img_patch[:, :, 2].astype(np.float64)
            )
        else:
            gray = img_patch.astype(np.float64)

        h, w = gray.shape

        # 2. Hanning window
        hanning_y = np.hanning(h)
        hanning_x = np.hanning(w)
        window = np.outer(hanning_y, hanning_x)
        windowed = gray * window

        # 3. 2D-FFT → shift → log power spectrum
        fft = np.fft.fft2(windowed)
        fft_shifted = np.fft.fftshift(fft)
        power = np.log(np.abs(fft_shifted) + 1.0)

        # 4. High-frequency radial average
        hf_energy, hf_mask = self._high_freq_radial(power)

        # 5. Peak detection in high-frequency region
        peak_ratio = self._detect_periodic_peaks(power, hf_mask)

        # 6. Sigmoid normalisation
        strength = self._sigmoid_normalise(peak_ratio)

        # Determine support label and interpretation text
        support, interp_text = self._classify(strength)

        return self._build_result(
            evidence_name="abnormal_high_frequency_residual",
            phenomenon=(
                f"The spatial-frequency spectrum shows "
                f"{'concentrated periodic grids' if strength > 0.5 else 'no significant periodic peaks'}. "
                f"(peak ratio: {peak_ratio:.4f})"
            ),
            reasoning=self._get_reasoning(strength, peak_ratio),
            strength=strength,
            support=support,
            interpretation_text=interp_text,
            raw_metric=peak_ratio,
        )

    # ------------------------------------------------------------------
    # Conditional reasoning
    # ------------------------------------------------------------------

    @staticmethod
    def _get_reasoning(strength: float, peak_ratio: float) -> str:
        if strength < 0.3:
            return (
                "No significant periodic peaks detected in the high-frequency "
                "power spectrum. The spectral pattern is consistent with natural "
                "image content — real camera-captured images do not exhibit the "
                "structured high-frequency periodicity characteristic of GAN or "
                "Diffusion upsampling artifacts."
            )
        elif strength < 0.7:
            return (
                f"Weak periodic energy concentration detected (peak ratio={peak_ratio:.4f}), "
                "but below the confident detection threshold. This could indicate "
                "mild post-processing or natural image texture rather than definitive "
                "AI generation. Additional expert analysis is recommended."
            )
        else:
            return (
                f"Strong periodic peaks detected in the high-frequency power spectrum "
                f"(peak ratio={peak_ratio:.4f}). This spectral pattern is mathematically "
                "consistent with upsampling / deconvolution grid artifacts common in "
                "GAN and Diffusion-based image synthesis. Real camera-captured images "
                "rarely exhibit such structured high-frequency periodicity."
            )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _high_freq_radial(self, power: np.ndarray) -> tuple:
        """
        Extract the high-frequency annulus of the power spectrum.

        Returns:
            (hf_values, hf_mask) — flattened high-freq values and a boolean mask.
        """
        h, w = power.shape
        cy, cx = h / 2.0, w / 2.0
        max_radius = min(cy, cx)

        inner_radius = max_radius * self.hp_radius_ratio

        # Build coordinate grids
        y_coords, x_coords = np.indices((h, w))
        radii = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

        mask = (radii >= inner_radius) & (radii <= max_radius)
        values = power[mask]

        return values, mask

    def _detect_periodic_peaks(self, power: np.ndarray, mask: np.ndarray) -> float:
        """
        Detect periodic peaks in the high-frequency region.

        A 'peak' is a local maximum whose value exceeds median + N*σ of the
        high-frequency region.

        Returns:
            peak_ratio — fraction of high-freq pixels that qualify as peaks.
            Higher values suggest stronger periodic artifacts.
        """
        hf_values = power[mask]
        if len(hf_values) == 0:
            return 0.0

        median_val = np.median(hf_values)
        std_val = np.std(hf_values)
        threshold = median_val + self.peak_sigma * std_val

        # Count local maxima above threshold
        # Use scipy-style local max detection: compare each pixel to its 8 neighbours
        from scipy.ndimage import maximum_filter

        # Build a masked version: zero out non-HF regions so they don't contribute
        masked_power = power.copy()
        masked_power[~mask] = 0.0

        # Local maxima (8-connected)
        local_max = maximum_filter(masked_power, size=3) == masked_power
        local_max[~mask] = False  # only consider HF region

        # Peaks are local maxima above threshold
        peak_mask = local_max & (masked_power > threshold)
        peak_count = np.sum(peak_mask)
        total_hf = np.sum(mask)

        return peak_count / max(total_hf, 1)

    def _sigmoid_normalise(self, x: float) -> float:
        """Map raw metric → [0, 1] via sigmoid."""
        return float(1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (x - self.sigmoid_midpoint))))

    @staticmethod
    def _classify(strength: float) -> tuple:
        """Map strength to support label and interpretation text."""
        if strength < STRENGTH_THRESHOLD_LOW:
            return "Real", STRENGTH_TEXT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return "Uncertain", STRENGTH_TEXT_MAP["medium"]
        else:
            return "AI-generated", STRENGTH_TEXT_MAP["high"]
