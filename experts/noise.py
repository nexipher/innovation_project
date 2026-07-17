"""
Noise Residual Consistency Forensic Expert.

Detects local noise inconsistency caused by splicing, inpainting, or
AI-based local editing.  Real camera sensors produce spatially consistent
micro-scale noise (PRNU / shot noise), while tampered regions show
anomalous variance collapse or inflation.

Algorithm (per the specification):
  1. Apply SRM (Spatial Rich Model) high-pass filter kernel to extract
     noise residuals per channel.
  2. Compute local noise variance in sliding windows over the patch.
  3. Compare local variance to global background variance of the patch.
  4. Normalise the inconsistency score via sigmoid.
"""

import numpy as np
import cv2

from .base import BaseExpert, ExpertResult
from config import (
    NOISE_SRM_KERNEL_ID,
    NOISE_WINDOW_SIZE,
    NOISE_SIGMOID_MIDPOINT,
    NOISE_SIGMOID_STEEPNESS,
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class NoiseExpert(BaseExpert):
    source_name = "noise_expert"

    # ------------------------------------------------------------------
    # SRM filter kernels (Spatial Rich Model for steganalysis)
    # Kernel #1: 5×5 high-pass — the classic "noise residual" kernel
    # ------------------------------------------------------------------
    SRM_KERNELS = {
        1: np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0],
        ], dtype=np.float64) / 4.0,

        2: np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1],
        ], dtype=np.float64) / 12.0,
    }

    def __init__(
        self,
        kernel_id: int = NOISE_SRM_KERNEL_ID,
        window_size: int = NOISE_WINDOW_SIZE,
        sigmoid_midpoint: float = NOISE_SIGMOID_MIDPOINT,
        sigmoid_steepness: float = NOISE_SIGMOID_STEEPNESS,
    ):
        self.kernel = self.SRM_KERNELS.get(kernel_id, self.SRM_KERNELS[1])
        self.window_size = window_size
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
            ExpertResult with noise inconsistency assessment.
        """
        h, w = img_patch.shape[:2]

        # 1. Extract noise residuals via SRM filtering (per-channel)
        residuals = self._apply_srm(img_patch)  # (H, W, 3) float

        # 2. Compute local noise variance map
        local_var_map = self._local_variance_map(residuals)  # (H, W)

        # 3. Global background variance
        global_var = float(np.var(residuals))

        # 4. Inconsistency score: how much local variance deviates from global
        inconsistency = self._compute_inconsistency(local_var_map, global_var)

        # 5. Sigmoid normalisation
        strength = self._sigmoid_normalise(inconsistency)

        support, interp_text = self._classify(strength)

        return self._build_result(
            evidence_name="noise_residual_inconsistency",
            phenomenon=(
                f"Localised noise variance measures "
                f"{'abnormally' if strength > 0.5 else 'within normal range'} "
                f"(inconsistency ratio: {inconsistency:.4f}). "
                f"{'Variance collapse or inflation detected' if strength > 0.5 else 'No significant local variance anomaly'}."
            ),
            reasoning=(
                "Real camera sensors produce spatially homogeneous micro-noise "
                "(shot noise + PRNU). Splicing, inpainting, or AI-based local "
                "editing disrupts this homogeneity, causing either variance "
                "collapse (over-smoothing) or inflation (unnatural texture). "
                "The micro-noise pattern exhibits localised variance anomalies "
                "detected via SRM high-pass filtering."
            ),
            strength=strength,
            support=support,
            interpretation_text=interp_text,
            raw_metric=inconsistency,
        )

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _apply_srm(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the SRM high-pass filter to each channel.
        Returns float residuals of same spatial dimensions.
        """
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        channels = img.shape[2]
        residuals = np.zeros(img.shape[:2] + (channels,), dtype=np.float64)

        for c in range(channels):
            residuals[:, :, c] = cv2.filter2D(
                img[:, :, c].astype(np.float64), -1, self.kernel,
            )

        return residuals

    def _local_variance_map(self, residuals: np.ndarray) -> np.ndarray:
        """
        Compute local noise variance using a sliding window.
        Returns a 2D map of variance values.
        """
        h, w = residuals.shape[:2]
        ws = self.window_size
        var_map = np.zeros((h, w), dtype=np.float64)

        # Average variance across all channels
        for c in range(residuals.shape[2]):
            ch = residuals[:, :, c]
            # Compute local variance using integral-image-style approach
            # For efficiency, compute mean in sliding window then squared deviation
            kernel = np.ones((ws, ws), dtype=np.float64) / (ws * ws)

            local_mean = cv2.filter2D(ch, -1, kernel)
            local_sq_mean = cv2.filter2D(ch * ch, -1, kernel)
            local_var = local_sq_mean - local_mean * local_mean
            local_var = np.maximum(local_var, 0.0)  # numerical stability

            var_map += local_var

        var_map /= residuals.shape[2]
        return var_map

    def _compute_inconsistency(
        self, local_var_map: np.ndarray, global_var: float
    ) -> float:
        """
        Measure how much local variance deviates from the global background.

        Uses the 95th percentile of local/global variance ratio as the
        inconsistency metric — robust to outliers.

        Returns:
            inconsistency ratio ∈ [0, ∞), typically 0.5 ~ 5.0.
        """
        if global_var < 1e-8:
            # Near-zero global variance: very suspicious (over-smoothed patch)
            return 5.0

        ratio_map = local_var_map / (global_var + 1e-8)
        # Use high percentile to capture worst-case deviation
        p95 = float(np.percentile(np.abs(ratio_map - 1.0), 95))
        return p95

    def _sigmoid_normalise(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (x - self.sigmoid_midpoint))))

    @staticmethod
    def _classify(strength: float) -> tuple:
        if strength < STRENGTH_THRESHOLD_LOW:
            return "Real", STRENGTH_TEXT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return "Uncertain", STRENGTH_TEXT_MAP["medium"]
        else:
            return "AI-generated", STRENGTH_TEXT_MAP["high"]
