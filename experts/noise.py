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
)


class NoiseExpert(BaseExpert):
    source_name = "noise_expert"

    # G2-b (§4.9 G2-e): on the format-balanced calibration set a HIGH residual
    # level means Real (camera sensor micro-noise survives re-encoding), and a
    # LOW level means Fake (generators produce smooth, low-variance output).
    # The original "inconsistency ⇒ manipulation" reading is inverted for this
    # task, so the token must say so.
    metric_polarity = -1

    inverted_text_map = {
        "low": ("Low residual micro-noise level — smooth, low-variance output of the "
                "kind the calibration set associates with generated imagery."),
        "medium": ("Intermediate residual level; the calibrated likelihood is close to "
                   "chance for this band."),
        "high": ("High residual micro-noise level — spatially rich sensor noise of the "
                 "kind the calibration set associates with camera capture."),
    }

    counter_explanation = (
        "该指标衡量残差水平而非\"不一致性\"：G2 校准（格式配平，5 格）显示 Real 中位数 3.6 对 Fake 2.0，"
        "方向与原始\"异常即伪造\"的解释相反，且跨格式稳定（AUROC 0.15-0.23）。"
        "降噪、锐化与重压缩同样会改变该水平，故只应作为弱证据与校准似然配合使用。"
    )

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

        support, interp_text = self.classify_metric(strength)

        return self._build_result(
            evidence_name="noise_residual_inconsistency",
            phenomenon=(
                # G2-e inversion: a high residual is the camera end of this
                # metric, a low one the generative end, so the description
                # follows the measured direction rather than the old reading.
                f"Localised noise variance at the level the calibration set associates with "
                f"{'camera capture (spatially rich sensor noise)' if strength > 0.5 else 'generated imagery (variance collapse)'} "
                f"(inconsistency ratio: {inconsistency:.4f}). "
                f"{'Residual consistent with a camera sensor' if strength > 0.5 else 'Variance collapse detected'}."
            ),
            reasoning=self._get_reasoning(strength, inconsistency),
            strength=strength,
            support=support,
            interpretation_text=interp_text,
            raw_metric=inconsistency,
        )


    # ------------------------------------------------------------------
    # Visual artifacts (G2 §4.9)
    # ------------------------------------------------------------------

    def render_artifacts(self, img_patch: np.ndarray) -> dict:
        """Render the SRM residual magnitude map (amplified 4x)."""
        residuals = self._apply_srm(img_patch)
        magnitude = np.abs(residuals).mean(axis=2) * 4.0
        normalized = np.clip(magnitude, 0, 255).astype(np.uint8)
        return {"noise_residual_map": cv2.applyColorMap(normalized, cv2.COLORMAP_JET)}

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
    def _get_reasoning(strength: float, inconsistency: float) -> str:
        if strength < 0.3:
            return (
                f"Low residual micro-noise level (metric={inconsistency:.4f}). "
                "The region is smooth at the sensor-noise scale — the condition "
                "under which the G2 calibration set identifies generated imagery "
                "(fake median 2.0 vs real median 3.6). Denoising or heavy "
                "recompression produces the same signature, so this is a weak "
                "signal on its own."
            )
        elif strength < 0.7:
            return (
                f"Intermediate residual micro-noise level (metric={inconsistency:.4f}). "
                "This band sits near the calibrated chance level; the measurement "
                "carries little directional information."
            )
        else:
            return (
                f"High residual micro-noise level (metric={inconsistency:.4f}). "
                "Camera sensors leave spatially rich micro-noise (shot noise + PRNU) "
                "that survives re-encoding, which is the condition the G2 calibration "
                "set associates with real capture. Note this is the opposite of the "
                "original manipulation reading: high residual does NOT indicate "
                "forgery. Sharpening can also inflate this metric."
            )

    @staticmethod
    def _classify(strength: float) -> tuple:
        """Deprecated alias of BaseExpert.classify_metric (kept for callers
        that imported it directly); it now honours the measured polarity."""
        return BaseExpert.classify_metric(strength)
