"""
JPEG Compression Artifact Forensic Expert.

Detects traces of double JPEG compression and abnormal blockiness
that indicate re-saving, splicing, or social-media re-compression.

Algorithm (per the specification):
  1. Compute blockiness metric: ratio of inter-block (8×8 boundary)
     gradient energy to intra-block gradient energy.  A double-compressed
     image has anomalous block-boundary patterns.
  2. Compute DCT coefficient histogram for each 8×8 block and check
     for "hollowed-out" bins characteristic of double quantization.
  3. Combine both metrics → sigmoid-normalised strength.
"""

import cv2
import numpy as np
from scipy.fft import dct

from .base import BaseExpert, ExpertResult
from config import (
    JPEG_BLOCK_SIZE,
    JPEG_SIGMOID_MIDPOINT,
    JPEG_SIGMOID_STEEPNESS,
)


class JPEGExpert(BaseExpert):
    source_name = "jpeg_expert"

    # G2-b (§4.9 G2-e): a HIGH level of blockiness/DCT structure means the
    # image carries JPEG-photo history — i.e. it leans Real, not forged.  The
    # signal collapses as the image is recompressed more (png 0.972 → q70
    # 0.569 separation), so a uniformly re-encoded image tells us almost
    # nothing.
    metric_polarity = -1

    inverted_text_map = {
        "low": ("Low JPEG structural level — the image carries little JPEG history, "
                "the condition the calibration set associates with natively generated "
                "(PNG) imagery."),
        "medium": ("Intermediate JPEG structural level; this band is near the "
                   "calibrated chance level, and re-compression weakens it further."),
        "high": ("High JPEG structural level — blockiness and DCT structure of the "
                 "kind the calibration set associates with a camera JPEG that was "
                 "saved or re-saved (real capture), not with forgery."),
    }

    counter_explanation = (
        "本指标衡量\"是否存在 JPEG 压缩历史\"而非伪造痕迹：G2 校准显示高值倾向 Real"
        "（PNG 格分离度 0.972、native 0.962，但重压缩至 q70 后降至 0.569），"
        "即它主要反映来源的压缩历史。统一重压缩后的图像上该证据近乎无效，不得作为独立判据。"
    )

    def __init__(
        self,
        block_size: int = JPEG_BLOCK_SIZE,
        sigmoid_midpoint: float = JPEG_SIGMOID_MIDPOINT,
        sigmoid_steepness: float = JPEG_SIGMOID_STEEPNESS,
    ):
        self.block_size = block_size
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
            ExpertResult with JPEG artifact assessment.
        """
        # Convert to grayscale for analysis
        if img_patch.ndim == 3:
            gray = (
                0.114 * img_patch[:, :, 0].astype(np.float64)
                + 0.587 * img_patch[:, :, 1].astype(np.float64)
                + 0.299 * img_patch[:, :, 2].astype(np.float64)
            )
        else:
            gray = img_patch.astype(np.float64)

        h, w = gray.shape

        # 1. Blockiness metric
        blockiness = self._compute_blockiness(gray)

        # 2. DCT histogram anomaly
        dct_anomaly = self._dct_histogram_anomaly(gray)

        # 3. Combined score (weighted average)
        combined = 0.6 * blockiness + 0.4 * dct_anomaly

        # 4. Sigmoid normalisation
        strength = self._sigmoid_normalise(combined)

        support, interp_text = self.classify_metric(strength)

        # Choose appropriate evidence name
        if blockiness > dct_anomaly:
            evidence_name = "abnormal_jpeg_blockiness"
            primary_phenom = "abnormal 8×8 block-boundary gradient ratios"
        else:
            evidence_name = "double_jpeg_quantization"
            primary_phenom = "DCT coefficient histogram hollowing"

        return self._build_result(
            evidence_name=evidence_name,
            phenomenon=(
                f"JPEG structural analysis reveals {primary_phenom}. "
                f"(blockiness: {blockiness:.4f}, DCT anomaly: {dct_anomaly:.4f})"
            ),
            reasoning=self._get_reasoning(strength, blockiness, dct_anomaly),
            strength=strength,
            support=support,
            interpretation_text=interp_text,
            raw_metric=combined,
            blockiness=blockiness,
            dct_anomaly=dct_anomaly,
        )


    # ------------------------------------------------------------------
    # Visual artifacts (G2 §4.9)
    # ------------------------------------------------------------------

    def render_artifacts(self, img_patch: np.ndarray) -> dict:
        """Render a block-boundary gradient map: energy concentrated on the
        8x8 grid is the visual signature of JPEG blockiness."""
        if img_patch.ndim == 3:
            gray = (
                0.114 * img_patch[:, :, 0].astype(np.float64)
                + 0.587 * img_patch[:, :, 1].astype(np.float64)
                + 0.299 * img_patch[:, :, 2].astype(np.float64)
            )
        else:
            gray = img_patch.astype(np.float64)

        h, w = gray.shape
        block = self.block_size
        boundary_map = np.zeros((h, w), dtype=np.float64)

        horizontal = np.abs(np.diff(gray, axis=1))  # (h, w-1)
        for column in range(block - 1, w - 1, block):
            boundary_map[:, column + 1] = horizontal[:, column]
        vertical = np.abs(np.diff(gray, axis=0))    # (h-1, w)
        for row in range(block - 1, h - 1, block):
            boundary_map[row + 1, :] = vertical[row, :]

        normalized = cv2.normalize(boundary_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return {"jpeg_blockiness_map": cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)}

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _compute_blockiness(self, gray: np.ndarray) -> float:
        """
        Compute the blockiness metric: ratio of inter-block gradient energy
        to intra-block gradient energy along 8×8 JPEG grid boundaries.

        Higher values suggest stronger block artifacts (double JPEG or
        heavy compression).

        Returns:
            Blockiness ratio ∈ [0, ∞), typically 0.8 ~ 3.0.
        """
        h, w = gray.shape
        B = self.block_size

        if h < B or w < B:
            return 0.0

        # Horizontal block boundaries (every 8th row)
        h_boundary_diff = 0.0
        h_boundary_count = 0
        for r in range(B - 1, h, B):
            if r + 1 < h:
                h_boundary_diff += np.sum(np.abs(gray[r + 1, :] - gray[r, :]))
                h_boundary_count += w

        # Horizontal intra-block differences (rows NOT at block boundaries)
        h_intra_diff = 0.0
        h_intra_count = 0
        for r in range(h - 1):
            if (r + 1) % B != 0:  # not a block boundary
                h_intra_diff += np.sum(np.abs(gray[r + 1, :] - gray[r, :]))
                h_intra_count += w

        # Vertical block boundaries (every 8th column)
        v_boundary_diff = 0.0
        v_boundary_count = 0
        for c in range(B - 1, w, B):
            if c + 1 < w:
                v_boundary_diff += np.sum(np.abs(gray[:, c + 1] - gray[:, c]))
                v_boundary_count += h

        # Vertical intra-block differences
        v_intra_diff = 0.0
        v_intra_count = 0
        for c in range(w - 1):
            if (c + 1) % B != 0:
                v_intra_diff += np.sum(np.abs(gray[:, c + 1] - gray[:, c]))
                v_intra_count += h

        # Blockiness = mean(|boundary diff|) / mean(|intra diff|)
        boundary_mean = (h_boundary_diff + v_boundary_diff) / max(h_boundary_count + v_boundary_count, 1)
        intra_mean = (h_intra_diff + v_intra_diff) / max(h_intra_count + v_intra_count, 1)

        if intra_mean < 1e-8:
            return 1.0  # flat image — neutral

        return float(boundary_mean / intra_mean)

    def _dct_histogram_anomaly(self, gray: np.ndarray) -> float:
        """
        Check for double-quantisation artifacts in the DCT coefficient histogram.

        Double JPEG compression causes periodic "holes" in the histogram of
        DCT coefficients because of the mismatch between two quantisation tables.

        Returns:
            Anomaly score ∈ [0, ∞), higher = more suspicious.
        """
        h, w = gray.shape
        B = self.block_size

        # Pad to multiple of block size
        h_pad = (h // B) * B
        w_pad = (w // B) * B
        if h_pad < B or w_pad < B:
            return 0.0

        gray_crop = gray[:h_pad, :w_pad]

        # Reshape into blocks
        blocks = gray_crop.reshape(h_pad // B, B, w_pad // B, B)
        blocks = blocks.transpose(0, 2, 1, 3)  # (n_rows, n_cols, B, B)
        blocks = blocks.reshape(-1, B, B)      # (n_blocks, B, B)

        # DCT each block, collect AC coefficients
        n_blocks = blocks.shape[0]
        if n_blocks == 0:
            return 0.0

        # Sample a subset if too many blocks (efficiency)
        max_blocks = 500
        if n_blocks > max_blocks:
            indices = np.linspace(0, n_blocks - 1, max_blocks, dtype=int)
            blocks = blocks[indices]

        all_dct_coeffs = []
        for blk in blocks:
            dct_coeffs = dct(dct(blk, axis=0, norm='ortho'), axis=1, norm='ortho')
            # Collect all coefficients except DC (position 0,0)
            ac = dct_coeffs.flatten()[1:]
            all_dct_coeffs.append(ac)

        all_coeffs = np.concatenate(all_dct_coeffs)

        # Build histogram and check for periodic gaps
        # Focus on small-magnitude coefficients where double-quantisation is most visible
        clipped = np.clip(all_coeffs, -5, 5)
        hist, _ = np.histogram(clipped, bins=50, range=(-5, 5))

        # Anomaly metric: coefficient of variation of adjacent histogram bins.
        # Double JPEG creates alternating full/empty bins → high CV.
        hist = hist.astype(np.float64) + 1e-8  # avoid div by zero
        adjacent_ratios = hist[1:] / hist[:-1]
        # High variance in adjacent ratios suggests periodic hollowing
        cv = float(np.std(adjacent_ratios) / (np.mean(adjacent_ratios) + 1e-8))

        return cv

    def _sigmoid_normalise(self, x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-self.sigmoid_steepness * (x - self.sigmoid_midpoint))))

    @staticmethod
    def _get_reasoning(strength: float, blockiness: float, dct_anomaly: float) -> str:
        if strength < 0.3:
            return (
                f"Little JPEG structure found (blockiness={blockiness:.4f}, "
                f"DCT anomaly={dct_anomaly:.4f}). The image carries almost no "
                "JPEG compression history — the condition the G2 calibration set "
                "associates with natively generated (PNG) imagery. Note this is a "
                "provenance cue, not proof: a screenshot or a heavily processed "
                "image looks the same."
            )
        elif strength < 0.7:
            return (
                f"Moderate JPEG structure found (blockiness={blockiness:.4f}, "
                f"DCT anomaly={dct_anomaly:.4f}). At this level the calibrated "
                "likelihood is near chance; re-compression weakens the signal "
                "further, so this band should not drive a verdict."
            )
        else:
            return (
                f"Strong JPEG compression structure found (blockiness={blockiness:.4f}, "
                f"DCT anomaly={dct_anomaly:.4f}). Gradient energy at 8×8 block "
                "boundaries and DCT histogram structure indicate the image originates "
                "from a JPEG photograph that was saved or re-saved — the direction the "
                "G2 calibration set associates with real capture. This is NOT a forgery "
                "marker: high values do not indicate tampering, and uniform re-compression "
                "(quality ≤ 70) removes most of this signal's discriminative power."
            )

    @staticmethod
    def _classify(strength: float) -> tuple:
        """Deprecated alias of BaseExpert.classify_metric, which honours the
        measured metric polarity (G2-e)."""
        return BaseExpert.classify_metric(strength)
