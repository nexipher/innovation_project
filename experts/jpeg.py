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

import numpy as np
from scipy.fft import dct

from .base import BaseExpert, ExpertResult
from config import (
    JPEG_BLOCK_SIZE,
    JPEG_SIGMOID_MIDPOINT,
    JPEG_SIGMOID_STEEPNESS,
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class JPEGExpert(BaseExpert):
    source_name = "jpeg_expert"

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

        support, interp_text = self._classify(strength)

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
            reasoning=(
                "Double JPEG compression and re-saving leave distinct forensic "
                "traces: (1) anomalous gradient energy at 8×8 block boundaries "
                "(blockiness metric), and (2) periodic gaps in the DCT coefficient "
                "histogram caused by re-quantisation with different quality factors. "
                "These are classic digital forgery markers."
            ),
            strength=strength,
            support=support,
            interpretation_text=interp_text,
            raw_metric=combined,
            blockiness=blockiness,
            dct_anomaly=dct_anomaly,
        )

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
    def _classify(strength: float) -> tuple:
        if strength < STRENGTH_THRESHOLD_LOW:
            return "Real", STRENGTH_TEXT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return "Uncertain", STRENGTH_TEXT_MAP["medium"]
        else:
            return "AI-generated", STRENGTH_TEXT_MAP["high"]
