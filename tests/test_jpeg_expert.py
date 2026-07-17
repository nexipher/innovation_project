"""Unit tests for JPEG Expert."""

import numpy as np
from experts.jpeg import JPEGExpert


class TestJPEGExpert:
    def test_output_range(self, jpeg_expert, synthetic_smooth_image):
        result = jpeg_expert.analyze(synthetic_smooth_image)
        assert 0.0 <= result.strength <= 1.0

    def test_smooth_image(self, jpeg_expert, synthetic_smooth_image):
        """Smooth uniform image — no block artifacts."""
        result = jpeg_expert.analyze(synthetic_smooth_image)
        # Very flat image: blockiness should be near 1.0 (neutral)
        # DCT anomaly may be low since no compression
        assert 0.0 <= result.strength <= 1.0

    def test_result_schema(self, jpeg_expert, synthetic_smooth_image):
        result = jpeg_expert.analyze(synthetic_smooth_image)
        assert result.source == "jpeg_expert"
        assert result.support in ("Real", "AI-generated", "Uncertain")
        assert result.evidence_name

    def test_small_patch(self, jpeg_expert):
        """Tiny patch (smaller than 8x8 block) should not crash."""
        tiny = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result = jpeg_expert.analyze(tiny)
        assert 0.0 <= result.strength <= 1.0
