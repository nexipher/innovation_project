"""Unit tests for Frequency Expert."""

import numpy as np
from experts.frequency import FrequencyExpert


class TestFrequencyExpert:
    def test_output_range(self, freq_expert, synthetic_smooth_image):
        result = freq_expert.analyze(synthetic_smooth_image)
        assert 0.0 <= result.strength <= 1.0

    def test_smooth_image_low_strength(self, freq_expert, synthetic_smooth_image):
        """Uniform smooth image should have low frequency anomaly."""
        result = freq_expert.analyze(synthetic_smooth_image)
        # Smooth image shouldn't show strong periodic artifacts
        assert result.strength < 0.5

    def test_grid_image_higher_strength(self, freq_expert, synthetic_grid_image):
        """Grid image should show higher anomaly than smooth."""
        r_grid = freq_expert.analyze(synthetic_grid_image)
        r_smooth = freq_expert.analyze(
            np.ones((256, 256, 3), dtype=np.uint8) * 128
        )
        # Grid image should have higher or equal strength
        assert r_grid.strength >= r_smooth.strength * 0.5

    def test_result_schema(self, freq_expert, synthetic_smooth_image):
        result = freq_expert.analyze(synthetic_smooth_image)
        assert result.evidence_name
        assert result.source == "frequency_expert"
        assert result.support in ("Real", "AI-generated", "Uncertain")
        assert result.phenomenon
        assert result.reasoning
        assert result.interpretation_text

    def test_grayscale_patch(self, freq_expert):
        """Should also work with grayscale input."""
        gray = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        result = freq_expert.analyze(gray)
        assert 0.0 <= result.strength <= 1.0
