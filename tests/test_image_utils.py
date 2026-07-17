"""Unit tests for image utilities."""

import numpy as np
import pytest
from utils.image_utils import ImageUtils


class TestCropBbox:
    def test_center_crop(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = ImageUtils.crop_bbox(img, [25, 25, 75, 75])
        assert crop.shape == (50, 50, 3)

    def test_out_of_bounds_clamped(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = ImageUtils.crop_bbox(img, [-10, -10, 200, 200])
        assert crop.shape == (100, 100, 3)

    def test_zero_area_bbox(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        crop = ImageUtils.crop_bbox(img, [50, 50, 50, 50])
        assert crop.size == 0


class TestGrayscale:
    def test_bgr_to_gray(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        gray = ImageUtils.to_grayscale(img)
        assert gray.ndim == 2
        assert gray.shape == (64, 64)

    def test_already_gray(self):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        gray = ImageUtils.to_grayscale(img)
        assert gray is img  # returned as-is


class TestDimensions:
    def test_get_dims(self):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = ImageUtils.get_dimensions(img)
        assert h == 480
        assert w == 640


class TestNormalizeDtype:
    def test_uint8_passthrough(self):
        img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        result = ImageUtils.normalize_dtype(img)
        assert result.dtype == np.uint8

    def test_float_scale(self):
        img = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
        result = ImageUtils.normalize_dtype(img)
        assert result.dtype == np.uint8
        assert result.max() == 255
