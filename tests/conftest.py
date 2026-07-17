"""
Shared pytest fixtures for the forensic agent test suite.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_grid_image():
    """256x256 RGB image with synthetic periodic grid (simulates GAN artifacts)."""
    h, w = 256, 256
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    grid = 128 + 40 * np.sin(2 * np.pi * xx / 4.0) * np.sin(2 * np.pi * yy / 4.0)
    img = np.stack([
        np.clip(grid, 0, 255).astype(np.uint8),
        np.clip(grid, 0, 255).astype(np.uint8),
        np.clip(grid, 0, 255).astype(np.uint8),
    ], axis=-1)
    return img


@pytest.fixture
def synthetic_smooth_image():
    """256x256 RGB uniformly smooth image."""
    return np.ones((256, 256, 3), dtype=np.uint8) * 128


@pytest.fixture
def synthetic_noise_inconsistent():
    """256x256 RGB: left half has heavy noise, right half minimal noise."""
    h, w = 256, 256
    rng = np.random.default_rng(42)
    base = rng.integers(100, 156, (h, w, 3), dtype=np.uint8).astype(np.int16)
    # Heavy noise on left half
    noise_h = rng.integers(-30, 30, (h, w // 2, 3), dtype=np.int16)
    base[:, :w // 2, :] += noise_h
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture
def synthetic_noise_uniform():
    """256x256 RGB with uniform light noise (simulates camera sensor)."""
    rng = np.random.default_rng(42)
    base = np.ones((256, 256, 3), dtype=np.int16) * 128
    noise = rng.integers(-5, 5, (256, 256, 3), dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Mock MLLM fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mllm_fast():
    from mllm.mock_client import MockMLLMClient
    return MockMLLMClient(mode="fast_verdict", seed=42)


@pytest.fixture
def mock_mllm_default():
    from mllm.mock_client import MockMLLMClient
    return MockMLLMClient(mode="two_calls", seed=42)


@pytest.fixture
def mock_mllm_conflict():
    from mllm.mock_client import MockMLLMClient
    return MockMLLMClient(mode="conflict", seed=42)


# ---------------------------------------------------------------------------
# Expert fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def freq_expert():
    from experts.frequency import FrequencyExpert
    return FrequencyExpert()


@pytest.fixture
def noise_expert():
    from experts.noise import NoiseExpert
    return NoiseExpert()


@pytest.fixture
def jpeg_expert():
    from experts.jpeg import JPEGExpert
    return JPEGExpert()
