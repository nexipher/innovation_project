"""
Shared pytest fixtures for the forensic agent test suite.
"""

import subprocess

import numpy as np
import pytest

from config import PROJECT_ROOT


# ---------------------------------------------------------------------------
# Trace isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_sft_dir(tmp_path_factory, monkeypatch):
    """
    Keep test sessions out of traces/sft_sessions/.

    Every test that runs the state machine serialises a trace; without this
    the suite would deposit mock and scripted sessions among the real ones
    that scripts/finalize_sft_data.py collects from.  SessionLogger resolves
    the module-level constant per call, so patching it here is enough.
    """
    import utils.logger

    directory = tmp_path_factory.mktemp("sft_sessions")
    monkeypatch.setattr(utils.logger, "SFT_SESSIONS_DIR", str(directory))
    return directory


# ---------------------------------------------------------------------------
# Calibration fixture for session-mechanics tests
# ---------------------------------------------------------------------------

@pytest.fixture
def legacy_calibration(monkeypatch):
    """
    Pin the pre-repair calibration table (v1) for session-mechanics tests.

    Those tests script a conversation and assert what the pipeline did with
    it; their scripts were written against the v1 posterior, and the repair of
    2026-10-13 legitimately moves the posteriors (the mappings are fitted on
    group A, and each format now uses its own cell's weight) — a different
    posterior accepts a candidate at a different turn.  Pinning v1 keeps them
    testing the ledger/protocol logic they are about.  The repaired table has
    its own tests (tests/test_calibration_source.py).
    """
    import os

    from config import PROJECT_ROOT
    from utils.reliability import ReliabilityTable

    pinned = ReliabilityTable.load(
        os.path.join(PROJECT_ROOT, "calibration", "reliability_table_v1.json"))
    assert pinned is not None
    monkeypatch.setattr(ReliabilityTable, "load",
                        classmethod(lambda cls, path=None: pinned))
    return pinned


@pytest.fixture
def v2_policy(monkeypatch):
    """
    Run a session under the frozen v2 policy.

    The runtime default is now v3 (the current version).  Tests that assert
    v2's own semantics — the override note, the closing-turn flow, the reused
    candidate report — say so explicitly instead of relying on a default.
    """
    from state_machine import controller as controller_module

    monkeypatch.setattr(controller_module, "HALTING_POLICY", "v2")
    return "v2"


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
    """The shipped frequency expert (v2) — v1 is deprecated (plan.md G2-e)."""
    from experts.frequency_v2 import FrequencyExpertV2
    return FrequencyExpertV2()


@pytest.fixture
def freq_expert_v1():
    """The deprecated v1 expert, kept for trace reproducibility only."""
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


# ---------------------------------------------------------------------------
# Repository pollution guard
# ---------------------------------------------------------------------------

#: 本会话开始/结束时比较"未跟踪文件"的仓库内目录。
_POLLUTION_WATCHED = ("sft_data/", "traces/", "dataset/")


def _untracked_under(root: str, prefixes) -> set:
    result = subprocess.run(["git", "status", "--porcelain", "-uall"],
                            cwd=root, capture_output=True, text=True)
    paths = set()
    for line in result.stdout.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:].strip()
        if path.startswith(tuple(prefixes)):
            paths.add(path)
    return paths


@pytest.fixture(scope="session", autouse=True)
def tests_leave_no_untracked_files():
    """
    **测试不得在仓库里留下未跟踪产物。**

    这条规则已经被违反过三次：批入口的 dry-run 记录、审阅视图的夹具、b2 输入比对的
    夹具，各在 `sft_data/` 下留过目录——用户看到的就是"工作区不干净"。逐个修不够，
    下一次写新测试的人未必记得，所以在会话级别兜底：

      * 只比较**未跟踪**集合（用户自己改动、已跟踪文件的修改不受影响）；
      * 结束时出现新的未跟踪路径 → 本次 pytest 以错误收场并列出路径；
      * 不检查内容、不删任何东西。

    `isolated_sft_dir` 已经在拦"会话 trace 写进 traces/"这一类；这一条覆盖的是
    `sft_data/`、`dataset/` 等测试自己拼路径写出来的产物。
    """
    root = str(PROJECT_ROOT)
    before = _untracked_under(root, _POLLUTION_WATCHED)
    yield
    new = sorted(_untracked_under(root, _POLLUTION_WATCHED) - before)
    if new:
        raise pytest.UsageError(
            "测试在仓库里留下了未跟踪文件（已拒绝通过）：\n  "
            + "\n  ".join(new)
            + "\n请把测试产物写到 tmp_path，或显式清理。")
