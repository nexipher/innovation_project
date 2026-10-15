"""
The official entry (main.py) after the stage-4 switch to Qwen3-VL.

Three things are pinned here because each of them has been wrong in this
project's history: the *current* model is the default and the old one needs an
explicit flag (the v1 "默认仍是 v2" class of mistake), the CPU path does not
import the GPU stack (a 470 MB regression once), and PROBE's mock/real choice
follows the client kind rather than a hard-coded string (which would have given
the historical client a mock detector).
"""

import subprocess
import sys

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core


class TestTheCurrentModelIsTheDefault:
    def test_the_flag_offers_a_current_and_a_historical_client(self):
        result = subprocess.run([sys.executable, "main.py", "--help"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True,
                                timeout=120)
        assert "qwen25" in result.stdout, "历史客户端必须可用但须显式选择"
        assert "Qwen3-VL-8B-Instruct" in result.stdout

    def test_qwen_means_qwen3(self, monkeypatch):
        """`--mllm qwen` must build the Qwen3 client, not the 2015-era one."""
        import main
        from mllm.mock_client import MockMLLMClient

        built = {}

        class StubQwen3(MockMLLMClient):
            def __init__(self, **kwargs):
                super().__init__(mode="probe_then_corroborate")
                built["class"] = "Qwen3VLClient"
                built["kwargs"] = kwargs

        import mllm.qwen3_client as qwen3

        monkeypatch.setattr(qwen3, "Qwen3VLClient", StubQwen3)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        main.build_pipeline("qwen", "probe_then_corroborate")
        assert built.get("class") == "Qwen3VLClient"

    def test_qwen25_builds_the_historical_client(self, monkeypatch):
        import main
        from mllm.mock_client import MockMLLMClient

        built = {}

        class StubQwen25(MockMLLMClient):
            def __init__(self, **kwargs):
                super().__init__(mode="probe_then_corroborate")
                built["class"] = "Qwen25Client"

        import mllm.qwen_client as qwen25

        monkeypatch.setattr(qwen25, "QwenVLClient", StubQwen25)
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        main.build_pipeline("qwen25", "probe_then_corroborate")
        assert built.get("class") == "Qwen25Client"


class TestTheCpuPathStaysLight:
    def test_importing_the_entry_does_not_load_the_gpu_stack(self):
        code = ("import sys, main; "
                "print('transformers' in sys.modules, 'torch' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                                capture_output=True, text=True, timeout=180)
        assert result.stdout.strip() == "False False", result.stdout + result.stderr

    def test_a_mock_pipeline_is_built_without_cuda(self):
        import main

        fsm = main.build_pipeline("mock", "two_calls")
        assert fsm is not None


class TestTheRealEntryRefusesToDegradeToMock:
    """
    用户 2026-10-15 点名的第二件事：真实入口**不得**在没有 GPU 时静默换成 Mock。

    曾经的写法是 `print("Warning: CUDA not available. Falling back to
    MockMLLMClient.")` 然后继续跑——产出的 trace、证据链与落盘记录与真实会话
    无异，验收会把它当成真结果。这是"看起来跑通"里最贵的一种。
    """

    def test_a_real_client_without_cuda_raises(self):
        import main
        from unittest import mock

        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError) as caught:
                main.build_pipeline("qwen")
        assert "mock" in str(caught.value).lower()       # 说清楚 CPU 该显式选 mock

    def test_the_historical_client_is_held_to_the_same_rule(self):
        import main
        from unittest import mock

        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                main.build_pipeline("qwen25")

    def test_the_message_says_how_to_run_on_cpu(self):
        import main
        from unittest import mock

        with mock.patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError) as caught:
                main.build_pipeline("qwen")
        assert "--mllm mock" in str(caught.value)

    def test_the_mock_path_is_still_explicitly_available(self):
        import main

        assert main.build_pipeline("mock") is not None


class TestProbesMockChoiceFollowsTheClientKind:
    def test_a_mock_session_gets_the_mock_probe(self):
        import main
        from experts.probe_dino import MockProbeExpert

        assert isinstance(main.build_probe_expert("mock"), MockProbeExpert)

    def test_a_real_session_does_not_get_the_mock_probe(self):
        """qwen25 is a real session too: it used to fall into the mock branch
        because the check was `mllm_type != "qwen"`."""
        import main
        from experts.probe_dino import MockProbeExpert

        expert = main.build_probe_expert("qwen25")
        assert not isinstance(expert, MockProbeExpert)   # None when weights absent
