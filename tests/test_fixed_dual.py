"""The fixed PROBE→TruFor dialogue mode."""

import pytest

from mllm.fixed_dual import FixedDualMLLM

pytestmark = pytest.mark.core


class _StubClient:
    mode = "qwen3_real"
    name = "stub-qwen"

    def __init__(self):
        self.system_prompt = "system"
        self.reset_calls = 0
        self.generate_calls = 0

    def reset(self):
        self.reset_calls += 1

    def generate(self, image_path, history):
        self.generate_calls += 1
        return "<reasoning>done</reasoning>"

    def identity(self):
        return {"name": self.name}


def test_first_two_turns_are_fixed_calls_then_the_real_client_runs():
    client = _StubClient()
    fixed = FixedDualMLLM(client)
    assert "call_probe" in fixed.generate("image.jpg", [])
    assert "call_trufor" in fixed.generate("image.jpg", [])
    assert fixed.generate("image.jpg", []) == "<reasoning>done</reasoning>"
    assert client.generate_calls == 1
    assert [entry["call"] for entry in fixed.forced_calls] == ["probe", "trufor"]
    assert all(entry["forced"] for entry in fixed.forced_calls)


def test_reset_restarts_the_fixed_sequence_and_delegates():
    client = _StubClient()
    fixed = FixedDualMLLM(client)
    fixed.generate("image.jpg", [])
    fixed.generate("image.jpg", [])
    fixed.reset()
    assert "call_probe" in fixed.generate("image.jpg", [])
    assert client.reset_calls == 1


def test_client_identity_and_prompt_are_not_replaced():
    client = _StubClient()
    fixed = FixedDualMLLM(client)
    assert fixed.mode == "qwen3_real"
    assert fixed.identity() == {"name": "stub-qwen"}
    assert fixed.system_prompt == "system"
    fixed.system_prompt = "new"
    assert client.system_prompt == "new"
