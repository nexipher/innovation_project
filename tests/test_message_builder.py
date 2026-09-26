"""Tests for the Qwen multi-turn image protocol (G1 §4.8).

The builder is intentionally torch/transformers-free so the conversation
protocol can be verified on CPU.
"""

from PIL import Image

from mllm.message_builder import (
    FORENSIC_SYSTEM_PROMPT,
    build_messages,
    collect_images,
)


def _make_png(path, size=(32, 32), color=(120, 80, 40)):
    Image.new("RGB", size, color).save(path)
    return str(path)


def _has_image(message):
    content = message.get("content", "")
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") == "image"
        for block in content
    )


def _text_of(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(
        block.get("text", "") for block in content if isinstance(block, dict)
    )


class TestBuildMessages:
    def test_no_history_attaches_original_image(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        messages = build_messages(original, [])
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == FORENSIC_SYSTEM_PROMPT
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) == 1
        assert _has_image(user_messages[0])

    def test_first_turn_keeps_original_image(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        history = [{"from": "user", "value": "<image>\n请分析这张图像的真实性。"}]
        messages = build_messages(original, history)
        user_messages = [m for m in messages if m["role"] == "user"]
        assert _has_image(user_messages[0])
        assert "<image>" not in _text_of(user_messages[0])

    def test_evidence_turn_attaches_region_image(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        region = _make_png(tmp_path / "region.png", color=(10, 200, 10))
        history = [
            {"from": "user", "value": "<image>\n请分析。"},
            {"from": "gpt", "value": "<planning>...</planning>"},
            {"from": "user", "value": '{"evidence_name": "x"}',
             "image_paths": [region]},
        ]
        messages = build_messages(original, history)
        evidence_message = messages[-1]
        assert _has_image(evidence_message)
        assert '"evidence_name"' in _text_of(evidence_message)

    def test_original_image_survives_later_turns(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        region = _make_png(tmp_path / "region.png")
        history = [
            {"from": "user", "value": "<image>\n请分析。"},
            {"from": "gpt", "value": "<planning>...</planning>"},
            {"from": "user", "value": "{}", "image_paths": [region]},
            {"from": "gpt", "value": "<reasoning>...</reasoning>"},
        ]
        messages = build_messages(original, history)
        images = collect_images(messages)
        assert len(images) == 2  # original + region, nothing dropped

    def test_missing_region_file_is_skipped(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        history = [
            {"from": "user", "value": "<image>\n请分析。"},
            {"from": "user", "value": "{}",
             "image_paths": [str(tmp_path / "does_not_exist.png")]},
        ]
        messages = build_messages(original, history)
        assert not _has_image(messages[-1])  # text-only, no crash

    def test_at_most_two_region_images_per_turn(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        regions = [_make_png(tmp_path / f"r{i}.png") for i in range(3)]
        history = [
            {"from": "user", "value": "<image>\n请分析。"},
            {"from": "user", "value": "{}", "image_paths": regions},
        ]
        messages = build_messages(original, history)
        attached = sum(
            1 for block in messages[-1]["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        )
        assert attached == 2

    def test_assistant_turns_are_text_only(self, tmp_path):
        original = _make_png(tmp_path / "orig.png")
        history = [{"from": "gpt", "value": "<verdict>{}</verdict>"}]
        messages = build_messages(original, history)
        assistant = [m for m in messages if m["role"] == "assistant"]
        assert assistant[0]["content"] == "<verdict>{}</verdict>"


class TestPromptMeasurementContract:
    """G3-a: the prompt must describe whole-image measurement truthfully."""

    def test_experts_are_described_as_measuring_the_whole_image(self):
        assert "the expert measured the WHOLE image" in FORENSIC_SYSTEM_PROMPT
        assert "Judge the whole image, not the crop" in FORENSIC_SYSTEM_PROMPT

    def test_repeat_calls_are_forbidden_by_expert_not_by_region(self):
        """A repeat call is a repeat measurement, wherever the bbox points."""
        assert "Never call an expert you have already called" in FORENSIC_SYSTEM_PROMPT
        assert "Do not repeat a call for a region you already measured" not in FORENSIC_SYSTEM_PROMPT

    def test_the_read_contract_explains_the_global_scope_field(self):
        assert "measurement_scope" in FORENSIC_SYSTEM_PROMPT


class TestToolPolicies:
    """G4-b: the prompt must describe only the tools a session will serve."""

    def test_default_prompt_is_byte_identical_to_the_shipped_one(self):
        """
        The refactor into a per-policy builder must not have moved a character
        of the all-tools prompt the G3 runs were measured under.
        """
        import hashlib

        from mllm.message_builder import build_forensic_prompt

        assert FORENSIC_SYSTEM_PROMPT == build_forensic_prompt(None)
        assert FORENSIC_SYSTEM_PROMPT == build_forensic_prompt(["freq", "noise", "jpeg"])
        assert hashlib.sha256(FORENSIC_SYSTEM_PROMPT.encode()).hexdigest()[:16] == \
            "09dd259ae0b130f6"

    def test_subset_lists_only_the_served_tools(self):
        from mllm.message_builder import build_forensic_prompt

        prompt = build_forensic_prompt(["noise"])
        actions = prompt.split("variations):")[1].split("FORBIDDEN")[0]
        assert "- <call_noise>" in actions
        assert "- <call_freq>" not in actions
        assert "- <call_jpeg>" not in actions

    def test_subset_drops_other_tools_measurement_guidance(self):
        from mllm.message_builder import build_forensic_prompt

        prompt = build_forensic_prompt(["noise"])
        measures = prompt.split("WHAT EACH TOOL ACTUALLY MEASURES")[1]
        assert "<call_noise>" in measures
        assert "<call_jpeg>" not in measures

    def test_subset_says_other_tools_do_not_exist(self):
        from mllm.message_builder import build_forensic_prompt

        subset = build_forensic_prompt(["noise", "jpeg"])
        assert "ONLY the tools listed above exist" in subset
        assert "ONLY the tools listed above exist" not in FORENSIC_SYSTEM_PROMPT

    def test_tool_order_is_stable_whatever_the_caller_passes(self):
        from mllm.message_builder import build_forensic_prompt

        assert build_forensic_prompt(["jpeg", "noise"]) == \
            build_forensic_prompt(["noise", "jpeg"])

    def test_empty_tool_set_is_rejected(self):
        import pytest

        from mllm.message_builder import build_forensic_prompt

        with pytest.raises(ValueError):
            build_forensic_prompt([])
