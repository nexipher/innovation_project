"""Tests for the Qwen multi-turn image protocol (G1 §4.8).

The builder is intentionally torch/transformers-free so the conversation
protocol can be verified on CPU.
"""

from PIL import Image

import pytest

from mllm.message_builder import (
    FORENSIC_SYSTEM_PROMPT,
    build_messages,
    collect_images,
)

pytestmark = pytest.mark.core

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

class TestAskAttachesTheImage:
    """
    `build_messages` attaches the primary image only when the turn carries the
    `<image>` marker.  The visual-grounding screen asked a custom question
    without it, so three GPU rounds judged samples from text alone — and the
    judge obligingly described a crocodile as "a cat".
    """

    def test_the_marker_is_added_when_missing(self):
        from mllm.qwen_client import with_image_marker

        marked = with_image_marker([{"from": "user", "value": "这张图里有什么？"}])
        assert marked[0]["value"].startswith("<image>\n")
        assert "这张图里有什么？" in marked[0]["value"]

    def test_an_existing_marker_is_left_alone(self):
        from mllm.qwen_client import with_image_marker

        history = [{"from": "user", "value": "<image>\n看图"}]
        assert with_image_marker(history) == history

    def test_the_message_builder_attaches_the_image_for_a_marked_turn(self, tmp_path):
        image = _make_png(tmp_path / "orig.png")
        messages = build_messages(image, [{"from": "user", "value": "<image>\n看图"}])
        blocks = messages[-1]["content"]
        assert any(block.get("type") == "image" for block in blocks)

    def test_without_the_marker_the_image_is_not_attached(self, tmp_path):
        """The behaviour that made the marker mandatory, stated as a test."""
        image = _make_png(tmp_path / "orig.png")
        messages = build_messages(image, [{"from": "user", "value": "看图"}])
        blocks = messages[-1]["content"]
        assert not any(block.get("type") == "image" for block in blocks)

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
        # 阶段 4.4：工具段由登记表生成，措辞随之更新（语义不变）。
        assert "it measures the WHOLE image and locates nothing" in FORENSIC_SYSTEM_PROMPT
        # 4.4 的读 token 契约用同样意思的新句子（区分 global 与 center_crop）
        assert "Do not restate a crop-based measurement" in FORENSIC_SYSTEM_PROMPT

    def test_repeat_calls_are_forbidden_by_expert_not_by_region(self):
        """A repeat call is a repeat measurement, wherever the bbox points."""
        assert "Call each expert AT MOST ONCE per session" in FORENSIC_SYSTEM_PROMPT
        assert "hard rule, not advice" in FORENSIC_SYSTEM_PROMPT
        assert "Do not repeat a call for a region you already measured" not in FORENSIC_SYSTEM_PROMPT

    def test_the_read_contract_explains_the_scope_field(self):
        """精简包（4.3）把字段改名为 `scope`，并新增 center_crop 的读法。"""
        assert "`scope`" in FORENSIC_SYSTEM_PROMPT
        assert "never saw the edges" in FORENSIC_SYSTEM_PROMPT

class TestThePromptContractAfterB1:
    """
    b1 的人工审核（18/32 待修）里，有两类是**这份提示词请它这么说的**，另外
    审核要求把"结论该说什么"与 confidence 的语义写进契约。这里逐条钉住。

    断言前先把换行压平：提示词按行排版，`cross-expert\n   comparison` 这种
    换行不该让一句契约"看起来不存在"。
    """

    @staticmethod
    def _flat(prompt: str) -> str:
        return " ".join(prompt.split())

    def test_the_presumption_of_innocence_is_gone_except_as_a_ban(self):
        flat = self._flat(FORENSIC_SYSTEM_PROMPT)
        assert flat.count("presumption of innocence") == 1
        index = flat.index("presumption of innocence")
        assert "Never invoke" in flat[max(0, index - 80):index]
        assert "instead of analysing the evidence" in flat[index:index + 90]

    def test_the_template_no_longer_asks_the_model_to_note_compression(self):
        flat = self._flat(FORENSIC_SYSTEM_PROMPT)
        assert "compression artifacts that may weaken" not in flat
        assert "do not assert their cause" in flat

    def test_the_conclusion_must_state_the_evidence_scope(self):
        flat = self._flat(FORENSIC_SYSTEM_PROMPT)
        assert "cross-expert comparison" in flat
        assert "independent cross-validation" in flat          # 只作为禁止的说法出现
        assert "a VALID measurement" in flat
        assert "how many distinct experts returned" in flat

    def test_visible_artefacts_are_observations_not_measurements(self):
        flat = self._flat(FORENSIC_SYSTEM_PROMPT)
        assert "not assert their cause" in flat
        assert 'never infer "high resolution" from appearance' in flat

    def test_the_confidence_contract_is_written_down(self):
        flat = self._flat(FORENSIC_SYSTEM_PROMPT)
        assert "uncalibrated certainty in THIS verdict" in flat
        assert "never P(Fake)" in flat
        assert "averaging or multiplying expert scores" in flat

    def test_the_training_baseline_carries_the_same_confidence_note(self):
        from mllm.message_builder import TRAINING_BASELINE_PROMPT

        assert "uncalibrated certainty in THIS verdict" in TRAINING_BASELINE_PROMPT

    def test_the_frozen_historical_baseline_is_untouched(self):
        """G2-d/G3-e 的对照臂在这份提示词下测过：改它等于悄悄重开那批结论。"""
        from mllm.message_builder import BASELINE_SYSTEM_PROMPT

        assert "uncalibrated certainty" not in BASELINE_SYSTEM_PROMPT
        assert len(BASELINE_SYSTEM_PROMPT) < 900      # 仍是一段短提示词，没被顺手扩写

    def test_probes_token_is_not_called_a_calibrated_probability(self):
        """
        这句话必须在**模型真正读到的地方**：`TOOL_READ_NOTES` 里的 probe 条目曾是
        死代码（登记表标签的说明由 `_registry_tool_text` 生成，本表在拼接时过滤掉
        这些标签），所以它现在写在登记表的 `prompt_note` 里。
        """
        from mllm.message_builder import TOOL_READ_NOTES, build_forensic_prompt

        assert "probe" not in TOOL_READ_NOTES, "probe 的读法不该再放回这张死表"
        prompt = self._flat(build_forensic_prompt(["probe", "npr", "univfd"]))
        assert "uncalibrated output" in prompt
        assert "cite it as a probability" not in prompt
        assert "not a cross-expert comparison" in prompt


class TestToolPolicies:
    """G4-b: the prompt must describe only the tools a session will serve."""

    def test_default_prompt_is_byte_identical_to_the_shipped_one(self):
        """
        The default prompt is a contract: a hash change is a deliberate change
        and must be re-pinned here.

        G7 entry 3 changed it twice: PROBE leads the tool list and the reading
        contract gained rule 6 (its score is a probability, not a located
        artifact) — 09dd259ae0b130f6 → e9e2120587d8049d — and then the planning
        template stopped inviting the call tag inside its prose (the acceptance
        run showed the model writing the tag twice in one response), which
        moved it to e9e2120587d8049d; entry 4 then made the once-per-session
        rule explicit rather than advisory; the repair pass (B segment step 2)
        added the single-action rule, the optional bbox, the "no clear
        anomaly" planning line and the pre-call claim rule, moving it to
        8c342dc61db40df7; the first v3 acceptance batch then showed every one
        of 56 sessions writing a <call_x> tag inside <planning> (the template
        offering the tag as the expert's "name"), so the template now shows a
        bare name and says the tag appears exactly once, after </planning>:
        f4e2eda32c60679b.  The v3 batch then produced the generic sentence
        "未观察到明确异常，需要统计证据" in all 56 sessions because the template
        offered it *in place of* a description, so the planning line now asks
        for a scene description and names the anomalies separately:
        84bc2c766c6d712a.  Stage 4.4 moved it again, twice over: the tool
        sections are now generated from the score-semantics registry (so each
        expert's conditions travel with it), and the reading contract describes
        the slim pack the model actually receives — b6d8a47c514dfe6c.  The
        pre-batch review then split the registry's notes into model-facing and
        internal ones and put the per-value verdict in the pack itself
        (`direction_for_this_value`: a low NPR score under JPEG is `Uncertain`,
        not "Real"), so the contract gained that field, the score's documented
        keys grew `threshold_chosen_on`, and the legacy measures section is now
        omitted when empty instead of leaving a header that describes tools the
        session cannot call — 8143461c167d31ed → a01272f96bbd66c7.

        The first acceptance batch (b1) then came back from human review with 18
        of 32 sessions marked for revision, and two of the criticised habits were
        written into this very template: it told the model to "apply presumption
        of innocence" (which it then invoked even when a single expert meant
        there was no conflict) and to "note" compression artifacts (which it
        turned into ruling them out).  Both are gone, the template now asks for
        what the conclusion must state, and three rules were added: the evidence
        scope (how many experts actually measured, and that a cross-expert
        comparison is not "independent cross-validation"), the boundary between
        describing an artefact and asserting its cause, and the meaning of
        `confidence` — a self-reported, uncalibrated certainty, never P(Real or
        Fake) — a01272f96bbd66c7 → db111b1eb1ecb03d.  Anything that moves it
        again has to say so in a commit.
        """
        import hashlib

        from mllm.message_builder import TOOL_ORDER, build_forensic_prompt

        assert FORENSIC_SYSTEM_PROMPT == build_forensic_prompt(None)
        assert FORENSIC_SYSTEM_PROMPT == build_forensic_prompt(list(TOOL_ORDER))
        assert hashlib.sha256(FORENSIC_SYSTEM_PROMPT.encode()).hexdigest()[:16] == \
            "434ac95a6bbb3e59"

    def test_probe_leads_the_tool_list(self):
        from mllm.message_builder import TOOL_ORDER

        assert TOOL_ORDER[0] == "probe"
        assert "<call_probe>" in FORENSIC_SYSTEM_PROMPT
        # 主检测器的定位现在写在登记表的 prompt_note 里（同一句话进提示词）
        assert "Strongest available evidence" in FORENSIC_SYSTEM_PROMPT

    def test_a_policy_without_probe_never_advertises_it(self):
        """The shared FORBIDDEN sentence names variants, not tools — but nothing
        may offer probe as callable, and no reading rule may describe it."""
        from mllm.message_builder import build_forensic_prompt

        prompt = build_forensic_prompt(["freq", "noise", "jpeg"])
        actions = prompt.split("variations):")[1].split("FORBIDDEN")[0]
        measures = prompt.split("WHAT EACH TOOL ACTUALLY MEASURES")[1].split("HOW TO READ")[0]
        assert "<call_probe>" not in actions
        assert "<call_probe>" not in measures
        assert "PRIMARY detector" not in prompt

    def test_subset_lists_only_the_served_tools(self):
        from mllm.message_builder import build_forensic_prompt

        prompt = build_forensic_prompt(["noise"])
        actions = prompt.split("variations):")[1].split("FORBIDDEN")[0]
        assert "- <call_noise>" in actions
        assert "- <call_freq>" not in actions
        assert "- <call_jpeg>" not in actions

    def test_subset_drops_other_tools_measurement_guidance(self):
        from mllm.message_builder import build_forensic_prompt

        # 登记表支撑的工具其说明已并入 ACTIONS 段，所以"子集只给自己的说明"
        # 要整篇检查，而不是只看 MEASURES 段。
        prompt = build_forensic_prompt(["probe"])
        assert "<call_probe>" in prompt
        assert "<call_npr>" not in prompt
        assert "<call_univfd>" not in prompt
        assert "<call_jpeg>" not in prompt
        # 旧工具（无登记条目）仍走手写段，子集过滤照旧生效
        legacy = build_forensic_prompt(["jpeg"])
        section = legacy.split("WHAT EACH TOOL ACTUALLY MEASURES")[1]
        assert "<call_jpeg>" in section
        assert "<call_freq>" not in legacy and "<call_noise>" not in legacy

    def test_the_legacy_measures_section_disappears_when_it_is_empty(self):
        """
        当前工具组全部由登记表供词，手写段正文是空的。留下段头，就是在提示词里
        向模型描述一批它这次看不到的工具——而段头里原本还带着我们的阶段编号
        （"the G7 fitting group of 62 sources"）。它是对我们才有意义的台账词汇。
        """
        from mllm.message_builder import build_forensic_prompt

        prompt = build_forensic_prompt(["probe", "npr", "univfd"])
        assert "WHAT EACH TOOL ACTUALLY MEASURES" not in prompt
        for internal in ("G7", "700-sample", "bootstrap", "AUROC"):
            assert internal not in prompt, internal

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
