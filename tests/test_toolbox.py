"""
The g8 default toolbox (stage 4.4).

Three properties the user's acceptance asks for, each of which has been wrong
in this project before: the default toolbox has a stated identity, it does not
load retired experts, and **every expert the prompt names exists in the
score-semantics registry** — a prompt that advertises a tool nobody can serve
spends the session's budget on dropped calls.
"""

import pytest

from config import PROJECT_ROOT
from mllm.message_builder import build_forensic_prompt
from state_machine.controller import ForensicStateMachine
from utils.score_semantics import SemanticsError, entry, load

pytestmark = pytest.mark.core

RETIRED = ("frequency_expert_v2", "jpeg_expert", "noise_expert")


class TestTheDefaultToolbox:
    def test_it_is_probe_npr_univfd(self):
        from main import build_pipeline

        fsm = build_pipeline("mock")
        assert set(fsm._experts) == {"probe_dino_expert", "npr_expert",
                                     "univfd_expert"}

    def test_no_retired_expert_is_registered_by_default(self):
        from main import build_pipeline

        fsm = build_pipeline("mock")
        for name in RETIRED:
            assert name not in fsm._experts, name

    def test_the_current_tags_are_the_three_the_evaluation_selected(self):
        assert set(ForensicStateMachine.CURRENT_TOOL_TAGS) == {"probe", "npr",
                                                               "univfd"}

    def test_retired_tags_stay_dispatchable_for_old_sessions(self):
        """退役 ≠ 无法复算：旧会话的派发与旧 trace 的解析仍要认得它们。"""
        for source in RETIRED:
            assert source in ForensicStateMachine.EXPERT_SOURCE_MAP


class TestThePromptAndTheRegistryAgree:
    def test_every_tool_the_prompt_names_is_registered(self):
        prompt = build_forensic_prompt(list(ForensicStateMachine.CURRENT_TOOL_TAGS))
        import re

        # 只抓"可用动作"段里逐行列出的动作行；提示词里的反例（"不要写
        # <call_call_probe>"）和模板占位（<call_EXPERT_NAME>）不是宣传的工具。
        named = set(re.findall(r"^- <call_(\w+)>", prompt, re.M))
        assert named == set(ForensicStateMachine.CURRENT_TOOL_TAGS)
        for tag in named:
            source = ForensicStateMachine.EXPERT_KEY_MAP[tag]
            record = entry(source)                      # 未登记会直接抛错
            assert record["display_name"] in prompt

    def test_every_registered_tool_can_be_served_by_the_parser(self):
        """解析器认不出标签，等于该工具在会话里不存在。"""
        from utils.parser import Parser

        for tag in ForensicStateMachine.CURRENT_TOOL_TAGS:
            parsed = Parser.extract_all_calls(f"<call_{tag}>[1, 2, 3, 4]</call_{tag}>")
            assert parsed and parsed[0][0] == tag, tag

    def test_the_registry_covers_the_current_toolbox(self):
        """当前工具箱里的每个专家都必须有登记条目（方向/范围/条件都得有出处）。"""
        experts = load()["experts"]
        for tag in ForensicStateMachine.CURRENT_TOOL_TAGS:
            assert ForensicStateMachine.EXPERT_KEY_MAP[tag] in experts, tag

    def test_retired_experts_may_be_unregistered_and_that_is_stated(self):
        """
        退役专家（jpeg/frequency_v2）没有登记条目——它们已退出候选主线，
        登记表只收当前与候选。它们仍可被派发（供 v2 复算），但**不参与**新的
        方向判定：未登记即"无已核实的尺度与方向"，这一点由精简包如实写出。
        """
        experts = load()["experts"]
        unregistered = [source for source in ("jpeg_expert", "frequency_expert_v2")
                        if source not in experts]
        assert set(unregistered) == {"jpeg_expert", "frequency_expert_v2"}


class TestTheSessionRuns:
    def test_a_full_mock_session_completes_with_the_new_toolbox(self):
        from main import build_pipeline

        fsm = build_pipeline("mock", "probe_then_corroborate")
        import os

        image = os.path.join(PROJECT_ROOT, "dataset", "GenImage_Test",
                             "Midjourney", "0_midjourney_169.png")
        result = fsm.run(image, "Fake")
        sources = {token["source"] for token in result["evidence_chain"]}
        assert sources, "新工具箱一个专家都没被调用"
        assert sources <= {"probe_dino_expert", "npr_expert", "univfd_expert"}
        assert result["final_verdict"]["verdict"] in ("Real", "Fake", "Uncertain")

    def test_an_unregistered_expert_class_is_not_in_the_registry(self):
        """登记表里没有的东西不能偷偷参与会话（未登记就抛错，不静默放行）。"""
        with pytest.raises(SemanticsError):
            entry("some_expert_nobody_registered")
