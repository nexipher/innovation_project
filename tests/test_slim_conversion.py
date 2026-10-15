"""
主线的投递 → 训练对话（阶段 4.5 小批前的 CPU 端到端检查）。

用户在跑小批前点名要确认三件事，这里各有一条机械检查：

1. **转换器适配精简包**：账本原先只认完整 token 的字段名（`source`/`raw_metric`），
   而转换器喂给它的正是 trace 里当时投递的那段 JSON——阶段 4.3 起那是一份精简包。
   于是同一句完全正确的引用（"PROBE 的分数 0.9 指向生成"）被判成
   `fabricated_expert_result`：新主线的每一个样本都会倒在审核这一关。
2. **模型的原始 report 不丢失**：判决 JSON 里的 `report` 是模型自己写的，转换
   必须原样带进最后一个答案。
3. **训练对话不重新注入后台字段**：模型当时只看到精简包，训练对话就不能把
   `strength`/`raw_metric`/`calibrated_likelihood` 塞回去——那是训练与推理不一致，
   等于教它引用自己没见过的字段。

外加一条由阶段 4.4 带出来的：会话现在可以在一次执行失败之后继续，失败记录也进
证据链但没有投递轮，所以交付轮必须按 `evidence_id` 对回 token，而不是按出现次序。
"""

import json

import pytest
from PIL import Image

from state_machine.controller import ForensicStateMachine
from utils.final_v2 import render_sample

pytestmark = pytest.mark.core

REPORT = "整图判别分数 0.9 指向生成，图像的纹理与边缘存在合成痕迹，判定为生成。"


class _Scripted:
    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def generate(self, image_path, history):
        value = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return value

    def reset(self):
        self._index = 0

    @property
    def name(self):
        return "scripted"

    @property
    def mode(self):
        return "scripted"


class _WorkingProbe:
    source_name = "probe_dino_expert"
    evidence_name = "aigi_discriminator_score"
    counter_explanation = ""

    def analyze(self, patch):
        from experts.base import ExpertResult

        return ExpertResult(source=self.source_name, evidence_name=self.evidence_name,
                            region=REGION, strength=0.9, raw_metric=0.9,
                            support="AI-generated",
                            phenomenon="整图判别分数 0.9000（外部判别器的原始输出，未经本地校准）。",
                            reasoning="分数偏高，与训练分布接近。",
                            interpretation_text="指向生成。")


class _BrokenNPR(_WorkingProbe):
    source_name = "npr_expert"
    evidence_name = "npr_upsampling_artefact_score"

    def analyze(self, patch):
        raise RuntimeError("权重文件不存在")


REGION = "patch_coordinates_[0, 0, 64, 64]"

CALL_PROBE = ("<planning>\nSuspected Region: none\n</planning>\n"
              "<call_probe>[0, 0, 1000, 1000]</call_probe>")
CALL_NPR = ("<planning>\nSuspected Region: none\n</planning>\n"
            "<call_npr>[0, 0, 1000, 1000]</call_npr>")
VERDICT = ('<reasoning>整图判别分数 0.9 指向生成，NPR 执行失败没有给出数值，'
           f'但主判据足够明确。</reasoning>\n<verdict>{{"verdict": "Fake", '
           f'"confidence": 0.8, "primary_evidence": [], "report": "{REPORT}"}}</verdict>')


@pytest.fixture(scope="module")
def converted(tmp_path_factory):
    """一次真实会话（含一次执行失败）→ 转换 → 训练样本。"""
    import os

    from utils.logger import SessionLogger

    work = tmp_path_factory.mktemp("slim_conversion")
    image = os.path.join(str(work), "input.png")
    Image.new("RGB", (64, 64), (90, 120, 60)).save(image)

    fsm = ForensicStateMachine(
        _Scripted([CALL_PROBE, CALL_NPR, VERDICT]),
        {"probe_dino_expert": _WorkingProbe(), "npr_expert": _BrokenNPR()},
        logger=SessionLogger(sft_dir=str(work)))
    result = fsm.run(image, "Fake")

    with open(result["sft_data_path"], encoding="utf-8") as handle:
        trace = json.load(handle)
    record = {
        "trajectory_id": "slim_e2e", "ground_truth": "Fake",
        "final_verdict": result["final_verdict"]["verdict"],
        "confidence": result["final_verdict"].get("confidence"),
        "posterior": None, "conflict_score": 0.0, "policy_version": "v3",
        "policy": "g8_default", "policy_reasons": [],
        "tools_served": ["probe_dino_expert", "npr_expert"],
        "split": "test", "generator": "SD", "treatment": "png", "container": "png",
        "image_path": image, "counters": {}, "scores": {},
        "evidence": result["evidence_chain"],
    }
    sample = render_sample(record, "positive", observation_text="模型视觉观察：纹理异常。",
                           trace=trace)
    return {"result": result, "sample": sample, "trace": trace}


class TestTheLedgerReadsWhatWasActuallyDelivered:
    def test_an_honest_session_has_no_defects(self, converted):
        """
        最要紧的一条：一次如实引用的会话，审核不许报缺陷。修复前这里会有一堆
        `fabricated_expert_result`（账本按 `source` 找不到精简包里的 `expert`）。
        """
        audit = converted["sample"]["fidelity"]["dialogue_audit"]
        assert audit["defect_count"] == 0, audit
        assert audit["defect_kinds"] == []

    def test_the_ledger_knows_the_expert_this_session_used(self, converted):
        from utils.evidence_ledger import Ledger, ledger_view

        pack = ledger_view({"expert": "probe_dino_expert", "status": "measured",
                            "scope": "global", "measurement": "aigi_discriminator_score",
                            "score": {"name": "aigi_discriminator_score", "value": 0.9,
                                      "direction_for_this_value": "AI-generated"},
                            "reading": "整图判别分数 0.9000。"})
        ledger = Ledger.from_session([pack], served=["probe_dino_expert"])
        assert ledger.ran("probe_dino_expert")
        assert ledger.measured["probe_dino_expert"].delivered == {"raw_metric": 0.9}
        assert ledger.findings("PROBE 的分数 0.9 指向生成。") == []

    def test_the_two_new_tool_names_are_recognised(self):
        """名字不在 SOURCE_WORDS 里，整句话对账本就是不可见的。"""
        from utils.evidence_ledger import _named_sources

        assert "npr_expert" in _named_sources("NPR 的分数是 0.02。")
        assert "univfd_expert" in _named_sources("UnivFD 给出 0.2。")


class TestTheTrainingConversationCarriesWhatTheModelSaw:
    def _evidence_payloads(self, sample):
        payloads = []
        for turn in sample["conversations"]:
            value = (turn.get("value") or "").lstrip()
            if turn.get("from") == "user" and value.startswith("{"):
                payloads.append(json.loads(value))
        return payloads

    def test_no_background_field_is_reinjected(self, converted):
        from state_machine.evidence_tokenizer import EvidenceTokenizer

        payloads = self._evidence_payloads(converted["sample"])
        assert payloads, "训练对话里没有任何证据轮"
        for payload in payloads:
            assert set(payload) <= set(EvidenceTokenizer.MODEL_VISIBLE_KEYS), payload

    def test_the_numbers_that_never_reached_the_model_are_absent(self, converted):
        blob = json.dumps(self._evidence_payloads(converted["sample"]),
                          ensure_ascii=False)
        for background in ("strength", "raw_metric", "calibrated_likelihood",
                           "reliability", "applicability", "reasoning"):
            assert background not in blob, background

    def test_the_delivered_turn_is_the_token_that_was_delivered(self, converted):
        """
        失败记录也进证据链，但没有投递轮：按次序对齐会把失败 token 顶到交付轮上。
        """
        payloads = self._evidence_payloads(converted["sample"])
        delivered_ids = {payload["evidence_id"] for payload in payloads}
        chain = converted["result"]["evidence_chain"]
        probe = next(t for t in chain if t["source"] == "probe_dino_expert")
        failed = next(t for t in chain if t["source"] == "npr_expert")
        assert probe["evidence_id"] in delivered_ids
        assert failed["evidence_id"] not in delivered_ids
        assert failed["status"] == "failed"


class TestOlderDeliveriesAreUntouched:
    """旧会话当时投递的就是整个 token：转换必须照旧，否则回放对不上。"""

    def _conversation(self, model_delivered, status="measured"):
        token = {"evidence_id": "E-1-noise_expert", "source": "noise_expert",
                 "evidence_name": "noise_residual_inconsistency", "strength": 0.71,
                 "raw_metric": 2.5, "support": "Real", "phenomenon": "残差不一致 2.5000。",
                 "reasoning": "x", "interpretation_text": "y",
                 "measurement_scope": "global", "status": status}
        if model_delivered is not None:
            token["model_delivered"] = model_delivered
        record = {"trajectory_id": "t", "ground_truth": "Fake", "final_verdict": "Real",
                  "confidence": 0.6, "policy_version": "v3", "policy": "noise",
                  "tools_served": ["noise_expert"], "evidence": [token]}
        trace = {"conversations": [
            {"from": "user", "value": "<image>\n看图。"},
            {"from": "gpt", "value": "<planning>…</planning><call_noise>[1,2,3,4]</call_noise>"},
            {"from": "user", "value": json.dumps(token, ensure_ascii=False)},
            {"from": "gpt", "value": "<reasoning>只是观察。</reasoning>\n"
                                      "<verdict>{\"verdict\": \"Real\"}</verdict>"},
        ]}
        sample = render_sample(record, "positive", trace=trace)
        return [json.loads(turn["value"]) for turn in sample["conversations"]
                if turn["from"] == "user" and turn["value"].lstrip().startswith("{")]

    def test_a_full_delivery_still_carries_the_whole_token(self):
        payloads = self._conversation("full")
        assert payloads and "strength" in payloads[0] and "raw_metric" in payloads[0]

    def test_a_token_without_a_stamp_is_treated_as_full(self):
        """旧批次没有这个戳，当时投递的就是完整 token。"""
        payloads = self._conversation(None)
        assert payloads and "strength" in payloads[0]

    def test_a_never_delivered_token_is_not_injected(self):
        """
        `model_delivered: "none"` 的记录在链上，但当时没有投递给模型——
        训练对话里也不该出现它。交付轮认领不到东西，就丢掉这一段。
        """
        assert self._conversation("none") == []


class TestTheModelsOwnReportSurvives:
    def test_the_report_is_in_the_answer_verbatim(self, converted):
        answer = converted["sample"]["conversations"][-1]["value"]
        assert REPORT in answer

    def test_the_reasoning_is_the_models_own_words(self, converted):
        answer = converted["sample"]["conversations"][-1]["value"]
        assert "NPR 执行失败没有给出数值" in answer
        assert converted["sample"]["fidelity"]["model_reasoning_verbatim"] is True
