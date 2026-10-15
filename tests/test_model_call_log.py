"""
模型侧原始记录（阶段 4.6，用户第 7 条）。

b1 的三处缺口之一就是"客户端内部重试不留痕"：`QwenVLClient.generate` 里最多三次
尝试、每次重试还会注入纠正，而 trace 里看起来只有一轮模型输出——**一次被拒的输出
和随后注入的纠正都不见了**。这里钉住补记后的行为，以及"没有就写空、空即未记录"
（旧批次不事后重建）。

本文件导入真实客户端模块（transformers），因此属**模块层**，不进 core。
"""

import json

import pytest

pytestmark = []


@pytest.fixture
def client(monkeypatch):
    """一个不加载权重的 QwenVLClient：推断被替换成脚本化的输出。"""
    from mllm import qwen_client

    client = qwen_client.QwenVLClient(system_prompt="你是取证分析员。")
    monkeypatch.setattr(client, "_ensure_loaded", lambda: None)
    monkeypatch.setattr(client, "_build_messages",
                        lambda image_path, history: [{"role": "user", "content": []}])
    return client


VALID_CALL = '<planning>\\nSuspected Region: none\\n</planning>\\n<call_probe>[0,0,1000,1000]</call_probe>'

VALID_VERDICT = '<reasoning>证据不足。</reasoning>\\n<verdict>{"verdict": "Uncertain", "confidence": 0.4, "report": "证据不足。"}</verdict>'


def _script(client, outputs, monkeypatch):
    """让 `_inference` 依次吐出给定输出，并像真实实现那样填统计。"""
    remaining = list(outputs)

    def fake_inference(messages):
        value = remaining.pop(0)
        client.last_prompt_text = f"<提示 {len(messages)} 条消息>"
        client.last_prompt_image_count = 1
        client.last_input_stats = {"input_tokens": 100, "image_tokens": 9,
                                   "pil_images_attached": 1}
        return value

    monkeypatch.setattr(client, "_inference", fake_inference)


class TestEveryAttemptIsRecorded:
    def test_a_valid_first_attempt_is_one_entry(self, client, monkeypatch):
        _script(client, [VALID_CALL],
                monkeypatch)
        client.generate("img.png", [{"from": "user", "value": "看图"}])
        log = client.call_log()
        assert len(log) == 1
        assert log[0]["turn"] == 1 and log[0]["attempt"] == 0 and log[0]["valid"] is True
        assert log[0]["format_error"] is None

    def test_a_rejected_output_and_its_correction_are_both_kept(self, client, monkeypatch):
        """被格式校验拒掉的原文与随后注入的纠正，都要留。"""
        _script(client, ["这不是合法输出",
                         VALID_VERDICT], monkeypatch)
        client.generate("img.png", [{"from": "user", "value": "看图"}])
        log = client.call_log()
        assert [entry["valid"] for entry in log] == [False, True]
        assert log[0]["raw_output"] == "这不是合法输出"
        assert log[0]["format_error"]
        assert log[1]["correction_from_previous_attempt"] is None or \
            "输出格式错误" in (log[1].get("correction_from_previous_attempt") or "")
        assert client.retry_count == 1

    def test_the_exact_prompt_text_is_kept_with_its_hash(self, client, monkeypatch):
        _script(client, ['<reasoning>x</reasoning>\n<verdict>{"verdict": "Fake"}</verdict>'], monkeypatch)
        client.generate("img.png", [{"from": "user", "value": "看图"}])
        entry = client.call_log()[0]
        assert entry["prompt_text"] == "<提示 1 条消息>"
        assert entry["prompt_sha256"]
        assert entry["images_attached"] == 1
        assert entry["input_stats"]["image_tokens"] == 9

    def test_two_turns_are_two_turn_numbers(self, client, monkeypatch):
        _script(client, [VALID_CALL,
                         '<reasoning>x</reasoning>\n<verdict>{"verdict": "Fake"}</verdict>'], monkeypatch)
        client.generate("img.png", [{"from": "user", "value": "看图"}])
        client.generate("img.png", [{"from": "user", "value": "继续"}])
        assert [entry["turn"] for entry in client.call_log()] == [1, 2]

    def test_reset_clears_the_log_between_sessions(self, client, monkeypatch):
        _script(client, ['<reasoning>x</reasoning>\n<verdict>{"verdict": "Fake"}</verdict>'], monkeypatch)
        client.generate("img.png", [{"from": "user", "value": "看图"}])
        client.reset()
        assert client.call_log() == []
        _script(client, ['<reasoning>x</reasoning>\n<verdict>{"verdict": "Real"}</verdict>'], monkeypatch)
        client.generate("img.png", [{"from": "user", "value": "看图"}])
        assert [entry["turn"] for entry in client.call_log()] == [1]

    def test_the_system_prompt_is_kept_with_its_hash(self, client):
        record = client.system_prompt_record()
        assert record["text"] == "你是取证分析员。"
        assert record["chars"] == len("你是取证分析员。") and record["sha256"]


class TestTheTraceCarriesIt:
    def _session(self, tmp_path, client_obj):
        from state_machine.controller import ForensicStateMachine
        from utils.logger import SessionLogger

        image = tmp_path / "i.png"
        from PIL import Image

        Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
        fsm = ForensicStateMachine(client_obj, {"probe_dino_expert": _Expert()},
                                   logger=SessionLogger(sft_dir=str(tmp_path)))
        return fsm.run(str(image), "Fake")

    def test_a_client_with_the_api_lands_in_the_trace(self, tmp_path, client, monkeypatch):
        _script(client, [VALID_CALL,
                         VALID_VERDICT], monkeypatch)
        result = self._session(tmp_path, client)
        trace = json.load(open(result["sft_data_path"], encoding="utf-8"))
        metadata = trace["metadata"]
        assert metadata["system_prompt"]["text"] == "你是取证分析员。"
        assert [entry["turn"] for entry in metadata["model_calls"]] == [1, 2]
        assert all(entry.get("prompt_text") for entry in metadata["model_calls"])

    def test_a_client_without_the_api_writes_empty_not_invented(self, tmp_path):
        """没有这套接口（旧实现/替身）→ 空结构；页面据此标"未记录"。"""
        result = self._session(tmp_path, _PlainClient())
        trace = json.load(open(result["sft_data_path"], encoding="utf-8"))
        assert trace["metadata"]["model_calls"] == []
        assert trace["metadata"]["system_prompt"] == {}


class _Expert:
    source_name = "probe_dino_expert"
    evidence_name = "aigi_discriminator_score"
    counter_explanation = ""

    def analyze(self, patch):
        from experts.base import ExpertResult

        return ExpertResult(source=self.source_name, evidence_name=self.evidence_name,
                            region="patch_coordinates_[0, 0, 32, 32]", strength=0.9,
                            raw_metric=0.9, support="AI-generated",
                            phenomenon="整图判别分数 0.9000。", reasoning="高。",
                            interpretation_text="指向生成。")


class _PlainClient:
    """老式替身：只有 generate，没有任何模型侧记录接口。"""

    def generate(self, image_path, history):
        return (VALID_CALL
                if len(history) < 3 else
                '<verdict>{"verdict": "Uncertain", "confidence": 0.4, "report": "证据不足。"}</verdict>')

    def reset(self):
        pass

    @property
    def name(self):
        return "plain"

    @property
    def mode(self):
        return "plain"
