"""Unit tests for MockMLLMClient."""

import json
from utils.parser import Parser


FAKE_PATH = "dataset/GenImage_Test/Midjourney/test.png"
REAL_PATH = "dataset/Real/test.jpg"


class TestMockMLLM:
    def test_fast_verdict_fake(self, mock_mllm_fast):
        resp = mock_mllm_fast.generate(FAKE_PATH, [])
        calls = Parser.extract_all_calls(resp)
        assert len(calls) >= 1, f"Expected calls in: {resp[:200]}"

    def test_fast_verdict_real(self, mock_mllm_fast):
        mock_mllm_fast.reset()
        resp = mock_mllm_fast.generate(REAL_PATH, [])
        calls = Parser.extract_all_calls(resp)
        assert len(calls) >= 1, f"Expected calls in: {resp[:200]}"

    def test_two_calls_with_evidence_triggers_verdict(self, mock_mllm_default):
        # Turn 0: planning + call
        resp0 = mock_mllm_default.generate(FAKE_PATH, [])
        calls = Parser.extract_all_calls(resp0)
        assert calls

        # Simulate evidence injection
        history = [
            {"from": "gpt", "value": resp0},
            {"from": "user", "value": json.dumps({
                "evidence_name": "test", "strength": 0.15, "source": "freq_expert",
                "support": "Real", "phenomenon": "test", "reasoning": "test",
                "interpretation_text": "test"
            })},
        ]
        resp1 = mock_mllm_default.generate(FAKE_PATH, history)
        verdict = Parser.parse_verdict(resp1)
        assert verdict and "verdict" in verdict, f"No verdict in: {resp1[:200]}"

    def test_conflict_mode_produces_uncertain(self, mock_mllm_conflict):
        resp0 = mock_mllm_conflict.generate(FAKE_PATH, [])
        calls0 = Parser.extract_all_calls(resp0)
        assert calls0

        history = [
            {"from": "gpt", "value": resp0},
            {"from": "user", "value": json.dumps({
                "evidence_name": "test", "strength": 0.9, "source": "freq_expert",
                "support": "AI-generated", "phenomenon": "test", "reasoning": "test",
                "interpretation_text": "test"
            })},
        ]
        resp1 = mock_mllm_conflict.generate(FAKE_PATH, history)
        # After calling noise on T1, should eventually produce uncertain verdict
        history.append({"from": "gpt", "value": resp1})
        history.append({"from": "user", "value": json.dumps({
            "evidence_name": "test2", "strength": 0.1, "source": "noise_expert",
            "support": "Real", "phenomenon": "test", "reasoning": "test",
            "interpretation_text": "test"
        })})
        resp2 = mock_mllm_conflict.generate(FAKE_PATH, history)
        verdict = Parser.parse_verdict(resp2)
        assert verdict and verdict.get("verdict") == "Uncertain"

    def test_reset(self, mock_mllm_default):
        mock_mllm_default.generate(FAKE_PATH, [])
        mock_mllm_default.reset()
        assert mock_mllm_default._turn_count == 0
        assert mock_mllm_default._image_label is None

    def test_response_contains_valid_xml(self, mock_mllm_default):
        resp = mock_mllm_default.generate(FAKE_PATH, [])
        valid, msg = Parser.validate_tag_structure(resp)
        assert valid, f"Invalid structure: {msg}\nResponse: {resp[:200]}"
