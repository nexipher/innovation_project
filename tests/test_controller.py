"""Unit tests for ForensicStateMachine controller."""

import json
import pytest
from state_machine.controller import ForensicStateMachine

REAL_PATH = "dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg"
FAKE_PATH = "dataset/GenImage_Test/Midjourney/0_midjourney_169.png"


@pytest.fixture
def state_machine(mock_mllm_default):
    from experts.frequency import FrequencyExpert
    from experts.noise import NoiseExpert
    from experts.jpeg import JPEGExpert
    experts = {
        "frequency_expert": FrequencyExpert(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }
    return ForensicStateMachine(mock_mllm_default, experts)


class TestStateMachineRun:
    def test_run_returns_dict(self, state_machine):
        result = state_machine.run(REAL_PATH, "Real")
        assert isinstance(result, dict)
        assert "final_verdict" in result
        assert "total_steps" in result
        assert "halting_reason" in result
        assert "evidence_chain" in result
        assert "sft_data_path" in result

    def test_halting_reason_is_valid(self, state_machine):
        result = state_machine.run(FAKE_PATH, "Fake")
        assert result["halting_reason"] in (
            "verdict_output", "budget_exhausted",
            "evidence_conflict", "info_gain_converged",
        )

    def test_counters_are_reported(self, state_machine):
        """G1 (§4.8): run() exposes separated observable counters."""
        result = state_machine.run(FAKE_PATH, "Fake")
        assert result["model_turn_count"] >= 1
        assert result["expert_call_count"] >= 1
        assert result["unique_evidence_count"] == len(result["evidence_chain"])
        assert result["suppressed_duplicate_count"] >= 0
        assert result["weighted_cost"] > 0

    def test_evidence_chain_not_empty(self, state_machine):
        result = state_machine.run(FAKE_PATH, "Fake")
        assert len(result["evidence_chain"]) >= 1

    def test_verdict_has_required_keys(self, state_machine):
        result = state_machine.run(FAKE_PATH, "Fake")
        v = result["final_verdict"]
        assert "verdict" in v
        assert "confidence" in v
        assert v["verdict"] in ("Real", "Fake", "Uncertain")

    def test_sft_file_created(self, state_machine):
        import os
        result = state_machine.run(FAKE_PATH, "Fake")
        assert os.path.exists(result["sft_data_path"])

    def test_real_image_path(self, state_machine):
        """Real image should not crash or produce nonsense."""
        result = state_machine.run(REAL_PATH, "Real")
        assert result["final_verdict"]["confidence"] > 0.0


class ScriptedMLLM:
    """Deterministic scripted client for controller-level G1 tests."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def generate(self, image_path, history):
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    def reset(self):
        self._index = 0

    @property
    def name(self):
        return "scripted"

    @property
    def mode(self):
        return "scripted"


def _build_experts():
    from experts.frequency import FrequencyExpert
    from experts.noise import NoiseExpert
    from experts.jpeg import JPEGExpert
    return {
        "frequency_expert": FrequencyExpert(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }


class TestEvidenceDeduplication:
    """G1 (§4.8): duplicate tool responses must not double-count evidence
    nor trigger false information-gain convergence (the 0b0d0ad4 failure)."""

    CALL = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
    VERDICT = ('<reasoning>正常。</reasoning>\n<verdict>'
               '{"verdict": "Real", "confidence": 0.7, "primary_evidence": [], "report": "ok"}</verdict>')

    def test_identical_repeat_call_is_suppressed(self):
        mllm = ScriptedMLLM([self.CALL, self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        assert result["unique_evidence_count"] == 1
        assert result["expert_call_count"] == 2
        assert result["suppressed_duplicate_count"] == 1
        # Duplicate suppression must prevent false info-gain convergence.
        assert result["halting_reason"] == "verdict_output"

    def test_duplicate_evidence_injected_once_in_conversation(self):
        mllm = ScriptedMLLM([self.CALL, self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        injections = [
            turn["value"] for turn in result["conversation"]
            if turn["from"] == "user" and turn["value"].lstrip().startswith("{")
        ]
        assert len(injections) == 1

    def test_different_regions_produce_two_unique_evidence(self):
        second_call = "<planning>\nSuspected Region: [400, 300, 700, 600]\n</planning>\n<call_noise>[400, 300, 700, 600]</call_noise>"
        mllm = ScriptedMLLM([self.CALL, second_call, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        assert result["unique_evidence_count"] == 2
        assert result["suppressed_duplicate_count"] == 0
