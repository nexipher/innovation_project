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
            "verdict_output", "max_steps_exceeded",
            "evidence_conflict", "info_gain_converged",
        )

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
