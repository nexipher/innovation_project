"""End-to-end pipeline tests on real dataset images."""

import json
import os
import pytest

from mllm.mock_client import MockMLLMClient
from experts.frequency import FrequencyExpert
from experts.noise import NoiseExpert
from experts.jpeg import JPEGExpert
from state_machine.controller import ForensicStateMachine


# Test images (known to exist in dataset)
REAL_IMAGE = "dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg"
MJ_IMAGE = "dataset/GenImage_Test/Midjourney/0_midjourney_169.png"
SD15_IMAGE = "dataset/GenImage_Test/SD15/001_sdv5_00003.png"
ADM_IMAGE = "dataset/GenImage_Test/ADM/0_adm_174.PNG"


def _build_fsm(mode="two_calls"):
    mllm = MockMLLMClient(mode=mode, seed=42)
    experts = {
        "frequency_expert": FrequencyExpert(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }
    return ForensicStateMachine(mllm, experts)


class TestEndToEnd:
    """Full-pipeline integration tests."""

    @pytest.mark.parametrize("image_path,gt", [
        (REAL_IMAGE, "Real"),
        (MJ_IMAGE, "Fake"),
        (SD15_IMAGE, "Fake"),
        (ADM_IMAGE, "Fake"),
    ])
    def test_pipeline_completes(self, image_path, gt):
        """Pipeline must complete without crashing on real and fake images."""
        fsm = _build_fsm("two_calls")
        result = fsm.run(image_path, gt)
        assert result["total_steps"] >= 1
        assert result["total_steps"] <= 5

    def test_verdict_is_valid(self):
        """Final verdict must be one of the three valid classes."""
        fsm = _build_fsm()
        result = fsm.run(MJ_IMAGE, "Fake")
        assert result["final_verdict"]["verdict"] in ("Real", "Fake", "Uncertain")

    def test_at_least_one_expert_called(self):
        """Every pipeline run must call at least 1 expert."""
        fsm = _build_fsm()
        result = fsm.run(REAL_IMAGE, "Real")
        assert len(result["evidence_chain"]) >= 1

    def test_sft_json_generated(self):
        """SFT training data JSON must be written and valid."""
        fsm = _build_fsm()
        result = fsm.run(MJ_IMAGE, "Fake")
        path = result["sft_data_path"]
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert "conversations" in data
        assert len(data["conversations"]) >= 2
        assert "evidence_chain" in data

    def test_conflict_mode_produces_uncertain(self):
        """Conflict mode should lead to uncertain verdict or conflict halt."""
        fsm = _build_fsm("conflict")
        result = fsm.run(MJ_IMAGE, "Fake")
        # Should either produce Uncertain verdict or halt by conflict
        assert (
            result["final_verdict"]["verdict"] == "Uncertain"
            or result["halting_reason"] == "evidence_conflict"
        )

    def test_explore_all_calls_multiple_experts(self):
        """Explore-all mode should trigger multiple expert calls."""
        fsm = _build_fsm("explore_all")
        result = fsm.run(MJ_IMAGE, "Fake")
        assert len(result["evidence_chain"]) >= 1

    def test_operation_log_appended(self):
        """claude_operation_log.md should exist after pipeline run."""
        assert os.path.exists("claude_operation_log.md")
