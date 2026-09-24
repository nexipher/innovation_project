"""Unit tests for ForensicStateMachine controller."""

import json
import pytest
from state_machine.controller import ForensicStateMachine

REAL_PATH = "dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg"
FAKE_PATH = "dataset/GenImage_Test/Midjourney/0_midjourney_169.png"


@pytest.fixture
def state_machine(mock_mllm_default):
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.noise import NoiseExpert
    from experts.jpeg import JPEGExpert
    experts = {
        "frequency_expert_v2": FrequencyExpertV2(),
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
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.noise import NoiseExpert
    from experts.jpeg import JPEGExpert
    return {
        "frequency_expert_v2": FrequencyExpertV2(),
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


class TestDiagnosticRegionArtifacts:
    """G1 (§4.8): expert calls persist region crops for multi-turn images."""

    CALL = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
    VERDICT = ('<reasoning>ok</reasoning>\n<verdict>'
               '{"verdict": "Real", "confidence": 0.7, "primary_evidence": [], "report": "ok"}</verdict>')

    def test_region_artifact_saved_and_linked(self):
        import json as _json
        import os as _os
        from config import PROJECT_ROOT

        mllm = ScriptedMLLM([self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        # The conversation turn carries the artifact path.
        evidence_turns = [
            turn for turn in result["conversation"]
            if turn["from"] == "user" and turn.get("image_paths")
        ]
        assert len(evidence_turns) == 1
        artifact_rel = evidence_turns[0]["image_paths"][0]
        assert artifact_rel.endswith(".png")
        assert _os.path.exists(_os.path.join(PROJECT_ROOT, artifact_rel))

        # The evidence chain entry records the same artifact.
        assert result["evidence_chain"][0]["diagnostic_region_image"] == artifact_rel

        # The persisted trace keeps image_paths for downstream SFT use.
        with open(result["sft_data_path"], encoding="utf-8") as handle:
            trace = _json.load(handle)
        persisted = [
            turn for turn in trace["conversations"]
            if turn["from"] == "user" and turn.get("image_paths")
        ]
        assert len(persisted) == 1

    def test_no_artifact_for_duplicate_evidence(self):
        import os as _os
        from config import PROJECT_ROOT

        mllm = ScriptedMLLM([self.CALL, self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        assert result["suppressed_duplicate_count"] == 1
        assert len(result["evidence_chain"]) == 1


class TestEvidenceBundleWiring:
    """G2 (§4.9): expert artifacts, counter-explanations and calibration
    fields flow into the evidence token and the conversation."""

    CALL = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
    VERDICT = ('<reasoning>ok</reasoning>\n<verdict>'
               '{"verdict": "Real", "confidence": 0.7, "primary_evidence": [], "report": "ok"}</verdict>')

    def test_expert_visual_artifact_attached_and_persisted(self):
        import os as _os
        from config import PROJECT_ROOT

        mllm = ScriptedMLLM([self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        token = result["evidence_chain"][0]
        artifacts = token.get("visual_artifacts")
        assert artifacts, "noise expert must render its residual map"
        assert artifacts[0].startswith("traces/evidence/")
        assert _os.path.exists(_os.path.join(PROJECT_ROOT, artifacts[0]))

        # region crop + artifact are both attached to the evidence turn
        evidence_turn = [
            turn for turn in result["conversation"] if turn.get("image_paths")
        ][0]
        assert len(evidence_turn["image_paths"]) == 2

    def test_counter_explanation_present_in_token(self):
        mllm = ScriptedMLLM([self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")
        token = result["evidence_chain"][0]
        assert token["counter_explanation"], token.get("source")

    def test_raw_metric_recorded(self):
        mllm = ScriptedMLLM([self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")
        assert "raw_metric" in result["evidence_chain"][0]


class TestEvidenceInjectionModes:
    """G2-d (§4.9): the four-condition axis controls what the model sees —
    text channel, visual channel, both, or nothing (RGB baseline)."""

    CALL = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
    VERDICT = ('<reasoning>ok</reasoning>\n<verdict>'
               '{"verdict": "Real", "confidence": 0.7, "primary_evidence": [], "report": "ok"}</verdict>')

    def _run(self, mode):
        mllm = ScriptedMLLM([self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts(), evidence_injection=mode)
        return fsm.run(FAKE_PATH, "Fake")

    def _evidence_turns(self, result):
        """User turns that are not the initial <image> prompt."""
        return [
            turn for turn in result["conversation"]
            if turn["from"] == "user" and "<image>" not in turn["value"]
        ]

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            ForensicStateMachine(ScriptedMLLM([self.VERDICT]), _build_experts(),
                                 evidence_injection="text+images")

    def test_none_mode_surfaces_nothing_but_still_audits(self):
        """RGB baseline: the call is executed and recorded, never shown."""
        result = self._run("none")

        assert result["expert_call_count"] == 1
        assert result["unique_evidence_count"] == 1  # audit chain intact
        assert self._evidence_turns(result) == []
        token = result["evidence_chain"][0]
        assert "diagnostic_region_image" not in token
        assert "visual_artifacts" not in token

    def test_text_mode_injects_json_without_images(self):
        """G1 behaviour, kept as the text-only arm of the comparison."""
        result = self._run("text")

        turns = self._evidence_turns(result)
        assert len(turns) == 1
        assert json.loads(turns[0]["value"])["evidence_name"]  # token JSON
        assert not turns[0].get("image_paths")

        token = result["evidence_chain"][0]
        assert "diagnostic_region_image" not in token

    def test_image_mode_injects_artifacts_without_numbers(self):
        """Visual-only arm: images are attached, the metric values are not."""
        result = self._run("image")

        turns = self._evidence_turns(result)
        assert len(turns) == 1
        assert turns[0]["image_paths"], "artifacts must be attached"
        text = turns[0]["value"]
        assert "evidence_name" not in text and "strength" not in text

        # The audit chain keeps the numbers the model does not get to see.
        assert result["evidence_chain"][0]["strength"] > 0

    def test_both_mode_is_the_default(self):
        default = self._run("text+image")
        mllm = ScriptedMLLM([self.CALL, self.VERDICT])
        explicit = ForensicStateMachine(
            mllm, _build_experts(), evidence_injection="text+image"
        ).run(FAKE_PATH, "Fake")

        for result in (default, explicit):
            turns = self._evidence_turns(result)
            assert len(turns) == 1
            assert json.loads(turns[0]["value"])["evidence_name"]
            assert turns[0]["image_paths"]
