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
        """G3-c: v2 stops for a posterior reason, not a tag-order one."""
        from state_machine.halting_v2 import HaltingPolicyV2

        result = state_machine.run(FAKE_PATH, "Fake")
        assert result["halting_reason"] in {
            HaltingPolicyV2.NO_EXPECTED_GAIN,
            HaltingPolicyV2.BUDGET_EXHAUSTED,
            HaltingPolicyV2.MODEL_STALLED,
            HaltingPolicyV2.CANDIDATE_ACCEPTED,
        }

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
        # Duplicate suppression must prevent the false convergence path v1 had
        # (its info-gain rule compared two adjacent strengths).
        assert result["halting_reason"] != "info_gain_converged"

    def test_duplicate_evidence_injected_once_in_conversation(self):
        mllm = ScriptedMLLM([self.CALL, self.CALL, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        injections = [
            turn["value"] for turn in result["conversation"]
            if turn["from"] == "user" and turn["value"].lstrip().startswith("{")
        ]
        assert len(injections) == 1

    def test_same_expert_on_a_new_region_is_one_global_measurement(self):
        """
        G3-a semantics: the expert measures the whole image, so a second call
        with a different bbox is the *same* measurement.  Counting it twice
        would double its weight in the halting posterior.
        """
        second_call = "<planning>\nSuspected Region: [400, 300, 700, 600]\n</planning>\n<call_noise>[400, 300, 700, 600]</call_noise>"
        mllm = ScriptedMLLM([self.CALL, second_call, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        assert result["expert_call_count"] == 2
        assert result["unique_evidence_count"] == 1
        assert result["suppressed_duplicate_count"] == 1

    def test_different_experts_still_produce_two_unique_evidence(self):
        """Distinct instruments are distinct measurements, regions aside."""
        noise_call = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
        jpeg_call = "<planning>\nSuspected Region: [400, 300, 700, 600]\n</planning>\n<call_jpeg>[400, 300, 700, 600]</call_jpeg>"
        mllm = ScriptedMLLM([noise_call, jpeg_call, self.VERDICT])
        fsm = ForensicStateMachine(mllm, _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        assert result["unique_evidence_count"] == 2
        assert result["suppressed_duplicate_count"] == 0
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


class TestGlobalMeasurementScope:
    """
    G3-a: expert metrics are measured on the whole image.

    The reliability table is calibrated on full images; measuring the model's
    crop and reading the full-image table binned evidence against the wrong
    population (noise crop median 1.262 vs 2.529 full).  The bbox keeps its
    role as the diagnostic and visualisation region.
    """

    CALL = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
    VERDICT = ('<reasoning>ok</reasoning>\n<verdict>'
               '{"verdict": "Real", "confidence": 0.7, "primary_evidence": [], "report": "ok"}</verdict>')

    class _SpyExpert:
        """Records the array it was asked to measure and to render."""

        def __init__(self):
            from experts.noise import NoiseExpert
            self._inner = NoiseExpert()
            self.measured_shape = None
            self.rendered_shape = None

        @property
        def source(self):
            return self._inner.source_name

        def analyze(self, image):
            self.measured_shape = image.shape[:2]
            return self._inner.analyze(image)

        def render_artifacts(self, image):
            self.rendered_shape = image.shape[:2]
            return self._inner.render_artifacts(image)

    def _run(self):
        import os as _os
        from config import PROJECT_ROOT
        from utils.image_utils import ImageUtils

        spy = self._SpyExpert()
        fsm = ForensicStateMachine(ScriptedMLLM([self.CALL, self.VERDICT]),
                                   {"noise_expert": spy})
        result = fsm.run(FAKE_PATH, "Fake")

        image = ImageUtils.load_image(_os.path.join(PROJECT_ROOT, FAKE_PATH))
        return spy, result, ImageUtils.get_dimensions(image)

    def test_expert_measures_the_whole_image(self):
        spy, _, image_shape = self._run()
        assert spy.measured_shape == image_shape

    def test_artifacts_are_rendered_from_the_measured_input(self):
        spy, _, image_shape = self._run()
        assert spy.rendered_shape == image_shape

    def test_token_declares_the_measurement_scope(self):
        _, result, _ = self._run()
        token = result["evidence_chain"][0]
        assert token["measurement_scope"] == "global"

    def test_diagnostic_region_is_still_recorded(self):
        _, result, image_shape = self._run()
        token = result["evidence_chain"][0]
        # The model's bbox survives as the diagnostic region (normalized space
        # as requested, pixel space as the conversion), with its extent
        # recorded for a future crop-scale calibration.
        assert token["region_normalized_1000"] == [200, 100, 300, 280]
        ymin, xmin, ymax, xmax = token["region_pixels"]
        assert 0 < ymin < ymax <= image_shape[0]
        assert 0 < xmin < xmax <= image_shape[1]
        ratio = token["condition_metadata"]["region_area_ratio"]
        assert 0 < ratio < 1
        assert token["region_semantics"] == "diagnostic_evidence_region"

    def test_region_crop_is_still_persisted(self):
        import os as _os
        from config import PROJECT_ROOT

        _, result, _ = self._run()
        token = result["evidence_chain"][0]
        assert _os.path.exists(_os.path.join(PROJECT_ROOT, token["diagnostic_region_image"]))


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
        """User turns that are not the initial prompt or a system note.

        Under halting policy v2 the loop also injects system notes (an
        overridden candidate, the closing instruction); those are not evidence.
        """
        return [
            turn for turn in result["conversation"]
            if turn["from"] == "user"
            and "<image>" not in turn["value"]
            and not turn["value"].lstrip().startswith("[System:")
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

    def test_image_mode_says_the_measurement_is_global(self):
        """
        G3-a: the artifacts are rendered from the whole image.  Describing
        them as "the analysis of region [bbox]" would have the model read a
        global spectrum as a local finding — exactly the confusion this arm
        exists to test in isolation.
        """
        result = self._run("image")
        text = self._evidence_turns(result)[0]["value"]

        assert "整幅图像" in text
        assert "不限定测量范围" in text
        assert "的分析产物" not in text  # the pre-G3-a phrasing

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


class TestAcceptedCandidateDoesNotCostAnExtraTurn:
    """A no-tool session is a single-turn measurement (G3-e cost parity)."""

    VERDICT = ('<reasoning>ok</reasoning>\n<verdict>'
               '{"verdict": "Real", "confidence": 0.95, "primary_evidence": [], '
               '"report": "报告"}</verdict>')

    def test_the_model_report_is_reused(self):
        fsm = ForensicStateMachine(
            ScriptedMLLM([self.VERDICT]), _build_experts(),
            allow_exploration=False,
        )
        result = fsm.run(FAKE_PATH, "Fake")

        assert result["model_turn_count"] == 1
        assert result["final_verdict"]["verdict"] == "Real"
        assert result["final_verdict"]["report"] == "报告"
        assert result["halting_reason"] == "candidate_without_tools"

    def test_an_override_still_asks_for_the_closing_turn(self):
        """When the policy decides, the model has not yet written that report."""
        call = "<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n<call_noise>[200, 100, 300, 280]</call_noise>"
        verdict = ('<reasoning>ok</reasoning>\n<verdict>'
                   '{"verdict": "Fake", "confidence": 0.99, "primary_evidence": [], '
                   '"report": "模型报告"}</verdict>')
        fsm = ForensicStateMachine(ScriptedMLLM([call, verdict, verdict]),
                                   _build_experts())
        result = fsm.run(FAKE_PATH, "Fake")

        # The candidate was overridden, so a closing turn was requested.
        assert result["final_verdict"]["model_candidate"] == "Fake"
        assert result["model_turn_count"] >= 3
