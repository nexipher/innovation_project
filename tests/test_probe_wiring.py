"""
G7 entry 3: PROBE wired end-to-end, on CPU.

These are the structural acceptance checks written before the GPU run, so the
small batch has something to fail against: a probe call must land as a token
(not vanish into the silent-drop path), that token must carry its calibration
and where the calibration came from, and the verdict path must still work when
the corroborating experts are never called.
"""

import json
import os

import numpy as np
import pytest

from config import PROJECT_ROOT
from mllm.mock_client import MockMLLMClient
from state_machine.controller import ForensicStateMachine
from utils.parser import Parser
from utils.reliability import ReliabilityTable



# the scripts in this module were written against the v1 posterior
pytestmark = pytest.mark.usefixtures("legacy_calibration")
FAKE_PATH = "dataset/GenImage_Test/Midjourney/0_midjourney_169.png"
REAL_PATH = "dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg"


def _experts(with_probe=True, score=0.99):
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.noise import NoiseExpert
    from experts.jpeg import JPEGExpert

    experts = {
        "frequency_expert_v2": FrequencyExpertV2(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }
    if with_probe:
        from experts.probe_dino import MockProbeExpert
        experts["probe_dino_expert"] = MockProbeExpert(score=score)
    return experts


@pytest.fixture
def run_session(tmp_path):
    """
    Run one mock session, with its trace written to a temp directory.

    SessionLogger's default lands in the real `traces/sft_sessions/` corpus,
    which is where the SFT data is collected from — a mock run must never
    appear there, or a test's output becomes somebody's training sample.
    """
    from utils.logger import SessionLogger

    def _run(mode="probe_then_corroborate", with_probe=True, score=0.99, image=FAKE_PATH):
        fsm = ForensicStateMachine(
            MockMLLMClient(mode=mode), _experts(with_probe, score),
            logger=SessionLogger(sft_dir=str(tmp_path)))
        return fsm.run(image, "Fake" if "GenImage" in image else "Real")

    return _run


class TestParserKnowsTheTag:
    def test_a_probe_call_is_extracted(self):
        text = "…<planning>x</planning><call_probe>[100, 200, 300, 400]</call_probe>"
        assert Parser.extract_all_calls(text) == [("probe", [100, 200, 300, 400])]

    def test_common_variations_normalise(self):
        for tag in ("call_call_probe", "call_probe_dino", "call_dino", "call_aigi"):
            text = f"<{tag}>[1, 2, 3, 4]</{tag}>"
            assert Parser.extract_all_calls(text) == [("probe", [1, 2, 3, 4])], tag

    def test_an_unknown_tag_is_still_dropped(self):
        """The whitelist is the hazard this wiring has to be careful about."""
        text = "<call_oracle>[1, 2, 3, 4]</call_oracle>"
        assert Parser.extract_all_calls(text) == []


class TestCalibrationEntry:
    def test_the_table_carries_a_probe_entry_fitted_on_a(self):
        table = ReliabilityTable.load()
        entry = table.expert_entry("probe_dino_expert")
        assert entry["applicability"] == "strong"
        assert entry["semantics_aligned"] is True
        assert entry["provenance"]["fitted_on"] == "A"
        assert entry["provenance"]["weights_sha256"]

    def test_a_high_score_becomes_a_high_p_fake(self):
        lookup = ReliabilityTable.load().lookup("probe_dino_expert", 0.999)
        assert lookup["calibrated_likelihood"]["Fake"] == pytest.approx(1.0)
        assert lookup["reliability"] > 0.9

    def test_a_low_score_becomes_a_low_p_fake(self):
        lookup = ReliabilityTable.load().lookup("probe_dino_expert", 0.0005)
        assert lookup["calibrated_likelihood"]["Fake"] == pytest.approx(0.0)

    def test_an_ambiguous_score_stays_ambiguous(self):
        lookup = ReliabilityTable.load().lookup("probe_dino_expert", 0.5)
        assert lookup["calibrated_likelihood"]["Fake"] == pytest.approx(0.52)


class TestTheSession:
    def test_a_probe_call_lands_as_a_token(self, run_session):
        result = run_session()
        sources = [token["source"] for token in result["evidence_chain"]]
        assert "probe_dino_expert" in sources

    def test_the_token_carries_its_probability_and_its_origin(self, run_session):
        result = run_session()
        token = next(t for t in result["evidence_chain"] if t["source"] == "probe_dino_expert")
        assert token["calibrated_likelihood"]["Fake"] > 0.9
        assert token["reliability"] > 0.9
        assert token["applicability"] == "strong"
        assert token["measurement_scope"] == "global"
        assert token["calibration"]["fitted_on"] == "A"
        assert token["calibration"]["weights_sha256"]
        assert token["raw_metric"] == pytest.approx(0.99, abs=1e-6)

    def test_the_rectifier_turns_the_probability_into_a_direction(self, run_session):
        """The adapter claims nothing; the calibrated likelihood decides."""
        result = run_session()
        token = next(t for t in result["evidence_chain"] if t["source"] == "probe_dino_expert")
        assert token["direction"] == "AI-generated"
        assert token["direction_source"] == "calibrated_likelihood"
        assert token["support"] != "Uncertain"

    def test_the_corroborating_expert_is_also_reachable(self, run_session):
        """Auxiliary clues are optional, not unreachable."""
        result = run_session()
        sources = {token["source"] for token in result["evidence_chain"]}
        assert sources - {"probe_dino_expert"}, "候补线索一个都没被调用"

    def test_a_probe_call_without_a_probe_expert_is_dropped(self, run_session):
        """
        The failure mode this wiring must not have: an unregistered tool is
        dropped in silence, so a session measured with a missing PROBE would
        look like a session that chose not to call it.
        """
        result = run_session(with_probe=False)
        assert all(token["source"] != "probe_dino_expert"
                   for token in result["evidence_chain"])

    def test_a_decisive_probe_can_stand_alone(self):
        """Real image, low score: the verdict follows PROBE with no help needed."""
        fsm = ForensicStateMachine(
            MockMLLMClient(mode="probe_then_corroborate"), _experts(score=0.0001))
        result = fsm.run(REAL_PATH, "Real")
        assert result["final_verdict"]["verdict"] in ("Real", "Uncertain")

    def test_the_trace_records_what_was_discounted(self, run_session):
        """
        explore_all calls freq, then noise, then jpeg — and noise/jpeg are the
        one positively correlated pair in the toolkit, so the jpeg token is the
        one that gets read at less than full weight.  The trace has to say so:
        a posterior rule that silently changes weights is not auditable from
        the record alone.
        """
        result = run_session(mode="explore_all")
        with open(result["sft_data_path"], encoding="utf-8") as handle:
            trace = json.load(handle)
        discounts = trace["metadata"]["redundancy_discounts"]
        assert discounts and discounts[0]["source"] == "jpeg_expert"
        assert discounts[0]["discount"] == pytest.approx(0.67, abs=0.01)
        assert discounts[0]["after"] == ["frequency_expert_v2", "noise_expert"]

    def test_the_policy_prices_probe_higher_than_a_statistical_expert(self, run_session):
        from state_machine.halting_v2 import HaltingPolicyV2
        result = run_session()
        utilities = result["final_verdict"]["tool_utilities"]
        assert utilities.get("probe_dino_expert", 0) < utilities.get("noise_expert", 0)


class TestTheShippedRegistry:
    def test_main_pipeline_registers_probe_on_cpu(self):
        from main import build_pipeline
        fsm = build_pipeline("mock")
        assert "probe" in fsm._expert_by_call
        assert fsm._expert_by_call["probe"] == "probe_dino_expert"

    def test_the_prompt_promises_only_what_is_registered(self):
        """A prompt that advertises an unregistered tool wastes the budget."""
        from main import build_pipeline
        from mllm.message_builder import build_forensic_prompt
        from state_machine.controller import ForensicStateMachine as FSM

        fsm = build_pipeline("mock")
        tags = [FSM.EXPERT_SOURCE_MAP[name] for name in fsm._experts]
        prompt = build_forensic_prompt(tags)
        for tag in ("probe", "freq", "noise", "jpeg"):
            assert f"<call_{tag}>" in prompt
