"""
The score-semantics registry: one definition, checked so it cannot drift.

These tests exist because v1 drifted exactly here — the audit and the runtime
disagreed about which experts meant "AI-generated", and the noise expert's
polarity was read backwards long enough to be published in a report.  The
registry is the single place that answers the question now, so the answers are
pinned here.
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from utils.score_semantics import (
    DIRECTIONS,
    SCHEMA,
    SemanticsError,
    checkpoints,
    direction_of,
    entry,
    is_calibrated_probability,
    load,
    measurement_scope_of,
    summary,
)

pytestmark = pytest.mark.core

THE_FOUR = ("probe_dino_expert", "noise_expert", "npr_expert", "univfd_expert")


class TestTheRegistryIsWellFormed:
    def test_it_loads(self):
        data = load()
        assert data["schema"] == SCHEMA
        assert set(data["experts"]) >= set(THE_FOUR)

    def test_every_direction_is_from_the_agreed_vocabulary(self):
        for name, record in load()["experts"].items():
            assert record["raw_score"]["direction"] in DIRECTIONS, name

    def test_a_broken_registry_is_refused_not_defaulted(self, tmp_path):
        broken = tmp_path / "score_semantics.json"
        data = json.loads(json.dumps(load()))
        data["experts"]["noise_expert"]["raw_score"]["direction"] = "high_is_fake"
        broken.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(SemanticsError) as excinfo:
            load(str(broken), use_cache=False)
        assert "direction" in str(excinfo.value)

    def test_a_missing_calibration_flag_is_refused(self, tmp_path):
        """sigmoid ≠ calibrated probability; the field is what keeps them apart."""
        broken = tmp_path / "score_semantics.json"
        data = json.loads(json.dumps(load()))
        del data["experts"]["npr_expert"]["raw_score"]["is_calibrated_probability"]
        broken.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        with pytest.raises(SemanticsError) as excinfo:
            load(str(broken), use_cache=False)
        assert "is_calibrated_probability" in str(excinfo.value)


class TestTheDirectionsAreTheCorrectedOnes:
    def test_noise_takes_its_corrected_polarity(self):
        """
        High noise-consistency means "the residual looks like a camera sensor",
        i.e. Real-leaning.  The pre-G2-e reading was the opposite, and a
        "helpful" flip back to higher_is_fake would silently invert every
        evidence check downstream.
        """
        assert direction_of("noise_expert") == "higher_is_real"

    def test_noises_two_fields_point_in_opposite_directions(self):
        """
        `raw_metric` is the SRM inconsistency (higher = more disrupted = Fake),
        `strength` is its normalised position inside the G2 band (higher = looks
        like a camera sensor = Real).  Treating them as interchangeable inverts
        this expert — which is the mistake this registry exists to prevent.
        """
        score = entry("noise_expert")["raw_score"]
        assert score["field"] == "strength"
        assert score["direction"] == "higher_is_real"
        recorded = score["also_recorded"][0]
        assert recorded["field"] == "raw_metric"
        assert recorded["direction"] == "higher_is_fake"
        assert "无上界" in recorded["range"]

    def test_the_discriminators_point_at_fake(self):
        for name in ("probe_dino_expert", "npr_expert", "univfd_expert"):
            assert direction_of(name) == "higher_is_fake", name

    def test_no_external_score_claims_to_be_a_calibrated_probability(self):
        for name in THE_FOUR:
            assert is_calibrated_probability(name) is False, name


class TestTheMeasurementScopesAreRecorded:
    def test_univfd_is_centre_crop_not_global(self):
        """Its preprocessing has no resize and crops to 224: a token that calls
        this measurement 'global' is the global_evidence_localised defect."""
        assert measurement_scope_of("univfd_expert") == "center_crop"

    def test_the_others_measure_the_whole_image(self):
        for name in ("probe_dino_expert", "noise_expert", "npr_expert"):
            assert measurement_scope_of(name) == "global", name


class TestTheCheckpointsCarryTheirLoadingConvention:
    def test_npr_has_two_checkpoints_with_distinct_loading(self):
        """Same architecture, different save formats: which one to use is a
        per-file question (the project's own test proved the wrong convention
        fails), so the registry records it per file."""
        items = checkpoints("npr_expert")
        assert len(items) == 2
        assert {item["load_as"] for item in items} == {"state_dict['model']",
                                                      "raw_state_dict"}
        for item in items:
            assert len(item["sha256"]) == 64
            assert item["bytes"] > 0

    def test_the_npr_selection_is_still_open_and_that_is_visible(self):
        """A、B 都在 val 上测（用户 2026-10-15 决定）：未选之前不能假装已选。"""
        record = entry("npr_expert")
        assert record["weights"]["selected"] is None
        assert "val" in record["weights"]["selection"]

    def test_univfd_records_that_the_backbone_is_required(self):
        backbone = entry("univfd_expert")["weights"]["backbone"]
        assert backbone["sha256_from_url"] in backbone["url"]
        assert entry("univfd_expert")["weights"]["head"]["bytes"] < 10_000


class TestTheMocksAreKeptOutOfTheResults:
    def test_mocks_live_in_their_own_section(self):
        data = load()
        assert "mock_experts" in data
        for name in data["mock_experts"]["entries"]:
            assert name not in data["experts"]

    def test_the_reason_is_stated(self):
        assert "不得进入任何性能结果" in load()["mock_experts"]["why"]


class TestTheSummaryReadsLikeATable:
    def test_it_names_every_expert_and_the_licence_problem(self):
        text = summary()
        for name in THE_FOUR:
            assert name in text
        # the unstated licences must be visible where a human looks
        assert "未明确" in text and "上游未声明" in text

    def test_the_registry_file_is_the_one_the_docs_point_at(self):
        assert os.path.exists(os.path.join(PROJECT_ROOT, "experts", "score_semantics.json"))
