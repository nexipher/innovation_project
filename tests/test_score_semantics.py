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

    def test_noises_two_fields_share_a_direction_and_differ_in_scale(self):
        """
        The code maps `raw_metric` (SRM inconsistency) to `strength` through a
        **monotone increasing** sigmoid, so the two fields point the same way;
        the registry used to claim they were opposite, which was wrong.  What is
        actually inverted is the *empirical* association against the algorithm's
        original assumption — that distinction is what has to survive.
        """
        score = entry("noise_expert")["raw_score"]
        assert score["field"] == "strength"
        assert score["direction"] == "higher_is_real"
        recorded = score["also_recorded"][0]
        assert recorded["field"] == "raw_metric"
        assert recorded["direction"] == score["direction"], "两个字段同号，不是相反"
        assert "无上界" in recorded["range"], "量纲不同才是两者的差别"

    def test_the_assumption_and_the_empirical_finding_are_kept_apart(self):
        history = entry("noise_expert")["direction_history"]
        assert "Fake" in history["algorithm_assumption"]          # 算法原读法
        assert "Real" in history["empirical_finding"]              # 经验关联
        assert "没有翻转" in history["code_state"] or "未做符号翻转" in history["code_state"]
        assert "是错的" in history["correction"], "要留下这次更正的痕迹"

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

    def test_the_npr_selection_carries_its_evidence(self):
        """
        A checkpoint may only be marked selected together with the val evidence
        that decided it (用户 2026-10-15：val 负责专家选择).  A bare name would
        be "we chose this" with nothing to check it against.
        """
        selected = entry("npr_expert")["weights"]["selected"]
        assert selected["checkpoint_id"] == "npr-a"
        assert "val" in selected["decided_on_partition"]
        evidence = selected["evidence"]
        assert evidence["npr-a"]["auroc"] > evidence["npr-b"]["auroc"]
        assert set(evidence["npr-a"]["per_format"]) == {"png", "jpeg_q70",
                                                        "jpeg_q85", "jpeg_q95"}
        # and the honest caveat travels with the choice
        assert "格式受限" in selected["caveat"]

    def test_the_selected_checkpoint_is_one_of_the_registered_ones(self):
        ids = {item["id"] for item in checkpoints("npr_expert")}
        assert entry("npr_expert")["weights"]["selected"]["checkpoint_id"] in ids

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
