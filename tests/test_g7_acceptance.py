"""
Tests for the G7 acceptance harness (CPU).

The GPU run is expensive and one-shot, so the parts that decide what it means
are pinned here: which sources are in the batch (and that none of them is
reserved for something else), which mapping the reference arm reads, and how
the summary counts agreement.  A batch that quietly contains a test-partition
image would make the acceptance a measurement of the wrong thing.
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from scripts.g7_acceptance import (
    BATCH_PATH,
    check_sources,
    frozen_probe_mapping,
    mapping_for_report,
    probe_probability,
    select_sources,
    summarise,
)


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class TestBatchSelection:
    def test_the_batch_is_18_development_sources_plus_6_fresh_ones(self):
        sources = select_sources()
        assert len(sources) == 24
        assert sum(1 for s in sources if s["origin"] == "B") == 18
        assert sum(1 for s in sources if s["origin"] == "val") == 6
        assert check_sources(sources) == []

    def test_the_same_seed_selects_the_same_batch(self):
        first = [s["source_path"] for s in select_sources(seed=7)]
        second = [s["source_path"] for s in select_sources(seed=7)]
        assert first == second

    def test_a_different_seed_moves_only_the_fresh_ones(self):
        """B is the whole development group: it is not sampled, it is taken."""
        default = {s["source_path"] for s in select_sources() if s["origin"] == "B"}
        other = {s["source_path"] for s in select_sources(seed=99) if s["origin"] == "B"}
        assert default == other

    def test_a_test_partition_image_is_refused(self):
        """The guard, not the data: the final partition stays untouched."""
        split_v2 = _read(os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json"))
        test_entry = next(e for e in split_v2["sources"].values() if e["split"] == "test")
        sources = select_sources()
        sources[0] = {**sources[0], "source_path": test_entry["path"]}
        problems = check_sources(sources)
        assert any("最终 test 分区" in problem for problem in problems)

    def test_the_fresh_sources_are_not_the_ones_D_used(self):
        manifest_d = _read(os.path.join(PROJECT_ROOT, "calibration", "set", "manifest_d.json"))
        d_paths = {s["source_path"] for s in manifest_d["samples"]}
        val_sources = {s["source_path"] for s in select_sources() if s["origin"] == "val"}
        assert not (val_sources & d_paths)

    def test_the_classes_and_generators_are_covered(self):
        sources = select_sources()
        labels = {s["label"] for s in sources}
        generators = {s["generator"] for s in sources if s["label"] == "Fake"}
        assert labels == {"Real", "Fake"}
        assert len(generators) >= 8      # B alone covers all eight


class TestTheReferenceArm:
    def test_the_mapping_is_the_frozen_a_group_one(self):
        mapping = frozen_probe_mapping()
        assert mapping["fit_group"] == "A"
        assert mapping["features"] == ["probe_dino_expert"]
        assert 0.0 < mapping["threshold"] < 1.0

    def test_the_probability_rises_with_the_score(self):
        mapping = frozen_probe_mapping()
        low, high = probe_probability(mapping, 0.01), probe_probability(mapping, 0.99)
        assert low < 0.5 < high
        assert low < 0.1 and high > 0.9

    def test_it_matches_the_frozen_coefficients_by_hand(self):
        """The reference must be entry 2's model, not a lookalike."""
        import numpy as np
        mapping = frozen_probe_mapping()
        score = 0.42
        z = (score - float(mapping["scaler_mean"][0])) / float(mapping["scaler_scale"][0])
        expected = 1.0 / (1.0 + np.exp(-(float(mapping["coefficients"][0]) * z
                                         + float(mapping["intercept"]))))
        assert probe_probability(mapping, score) == pytest.approx(float(expected), abs=1e-9)

    def test_the_report_version_carries_no_numpy(self):
        payload = mapping_for_report(frozen_probe_mapping())
        json.dumps(payload)                       # would raise on a numpy array
        assert "不合并" in payload["note"]


class TestSummary:
    def _record(self, verdict, calls=("probe_dino_expert",), reason="candidate_matches_posterior"):
        return {"sample_id": "s", "treatment": "png", "verdict": verdict,
                "halting_reason": reason, "calls": list(calls),
                "call_tags": ["probe" if c == "probe_dino_expert" else "noise" for c in calls],
                "elapsed_s": 1.0, "model_turns": 2}

    def test_agreement_counts_each_outcome_once(self):
        records = [self._record("Fake"), self._record("Real"), self._record("Uncertain")]
        reference = {"s": {"verdict": "Fake"}}
        summary = summarise(records, reference)
        assert summary["agreement_with_probe_alone"] == {
            "agree": 1, "disagree": 1, "agent_abstained": 1}

    def test_a_missing_reference_is_not_an_agreement(self):
        summary = summarise([self._record("Fake")], {})
        assert summary["agreement_with_probe_alone"] == {"no_reference": 1}

    def test_the_probe_call_rate_is_counted_from_tokens(self):
        records = [self._record("Fake"),
                   self._record("Real", calls=("noise_expert",))]
        summary = summarise(records, {})
        assert summary["probe_called_in"] == 1
        assert summary["probe_token_rate"] == pytest.approx(0.5)

    def test_first_calls_show_what_the_model_reaches_for(self):
        summary = summarise([self._record("Fake")], {})
        assert summary["first_calls"] == {"probe": 1}


class TestTheCommittedBatch:
    def test_the_manifest_matches_the_selection(self):
        batch = _read(BATCH_PATH)
        assert batch["counts"]["sources"] == 24
        assert batch["counts"]["samples"] == 120
        assert batch["counts"]["by_treatment"] == {
            "native": 24, "png": 24, "jpeg_q95": 24, "jpeg_q85": 24, "jpeg_q70": 24}
        assert {s["source_path"] for s in batch["samples"]} == {
            s["source_path"] for s in select_sources()}

    def test_every_source_contributes_exactly_five_formats(self):
        batch = _read(BATCH_PATH)
        per_source = {}
        for sample in batch["samples"]:
            per_source.setdefault(sample["source_id"], set()).add(sample["treatment"])
        assert len(per_source) == 24
        assert all(len(treatments) == 5 for treatments in per_source.values())

    def test_no_sample_came_from_the_final_partition(self):
        batch = _read(BATCH_PATH)
        split_v2 = _read(os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json"))
        test_paths = {e["path"] for e in split_v2["sources"].values() if e["split"] == "test"}
        assert not ({s["source_path"] for s in batch["samples"]} & test_paths)
