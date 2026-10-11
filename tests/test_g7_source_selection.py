"""
Tests for the first batch's source selection (G7 entry 4, step 2a).

The rule was written before the scores arrived, so it is pinned here the same
way the acceptance criteria were: the quota and the preference order are part
of the experiment, not implementation detail.
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from scripts.g7_select_sources import (
    BANDS,
    SELECTION,
    band_of,
    check_candidates,
    select_candidates,
    summarise_per_source,
)


class TestBands:
    def test_the_edges_land_where_the_table_says(self):
        assert band_of(0.99) == "strong_fake"
        assert band_of(0.75) == "lean_fake"
        assert band_of(0.52) == "mid"
        assert band_of(0.2) == "lean_real"
        assert band_of(0.01) == "strong_real"

    def test_the_band_edges_are_ordered_and_cover_zero_to_one(self):
        ordered = sorted(BANDS, key=lambda band: band[1])
        assert ordered[0][1] <= 0.0 and ordered[-1][2] > 1.0
        for (_, _, high), (_, low, _) in zip(ordered, ordered[1:]):
            assert high == low


class TestCandidatePool:
    def test_the_pool_comes_from_the_train_partition_only(self):
        candidates = select_candidates()
        assert candidates
        assert all(source["split"] == "train" for source in candidates)
        assert check_candidates(candidates) == []

    def test_the_pool_covers_both_classes_and_every_generator(self):
        candidates = select_candidates()
        assert {s["label"] for s in candidates} == {"Real", "Fake"}
        assert len({s["generator"] for s in candidates}) >= 8

    def test_the_same_seed_draws_the_same_pool(self):
        first = [s["source_id"] for s in select_candidates(seed=5)]
        second = [s["source_id"] for s in select_candidates(seed=5)]
        assert first == second

    def test_a_blocked_source_is_refused(self):
        """The guard: nothing already used may enter the pool."""
        candidates = select_candidates()
        with open(os.path.join(PROJECT_ROOT, "calibration", "set",
                               "acceptance_batch.json"), encoding="utf-8") as handle:
            blocked = json.load(handle)["samples"][0]["source_path"]
        candidates[0] = {**candidates[0], "path": blocked}
        problems = check_candidates(candidates)
        assert any("重叠" in problem for problem in problems)


class TestSummaryAndChoice:
    def _rows(self, source_id, label, bands):
        generators = {"Real": "Real"}
        return [{"source_id": source_id, "label": label,
                 "generator": generators.get(label, "ADM"),
                 "treatment": "png", "path": "x", "raw_score": 0.5,
                 "p_fake": 0.5, "band": band} for band in bands]

    def test_the_summary_counts_bands_per_source(self):
        summary = summarise_per_source(
            self._rows("s1", "Fake", ["strong_fake", "mid", "mid"]))
        assert summary[0]["bands"] == {"strong_fake": 1, "mid": 2}
        assert summary[0]["n_strong"] == 1 and summary[0]["n_mid"] == 2
        assert summary[0]["spread"] is True

    def test_a_single_band_source_is_not_spread(self):
        summary = summarise_per_source(self._rows("s1", "Fake", ["mid", "mid"]))
        assert summary[0]["spread"] is False


class TestTheRuleItself:
    def _summary(self, label, count, generator="ADM", strong=1, mid=1):
        return [{"source_id": f"{label}_{generator}_{i:03d}", "label": label,
                 "generator": generator, "bands": {}, "band_set": [],
                 "n_strong": strong, "n_mid": mid,
                 "spread": bool(strong and mid)}
                for i in range(count)]

    def test_the_quota_is_five_real_and_seven_fake(self):
        from scripts.g7_select_sources import choose

        summary = self._summary("Real", 10) + \
            [dict(entry, generator=gen)
             for gen in ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15",
                         "VQDM")
             for entry in self._summary("Fake", 3, generator=gen)]
        chosen = choose(summary)
        labels = [entry["label"] for entry in chosen]
        assert labels.count("Real") == SELECTION["Real"]
        assert labels.count("Fake") == SELECTION["Fake"]

    def test_fake_choices_spread_across_generators(self):
        from scripts.g7_select_sources import choose

        generators = ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15",
                      "VQDM", "Wukong")
        summary = self._summary("Real", 10) + \
            [dict(entry, generator=gen) for gen in generators
             for entry in self._summary("Fake", 3, generator=gen)]
        chosen = [entry for entry in choose(summary) if entry["label"] == "Fake"]
        assert len({entry["generator"] for entry in chosen}) == 7

    def test_a_source_covering_both_readings_is_preferred(self):
        from scripts.g7_select_sources import choose

        spread = {"source_id": "Fake_spread", "label": "Fake", "generator": "ADM",
                  "bands": {}, "band_set": [], "n_strong": 1, "n_mid": 1,
                  "spread": True}
        flat = {"source_id": "Fake_flat", "label": "Fake", "generator": "ADM",
                "bands": {}, "band_set": [], "n_strong": 0, "n_mid": 0,
                "spread": False}
        real = self._summary("Real", 5)
        chosen = choose([flat] + real + [spread])
        assert spread in chosen
