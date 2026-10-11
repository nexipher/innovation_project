"""
Tests for group D — the fresh set the TruFor re-check runs on.

D exists to answer one question (does the q85 reversal replicate?) on images
nothing has seen, so what needs pinning is the *disjointness*: from the 98
calibration sources, from A/B/C, and above all from the test partition.  The
rest is bookkeeping: same treatments as the calibration set, balanced classes,
deterministic selection.
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from scripts.build_g7_group_d import (
    MANIFEST_PATH,
    SPLIT_PATH,
    check_disjoint,
    fingerprint,
    load_val_sources,
    select,
)
from scripts.evaluate_external_expert import load_group


def _read(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class TestSelection:
    def test_the_same_seed_selects_the_same_sources(self):
        val = load_val_sources()
        first = [s["source_path"] for s in select(val, seed=20261011)]
        second = [s["source_path"] for s in select(val, seed=20261011)]
        assert first == second
        assert len(set(first)) == len(first)

    def test_the_classes_are_balanced_and_every_generator_is_present(self):
        chosen = select(load_val_sources(), seed=20261011)
        labels = [s["label"] for s in chosen]
        assert labels.count("Real") == 32 and labels.count("Fake") == 32
        fake = {s["generator"] for s in chosen if s["label"] == "Fake"}
        assert len(fake) == 8

    def test_a_source_already_used_is_rejected(self):
        """The guard, not the data: overlap with anything measured must stop the build."""
        val = load_val_sources()
        chosen = select(val, seed=20261011)[:3]
        import scripts.build_g7_group_d as module
        original = module.previous_paths
        module.previous_paths = lambda: {chosen[0]["source_path"]}
        try:
            problems = check_disjoint(chosen, val)
        finally:
            module.previous_paths = original
        assert problems and "重叠" in problems[0]

    def test_the_current_selection_is_clean(self):
        val = load_val_sources()
        assert check_disjoint(select(val, seed=20261011), val) == []


class TestTheCommittedGroup:
    def test_the_manifest_matches_the_calibration_set_s_encoding(self):
        manifest = _read(MANIFEST_PATH)
        assert manifest["group"] == "D" and manifest["partition"].startswith("val")
        treatments = {s["treatment"] for s in manifest["samples"]}
        assert treatments == {"native", "png", "jpeg_q95", "jpeg_q85", "jpeg_q70"}
        assert manifest["counts"] == {"samples": 320, "sources": 64,
                                      "by_label": {"Real": 160, "Fake": 160}}
        # Real is JPEG and Fake is PNG in the native container, as in the
        # calibration set — the confound is reproduced, not papered over.
        native = {label: {s["container"] for s in manifest["samples"]
                          if s["treatment"] == "native" and s["label"] == label}
                  for label in ("Real", "Fake")}
        assert native == {"Real": {"jpg"}, "Fake": {"png"}}

    def test_no_source_straddles_a_treatment(self):
        manifest = _read(MANIFEST_PATH)
        by_source = {}
        for sample in manifest["samples"]:
            by_source.setdefault(sample["source_id"], set()).add(sample["treatment"])
        assert len(by_source) == 64
        assert all(len(treatments) == 5 for treatments in by_source.values())

    def test_d_is_disjoint_from_every_group_that_decides(self):
        """The one property the re-check rests on."""
        split_d = _read(SPLIT_PATH)
        manifest_d = _read(MANIFEST_PATH)
        d_paths = {s["source_path"] for s in manifest_d["samples"]}

        split_abc = _read(os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json"))
        manifest = _read(os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json"))
        abc_paths = {s["source_path"] for s in manifest["samples"]
                     if s["source_id"] in set(split_abc["groups"]["A"])
                     | set(split_abc["groups"]["B"]) | set(split_abc["groups"]["C"])}

        split_v2 = _read(os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json"))
        test_paths = {entry["path"] for entry in split_v2["sources"].values()
                      if entry["split"] == "test"}
        train_paths = {entry["path"] for entry in split_v2["sources"].values()
                       if entry["split"] == "train"}

        assert not (d_paths & abc_paths)
        assert not (d_paths & test_paths)
        assert not (d_paths & train_paths)
        assert split_d["hash"] == fingerprint(sorted(split_d["groups"]["D"]))
        assert len(split_d["groups"]["D"]) == 64

    def test_the_evaluator_reads_group_d_and_still_refuses_nonsense(self):
        samples, split = load_group("D")
        assert len(samples) == 320 and split["hash"] == fingerprint(sorted(split["groups"]["D"]))
        assert {s["label"] for s in samples} == {"Real", "Fake"}
        with pytest.raises(SystemExit, match="unknown group"):
            load_group("E")
