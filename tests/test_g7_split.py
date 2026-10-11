"""
Tests for the G7 source-level split (fitting / development / validation).

The rule the review set is that no number may come from a group it was not
supposed to come from, which only holds if the groups are disjoint at the
*source* level and the project's test partition stays out of them entirely.
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from scripts.build_g7_split import (
    DEFAULT_SEED,
    assign,
    build,
    fingerprint,
    load_sources,
    validate,
)


def _manifest(sources):
    """(label, generator, source_id, treatments) tuples → a small manifest."""
    samples = []
    for label, generator, source, treatments in sources:
        for treatment in treatments:
            samples.append({
                "source_id": source, "label": label, "generator": generator,
                "treatment": treatment, "cell": f"{label.lower()}_{treatment}",
                "sample_id": f"{source}_{treatment}", "resolution_bucket": "mid",
            })
    return {"samples": samples}


@pytest.fixture
def manifest_path(tmp_path):
    generators = ["ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]
    sources = []
    for generator in generators:
        for index in range(6):
            sources.append(("Fake", generator, f"fake_{generator}_{index}",
                            ("png", "jpeg_q95", "jpeg_q85", "jpeg_q70")))
    for index in range(50):
        sources.append(("Real", "Real", f"real_{index:04d}",
                        ("png", "jpeg_q95", "jpeg_q85", "jpeg_q70")))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest(sources)), encoding="utf-8")
    return str(path)


class TestAssignment:
    def test_it_is_deterministic_for_a_seed(self, manifest_path):
        sources = load_sources(manifest_path)
        first = assign(sources, DEFAULT_SEED)
        second = assign(sources, DEFAULT_SEED)
        assert first == second
        assert fingerprint(first) == fingerprint(second)

    def test_a_different_seed_gives_a_different_split(self, manifest_path):
        sources = load_sources(manifest_path)
        assert assign(sources, DEFAULT_SEED) != assign(sources, "another-seed")

    def test_the_groups_are_disjoint_and_complete(self, manifest_path):
        sources = load_sources(manifest_path)
        groups = assign(sources, DEFAULT_SEED)
        members = [s for name in groups for s in groups[name]]
        assert len(members) == len(set(members)) == len(sources)
        assert validate(groups, sources) == []

    def test_every_group_can_see_every_generator(self, manifest_path):
        """Otherwise a group's numbers are a different experiment."""
        sources = load_sources(manifest_path)
        groups = assign(sources, DEFAULT_SEED)
        for name, members in groups.items():
            generators = {sources[s]["generator"] for s in members}
            assert len(generators) == 9, (name, generators)

    def test_a_source_keeps_all_its_treatments(self, manifest_path):
        sources = load_sources(manifest_path)
        groups = assign(sources, DEFAULT_SEED)
        for name, members in groups.items():
            for source_id in members:
                assert len(sources[source_id]["treatments"]) == 4
        # every sample of a source lands in exactly one group
        placement = {}
        for name, members in groups.items():
            for source_id in members:
                for sample in sources[source_id]["samples"]:
                    placement.setdefault(sample, set()).add(name)
        assert all(len(where) == 1 for where in placement.values())


class TestValidation:
    def test_an_overlap_is_refused(self, manifest_path):
        sources = load_sources(manifest_path)
        groups = assign(sources, DEFAULT_SEED)
        groups["B"].append(groups["A"][0])
        assert any("重叠" in problem for problem in validate(groups, sources))

    def test_a_missing_source_is_refused(self, manifest_path):
        sources = load_sources(manifest_path)
        groups = assign(sources, DEFAULT_SEED)
        groups["C"] = groups["C"][:-1]
        assert any("不完整" in problem for problem in validate(groups, sources))

    def test_a_degenerate_manifest_cannot_be_split(self, tmp_path):
        tiny = tmp_path / "tiny.json"
        tiny.write_text(json.dumps(_manifest([
            ("Real", "Real", "real_0", ("png",)),
            ("Fake", "ADM", "fake_0", ("png",)),
        ])), encoding="utf-8")
        with pytest.raises(SystemExit, match="划分不合法"):
            build(str(tiny), str(tmp_path / "out.json"), DEFAULT_SEED)


class TestTheRealArtifacts:
    """The split that was actually written, checked against the real pool."""

    def _report(self):
        with open(os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json"),
                  encoding="utf-8") as handle:
            return json.load(handle)

    def test_it_covers_the_calibration_pool(self):
        report = self._report()
        assert sum(report["counts"].values()) == 98
        assert sum(report["samples"].values()) == 700

    def test_the_ratios_are_roughly_60_20_20(self):
        report = self._report()
        for name, target in (("A", 0.6), ("B", 0.2), ("C", 0.2)):
            assert abs(report["counts"][name] / 98 - target) <= 0.08

    def test_no_group_touches_the_project_test_partition(self):
        """
        The final paired comparison runs on split_v2's test partition; the
        fitting groups must not contain any of it.
        """
        report = self._report()
        with open(os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json"),
                  encoding="utf-8") as handle:
            pool = {key.replace("/", "_") for key in json.load(handle)["sources"]}
        assert not (set(report["sources"]) & pool)

    def test_each_group_has_both_labels_and_all_generators(self):
        report = self._report()
        for name in ("A", "B", "C"):
            assert set(report["labels"][name]) == {"Real", "Fake"}
            assert len(report["generators"][name]) == 9
