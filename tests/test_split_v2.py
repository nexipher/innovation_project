"""Tests for the G4-a source-level split and its leak guards."""

import json

import pytest
from PIL import Image

from scripts.build_split_v2 import (
    PARTITIONS,
    build,
    content_hash,
    counts_by,
    enumerate_sources,
    source_id_from_path,
    split_for_variant,
    stratified_split,
    validate,
)


@pytest.fixture
def dataset(tmp_path):
    """A miniature dataset: 20 Real plus 4 generated sources per two generators."""
    root = tmp_path / "proj"
    real = root / "dataset" / "Real"
    real.mkdir(parents=True)
    for index in range(20):
        Image.new("RGB", (32, 24), (index, 10, 10)).save(real / f"r{index:02d}.jpg")
    for generator in ("ADM", "BigGAN"):
        directory = root / "dataset" / "GenImage_Test" / generator
        directory.mkdir(parents=True)
        for index in range(4):
            Image.new("RGB", (16, 16), (0, index, 0)).save(directory / f"f{index}.png")
    return root


def _sources(root):
    return enumerate_sources(str(root))


class TestEnumerate:
    def test_labels_and_generators(self, dataset):
        sources = _sources(dataset)
        assert len(sources) == 28
        generators = {s["generator"] for s in sources}
        assert generators == {"Real", "ADM", "BigGAN"}

    def test_resolution_is_recorded(self, dataset):
        sources = _sources(dataset)
        real = [s for s in sources if s["generator"] == "Real"][0]
        assert real["resolution"] == [24, 32]  # height, width

    def test_source_ids_are_generator_qualified(self, dataset):
        sources = _sources(dataset)
        ids = {s["source_id"] for s in sources}
        assert "Real/r00" in ids
        assert "ADM/f0" in ids and "BigGAN/f0" in ids  # same stem, different generator


class TestSourceIdFromPath:
    def test_real_source(self, tmp_path):
        path = str(tmp_path / "dataset" / "Real" / "abc.jpg")
        assert source_id_from_path(path, root=str(tmp_path)) == "Real/abc"

    def test_genimage_source_uses_the_generator(self, tmp_path):
        path = str(tmp_path / "dataset" / "GenImage_Test" / "SD14" / "xyz.png")
        assert source_id_from_path(path, root=str(tmp_path)) == "SD14/xyz"


class TestVariantResolution:
    """Every re-encoding of a source inherits the source's partition."""

    def test_treatment_suffix_resolves_to_the_source(self, tmp_path):
        sources = {"Real/abc": {"split": "train"}}
        for variant in ("abc.png", "abc_jpeg_q70.png", "abc_jpeg_q95.jpg",
                        "abc_native.jpg", "abc_blur.png", "abc_screenshot.png"):
            path = str(tmp_path / "dataset" / "Real" / variant)
            assert split_for_variant(path, sources, root=str(tmp_path)) == "train"

    def test_unknown_source_has_no_split(self, tmp_path):
        path = str(tmp_path / "dataset" / "Real" / "ghost.png")
        assert split_for_variant(path, {}, root=str(tmp_path)) is None

    def test_planted_cross_partition_variant_is_detected(self, tmp_path):
        """The failure this exists to prevent: PNG in train, q70 in test."""
        sources = {"Real/abc": {"split": "train"}}
        png = str(tmp_path / "dataset" / "Real" / "abc_png.png")
        q70 = str(tmp_path / "dataset" / "Real" / "abc_jpeg_q70.png")
        assert split_for_variant(png, sources, root=str(tmp_path)) == \
            split_for_variant(q70, sources, root=str(tmp_path))


class TestStratifiedSplit:
    def test_deterministic_for_a_seed(self, dataset):
        sources = _sources(dataset)
        first, _ = stratified_split(sources, seed=7)
        second, _ = stratified_split(sources, seed=7)
        assert first == second

    def test_every_source_lands_in_exactly_one_partition(self, dataset):
        sources = _sources(dataset)
        assignment, _ = stratified_split(sources, seed=7)
        assert set(assignment) == {s["source_id"] for s in sources}
        assert set(assignment.values()) <= set(PARTITIONS)

    def test_evaluation_partitions_are_class_balanced(self, dataset):
        """The dataset is 1:8 Real:Fake; the evaluation sets must not be."""
        sources = _sources(dataset)
        assignment, _ = stratified_split(sources, ratios=(0.6, 0.2, 0.2), seed=7)
        generator_of = {s["source_id"]: s["generator"] for s in sources}
        for partition in ("val", "test"):
            members = [i for i, p in assignment.items() if p == partition]
            real = sum(1 for i in members if generator_of[i] == "Real")
            fake = len(members) - real
            assert real > 0
            assert abs(real - fake) <= 1, (partition, real, fake)

    def test_evaluation_fakes_span_generators(self, dataset):
        sources = _sources(dataset)
        assignment, _ = stratified_split(sources, ratios=(0.6, 0.2, 0.2), seed=7)
        generator_of = {s["source_id"]: s["generator"] for s in sources}
        fakes = {generator_of[i] for i, p in assignment.items()
                 if p == "test" and generator_of[i] != "Real"}
        assert fakes == {"ADM", "BigGAN"}

    def test_sampling_weights_offset_the_natural_ratio(self, dataset):
        sources = _sources(dataset)
        assignment, weights = stratified_split(sources, seed=7)
        generator_of = {s["source_id"]: s["generator"] for s in sources}
        train_real = sum(1 for i, p in assignment.items()
                         if p == "train" and generator_of[i] == "Real")
        train_fake = sum(1 for i, p in assignment.items()
                         if p == "train" and generator_of[i] != "Real")
        assert weights["Real"] == 1.0
        assert weights["Fake"] == pytest.approx(train_fake / train_real, abs=1e-4)

    def test_bad_ratios_are_rejected(self, dataset):
        with pytest.raises(ValueError):
            stratified_split(_sources(dataset), ratios=(0.5, 0.2, 0.2))


class TestValidate:
    def _entries(self):
        return {
            "Real/a": {"split": "train", "label": "Real", "generator": "Real"},
            "ADM/b": {"split": "val", "label": "Fake", "generator": "ADM"},
            "ADM/c": {"split": "test", "label": "Fake", "generator": "ADM"},
        }

    def test_a_healthy_split_has_no_problems(self):
        assert validate(self._entries(), {}) == []

    def test_empty_partition_is_reported(self):
        entries = self._entries()
        entries["ADM/c"]["split"] = "train"
        problems = validate(entries, {})
        assert any("test is empty" in p for p in problems)

    def test_holdout_overlap_is_reported(self):
        problems = validate(self._entries(), {"Real/a": "calibration_g2b"})
        assert any("calibration sources" in p for p in problems)

    def test_unknown_partition_and_missing_provenance_are_reported(self):
        entries = self._entries()
        entries["ADM/b"]["split"] = "holdout"
        entries["ADM/c"]["generator"] = ""
        problems = validate(entries, {})
        assert any("unknown partition" in p for p in problems)
        assert any("lacks label/generator" in p for p in problems)


class TestBuild:
    def _manifest(self, root, holdout_ids):
        """Write a calibration manifest that claims some Real sources."""
        manifest_dir = root / "calibration" / "set"
        manifest_dir.mkdir(parents=True)
        samples = [{"source_path": f"dataset/Real/{i}.jpg"} for i in holdout_ids]
        (manifest_dir / "manifest.json").write_text(
            json.dumps({"samples": samples}), encoding="utf-8")

    def test_holdout_sources_are_excluded_from_every_partition(self, dataset, monkeypatch):
        import scripts.build_split_v2 as module

        self._manifest(dataset, ["r00", "r01"])
        monkeypatch.setattr(module, "PROJECT_ROOT", str(dataset))
        monkeypatch.setattr(module, "CALIBRATION_MANIFEST",
                            str(dataset / "calibration" / "set" / "manifest.json"))

        report = build(_sources(dataset), seed=7)
        assert set(report["holdout"]) == {"Real/r00", "Real/r01"}
        assert not (set(report["holdout"]) & set(report["sources"]))

    def test_report_records_counts_generators_and_a_hash(self, dataset, monkeypatch):
        import scripts.build_split_v2 as module

        monkeypatch.setattr(module, "PROJECT_ROOT", str(dataset))
        report = build(_sources(dataset), seed=7)
        assert set(report["counts"]) == set(PARTITIONS)
        assert report["counts"]["train"]["generators"]
        assert report["hash"] == content_hash(report["sources"])
        assert report["train_sampling_weights"]["Fake"] > 0

    def test_hash_changes_when_an_assignment_changes(self):
        base = {"Real/a": {"split": "train"}, "ADM/b": {"split": "test"}}
        moved = {"Real/a": {"split": "train"}, "ADM/b": {"split": "val"}}
        assert content_hash(base) != content_hash(moved)
