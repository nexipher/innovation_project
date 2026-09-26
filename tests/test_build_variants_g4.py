"""Tests for the G4-b variant builder."""

import json

import numpy as np
import pytest
from PIL import Image

from scripts.build_variants_g4 import (
    build_variants,
    counts_by,
    select_sources,
    variant_stem,
)


def _split(real=6, fake_per_generator=3, generators=("ADM", "BigGAN")):
    sources = {}
    for index in range(real):
        sources[f"Real/r{index}"] = {"split": "train", "path": f"dataset/Real/r{index}.jpg",
                                     "label": "Real", "generator": "Real",
                                     "resolution": [32, 32]}
    for generator in generators:
        for index in range(fake_per_generator):
            sources[f"{generator}/f{index}"] = {
                "split": "train", "path": f"dataset/GenImage_Test/{generator}/f{index}.png",
                "label": "Fake", "generator": generator, "resolution": [32, 32]}
    sources["Real/held"] = {"split": "test", "path": "dataset/Real/held.jpg",
                            "label": "Real", "generator": "Real", "resolution": [32, 32]}
    return {"hash": "test", "sources": sources}


@pytest.fixture
def dataset(tmp_path):
    """Write the images the split references so the builder can read them."""
    for identifier, entry in _split()["sources"].items():
        path = tmp_path / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((32, 32, 3), 120, dtype=np.uint8)).save(path)
    return tmp_path


class TestSelectSources:
    def test_balanced_draw_takes_equal_classes(self):
        chosen = select_sources(_split(), "train", 6, balanced=True, seed=1)
        assert len(chosen) == 6
        real = sum(1 for c in chosen if c.startswith("Real/"))
        assert real == 3

    def test_it_is_deterministic_for_a_seed(self):
        assert select_sources(_split(), "train", 6, seed=1) == \
            select_sources(_split(), "train", 6, seed=1)

    def test_it_stays_inside_the_partition(self):
        chosen = select_sources(_split(), "train", 20, seed=1)
        assert "Real/held" not in chosen

    def test_natural_ratio_keeps_the_dataset_mix(self):
        # A 3 Real / 6 Fake pool stands in for the dataset's 1:8; asking for a
        # budget of 2 Real sources must still return the natural proportion.
        split = _split(real=3, fake_per_generator=3, generators=("ADM", "BigGAN"))
        chosen = select_sources(split, "train", 2, balanced=False, seed=1)
        real = sum(1 for c in chosen if c.startswith("Real/"))
        fake = len(chosen) - real
        assert (real, fake) == (2, 4)

    def test_unknown_partition_yields_nothing(self):
        assert select_sources(_split(), "ghost", 5) == []


class TestVariantStem:
    def test_generator_stays_in_the_filename(self):
        assert variant_stem("ADM/xyz") == "ADM_xyz"


class TestBuildVariants:
    def test_four_treatments_per_source(self, dataset):
        split = _split()
        manifest = build_variants(["Real/r0"], split, out_dir=str(dataset / "out"),
                                  root=str(dataset))
        assert len(manifest) == 4
        assert {e["treatment"] for e in manifest.values()} == \
            {"png", "jpeg_q95", "jpeg_q85", "jpeg_q70"}
        assert all(e["split"] == "train" for e in manifest.values())

    def test_manifest_carries_provenance(self, dataset):
        split = _split()
        manifest = build_variants(["ADM/f0"], split, out_dir=str(dataset / "out"),
                                  root=str(dataset))
        entry = next(iter(manifest.values()))
        assert entry["generator"] == "ADM"
        assert entry["label"] == "Fake"
        assert entry["source_id"] == "ADM/f0"
        assert entry["resolution"] == [32, 32]

    def test_quality_is_recorded_for_jpeg_only(self, dataset):
        split = _split()
        manifest = build_variants(["Real/r0"], split, out_dir=str(dataset / "out"),
                                  root=str(dataset))
        by_treatment = {e["treatment"]: e for e in manifest.values()}
        assert by_treatment["png"]["quality"] is None
        assert by_treatment["jpeg_q85"]["quality"] == 85
        assert by_treatment["jpeg_q85"]["container"] == "jpg"

    def test_rerun_skips_existing_variants(self, dataset, capsys):
        split = _split()
        out = str(dataset / "out")
        build_variants(["Real/r0"], split, out_dir=out, root=str(dataset))
        build_variants(["Real/r0"], split, out_dir=out, root=str(dataset))
        assert "skipped: 1" in capsys.readouterr().out

    def test_unreadable_source_is_counted_not_raised(self, dataset, capsys):
        split = _split()
        split["sources"]["Real/r0"]["path"] = "dataset/Real/missing.jpg"
        manifest = build_variants(["Real/r0"], split, out_dir=str(dataset / "out"),
                                  root=str(dataset))
        assert manifest == {}
        assert "failed: 1" in capsys.readouterr().out


class TestCounts:
    def test_counts_summarise_variants_and_sources(self, dataset):
        split = _split()
        manifest = build_variants(["Real/r0", "ADM/f0"], split,
                                  out_dir=str(dataset / "out"), root=str(dataset))
        summary = counts_by(manifest)
        assert summary["variants"] == 8
        assert summary["sources"] == 2
        assert summary["labels"] == {"Real": 4, "Fake": 4}
        assert summary["treatments"] == {"jpeg_q70": 2, "jpeg_q85": 2,
                                        "jpeg_q95": 2, "png": 2}
