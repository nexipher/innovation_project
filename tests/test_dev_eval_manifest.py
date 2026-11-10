"""CPU-only checks for the 200-input g8 dev evaluation manifest."""

import json
from collections import Counter
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core
BATCH = Path(PROJECT_ROOT) / "sft_data/g8_dev_eval_20261110"
MANIFEST = BATCH / "manifest.json"


def _manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_composition_is_exact_and_reproducible():
    payload = _manifest()
    rows = payload["samples"]
    assert payload["seed"] == 20261110
    assert len(rows) == 200
    assert Counter(row["group"] for row in rows) == Counter({
        "Real": 30, "ADM": 10, "BigGAN": 10, "Glide": 10,
        "Midjourney": 10, "SD14": 10, "SD15": 10, "VQDM": 10,
        "Wukong": 10, "AutoSplice_Forged": 60,
        "AutoSplice_Authentic": 30,
    })
    assert Counter(row["task_truth"]["whole_image_generation"] for row in rows) == {
        "Real": 120, "AI-generated": 80}
    assert Counter(row["task_truth"]["local_manipulation"] for row in rows) == {
        "Not tampered": 60, "Not applicable": 80, "Tampered": 60}


def test_sources_and_hashes_are_unique_and_one_input_per_base():
    rows = _manifest()["samples"]
    for key in ("sample_id", "variant_id", "path", "q95_sha256", "source_sha256"):
        values = [row[key] for row in rows]
        assert len(values) == len(set(values)), key
    auto = [row for row in rows if row["family"] == "AutoSplice_q95"]
    auto_ids = [row["source_id"] for row in auto]
    assert len(auto_ids) == len(set(auto_ids))
    forged = {row["source_id"] for row in auto
              if row["source_role"] == "autosplice_forged"}
    authentic = {row["source_id"] for row in auto
                 if row["source_role"] == "autosplice_authentic"}
    assert not forged & authentic


def test_previous_24_sources_and_autosplice_bases_are_excluded():
    payload = _manifest()
    previous = json.loads(
        (Path(PROJECT_ROOT) / "sft_data/g8_acceptance/dual_20261107/manifest.json")
        .read_text(encoding="utf-8"))
    old_gen = {row["source_id"] for row in previous["samples"]
               if row["family"] == "GenImage_q95"}
    old_auto = {str(row["source_id"]) for row in previous["samples"]
                if row["family"] != "GenImage_q95"}
    rows = payload["samples"]
    new_gen = {row["source_id"] for row in rows if row["family"] == "GenImage_q95"}
    new_auto = {str(row["source_id"]) for row in rows if row["family"] == "AutoSplice_q95"}
    assert not old_gen & new_gen
    assert not old_auto & new_auto
    assert set(payload["exclusions"]["autosplice_bases"]) == old_auto


def test_hashes_and_masks_match_the_recorded_identity():
    payload = _manifest()
    rows = payload["samples"]
    assert payload["verification"]["failures"] == []
    assert payload["verification"]["samples_checked"] == 200
    assert payload["verification"]["q95_hashes_checked"] == 200
    assert payload["verification"]["source_hashes_checked"] == 200
    assert payload["verification"]["mask_hashes_checked"] == 60
    for row in rows:
        assert (Path(PROJECT_ROOT) / row["path"]).is_file()
        assert (Path(PROJECT_ROOT) / row["source_path"]).is_file()
        if row["source_role"] == "autosplice_forged":
            assert row["mask_path"] and row["mask_sha256"]
            assert row["mask_shared_by_base"] is True
