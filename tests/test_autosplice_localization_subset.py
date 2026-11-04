"""Invariants for the fixed AutoSplice localization subset (plan §11.5)."""

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core

PATH = Path(PROJECT_ROOT) / "sft_data/g8_localization/autosplice_dev_subset_20261102.json"


def _load():
    return json.load(open(PATH, encoding="utf-8"))


def test_subset_counts_and_strata_are_fixed():
    data = _load()
    assert data["seed"] == 20261102
    assert data["rules"]["model_runs"] == 0
    assert data["counts"]["bases"] == 100
    assert data["counts"]["inputs"] == 220
    assert data["counts"]["real"] == 100
    assert data["counts"]["fake"] == 120
    assert data["counts"]["by_role"] == {
        "authentic_only": 30, "forged_single": 35,
        "forged_double": 20, "forged_triple": 15,
    }
    assert data["counts"]["by_area_bin"] == {"small": 24, "mid": 24, "large": 22}
    assert data["counts"]["by_size_bin"] == {"small": 10, "mid": 10, "large": 10}


def test_every_base_keeps_all_variants_and_its_own_partition():
    data = _load()
    bases = {base["source_id"]: base for base in data["bases"]}
    inputs = data["inputs"]
    assert len(bases) == 100
    assert len({row["source_id"] for row in inputs}) == 100
    assert len({row["variant_id"] for row in inputs}) == 220
    assert all(row["split"] == "dev" for row in inputs)
    for sid, base in bases.items():
        rows = [row for row in inputs if row["source_id"] == sid]
        assert len(rows) == 1 + base["variant_count"]
        assert sum(row["label"] == "Real" for row in rows) == 1
        assert sum(row["label"] == "Fake" for row in rows) == base["variant_count"]
        assert all(row["split"] == "dev" for row in rows)
        if base["variant_count"]:
            assert base["mask_shared_by_base"] is True
            assert base["mask"]["area_fraction"] is not None
        else:
            assert base["mask"] is None


def test_hashes_were_verified_and_are_present():
    data = _load()
    assert data["verification"]["failures"] == []
    assert data["verification"]["q95_checked"] == 220
    assert data["verification"]["source_checked"] == 220
    assert data["verification"]["mask_checked"] == 190
    for row in data["inputs"]:
        assert len(row["sha256"]) == 64
        assert len(row["source_sha256"]) == 64
        if row["label"] == "Fake":
            assert len(row["mask_sha256"]) == 64
