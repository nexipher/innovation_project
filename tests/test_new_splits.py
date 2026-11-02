"""Invariants for the 2026-11-02 AutoSplice/GenImage q95 splits (plan §10.4)."""

import json
from collections import defaultdict
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core

AUTO_INPUTS = Path(PROJECT_ROOT) / "dataset/AutoSplice_q95/inputs_manifest.json"
GEN_INPUTS = Path(PROJECT_ROOT) / "dataset/GenImage_q95/manifest.json"


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def test_autosplice_keeps_every_base_in_one_partition():
    payload = _load(AUTO_INPUTS)
    by_source = defaultdict(set)
    for row in payload["inputs"]:
        assert row["split"] in {"dev", "val", "test"}
        by_source[row["source_id"]].add(row["split"])
    assert len(by_source) == 2273
    assert all(len(parts) == 1 for parts in by_source.values())
    # One authentic + 1-3 forged per manipulated base; all share the partition.
    by_rows = defaultdict(list)
    for row in payload["inputs"]:
        by_rows[row["source_id"]].append(row)
    assert all(len({r["split"] for r in rows}) == 1
               for rows in by_rows.values())


def test_genimage_sd15_duplicate_is_excluded_and_canonical_is_fixed():
    payload = _load(GEN_INPUTS)
    excluded = [r for r in payload["inputs"] if r["split"] == "excluded_duplicate"]
    assert len(excluded) == 1
    assert excluded[0]["path"].endswith("SD15/680_sdv5_00100.jpg")
    assert excluded[0]["duplicate_of"].endswith("SD15/459_sdv5_00039.png")


def test_genimage_every_source_is_in_one_partition():
    payload = _load(GEN_INPUTS)
    by_source = defaultdict(set)
    for row in payload["inputs"]:
        if row["split"] != "excluded_duplicate":
            assert row["split"] in {"dev", "val", "test"}
            by_source[row["source_id"]].add(row["split"])
    assert len(by_source) == 8999
    assert all(len(parts) == 1 for parts in by_source.values())
