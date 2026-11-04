"""Invariants for the standalone MNW/DIV2K q95 report (plan §11.4)."""

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core

PATH = Path(PROJECT_ROOT) / "sft_data/g8_external_eval/mnw_q95_existing_20261102.json"
GROUPS = {"Adobe_Firefly_v4", "ByteDance_Seedream_v4", "Google_Imagen4",
          "Midjourney_v7"}


def test_mnw_q95_report_is_separate_and_complete():
    data = json.load(open(PATH, encoding="utf-8"))
    assert data["rules"]["only_q95"] is True
    assert data["rules"]["no_genimage_mixing"] is True
    assert data["rules"]["no_threshold_fitting"] is True
    assert set(data["groups"]) == GROUPS | {"Real_only"}
    for group in GROUPS:
        for expert in ("probe", "spai"):
            metrics = data["groups"][group][expert]
            assert metrics["n_fake"] == 20
            assert metrics["n_real"] == 20
            assert metrics["n_measured"] == 40
            assert metrics["failures"] == 0
            assert metrics["threshold"] == 0.5
            assert metrics["auroc"] is not None
            assert metrics["score_distribution"]["fake"]["n"] == 20
            assert metrics["score_distribution"]["real"]["n"] == 20
    assert "original" not in json.dumps(data)
