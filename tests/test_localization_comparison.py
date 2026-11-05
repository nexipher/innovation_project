"""Invariants for the TruFor vs IML-ViT localization comparison."""

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core
PATH = Path(PROJECT_ROOT) / "sft_data/g8_localization/localization_comparison_20261105.json"


def test_localization_inputs_and_failures_are_complete():
    data = json.load(open(PATH, encoding="utf-8"))
    assert data["subset"]["bases"] == 100
    assert data["subset"]["inputs"] == 220
    assert data["subset"]["real"] == 100
    assert data["subset"]["fake"] == 120
    for name in ("trufor", "iml_vit"):
        assert data["experts"][name]["report"]
    assert data["aggregate"]["iml_vit"]["forged_variant_iou"]["n"] == 120
    assert data["aggregate"]["trufor"]["forged_variant_iou"]["n"] == 120


def test_trufor_localizes_better_and_iml_has_no_image_score():
    data = json.load(open(PATH, encoding="utf-8"))
    t = data["aggregate"]["trufor"]
    i = data["aggregate"]["iml_vit"]
    assert t["forged_variant_iou"]["mean"] > i["forged_variant_iou"]["mean"]
    assert t["forged_variant_f1"]["mean"] > i["forged_variant_f1"]["mean"]
    assert t["auroc_source_score"] is not None
    assert i["auroc_source_score"] is None
    assert data["paired_forged_sources"]["trufor_better"] > data["paired_forged_sources"]["iml_better"]
    assert data["experts"]["iml_vit"]["checkpoint"]["strict_load"] is True
