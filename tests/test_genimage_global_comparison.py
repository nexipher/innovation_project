"""Invariants for the PROBE vs SPAI GenImage comparison."""

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core
PATH = Path(PROJECT_ROOT) / "sft_data/g8_external_eval/genimage_dev_20261102/comparison_20261105.json"
GROUPS = ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong")


def test_probe_beats_spai_on_every_generator_in_dev_and_val():
    data = json.load(open(PATH, encoding="utf-8"))
    assert data["inputs"]["real_shared"]
    for set_name in ("dev_no_known_use", "dev_all"):
        for group in GROUPS:
            probe = data["groups"][set_name][group]["probe"]
            spai = data["groups"][set_name][group]["spai"]
            assert probe["auroc"]["auroc"] > spai["auroc"]["auroc"]
            assert probe["balanced_accuracy"] > spai["balanced_accuracy"]
            assert probe["failures"] == 0 and spai["failures"] == 0
    for group in GROUPS:
        probe = data["validation"]["groups"][group]["probe"]
        spai = data["validation"]["groups"][group]["spai"]
        assert probe["auroc"]["auroc"] > spai["auroc"]["auroc"]
    assert data["validation"]["groups"]["BigGAN"]["spai"]["auroc"]["auroc"] < 0.5
