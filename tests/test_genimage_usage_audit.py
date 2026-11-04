"""Invariants for the GenImage usage audit (plan §11)."""

import json
import re
from pathlib import Path

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core

ROOT = Path(PROJECT_ROOT)
AUDIT = ROOT / "sft_data/g8_audit/genimage_usage_history_20261102.json"
DEV = ROOT / "dataset/GenImage_q95/eval_dev_manifest_20261102.json"
VAL = ROOT / "dataset/GenImage_q95/eval_val_manifest_20261102.json"


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def test_new_test_old_split_composition_is_recorded():
    audit = _load(AUDIT)
    test = audit["new_split"]["test"]
    assert test["inputs"] == 1350
    assert test["old_split"] == {"train": 1258, "val": 37, "test": 42,
                                 "holdout": 13}
    assert test["known_use"] == 138
    assert test["no_known_use"] == 1212
    assert audit["protocol"]["unseen_test_available"] is False


def test_actual_training_is_not_inferred_from_old_train_membership():
    audit = _load(AUDIT)
    assert audit["actual_training"]["sources"] == 20
    assert audit["actual_training"]["samples"] == 237
    # The old split has thousands of train members; only 20 have per-source
    # training samples.  This is the distinction the audit exists to preserve.
    assert audit["old_split"]["by_split"]["train"] == 8332
    assert audit["old_test_prior_use"]["with_known_use"] == 30
    assert audit["old_test_prior_use"]["split_only_unknown"] == 256


def test_selection_sets_and_shared_real_counts():
    audit = _load(AUDIT)
    assert audit["new_split"]["dev"]["no_known_use"] == 5619
    assert audit["new_split"]["val"]["no_known_use"] == 1192
    assert audit["shared_real"]["dev_inputs"] == 700
    assert audit["shared_real"]["dev_no_known_use"] == 435
    assert audit["shared_real"]["val_inputs"] == 150
    assert audit["shared_real"]["val_no_known_use"] == 93


@pytest.mark.parametrize("path,split", [(DEV, "dev"), (VAL, "val")])
def test_eval_manifests_are_usage_consistent_and_have_hashes(path, split):
    payload = _load(path)
    assert payload["split"] == split
    assert payload["inputs"]
    assert all(row["new_split"] == split for row in payload["inputs"])
    assert all(row["condition"] == "jpeg_q95" for row in payload["inputs"])
    for row in payload["inputs"]:
        assert re.fullmatch(r"[0-9a-f]{64}", row["source_sha256"])
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
        assert row["no_known_use"] == (not row["usage_flags"])
        assert f"{split}_all" in row["eval_sets"]
        assert (f"{split}_no_known_use" in row["eval_sets"]) == row["no_known_use"]
        prefix = ("dataset/Real_q95/" if row["group"] == "Real"
                  else f"dataset/GenImage_q95/{row['group']}/")
        assert row["path"].startswith(prefix)
    assert not (ROOT / "dataset/GenImage_q95/eval_test_manifest_20261102.json").exists()
