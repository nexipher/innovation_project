"""certainty_v1 against real dual_20261108_peakcrop records."""

import json
from pathlib import Path

import pytest

from config import PROJECT_ROOT
from utils.certainty import certainty_issues
from utils.coverage import actionable, coverage_issues

pytestmark = pytest.mark.core
SERVED = ["probe_dino_expert", "trufor_expert"]


def _records():
    batch = Path(PROJECT_ROOT) / "sft_data/g8_acceptance/dual_20261108_peakcrop"
    return {r["variant_id"]: r for r in
            json.loads((batch / "records.json").read_text(encoding="utf-8"))}


def _evidence(record):
    return [{"source": token.get("source"),
             "status": token.get("status", "measured")}
            for token in record.get("evidence") or []]


@pytest.mark.parametrize("variant_id", [
    "autosplice__41187__0__q95",
    "autosplice__41187__1__q95",
    "autosplice__57501__0__q95",
])
def test_real_tampered_cases_flag_overall_real_inconsistency(variant_id):
    record = _records()[variant_id]
    assert record["final_verdict"] == "Real"
    assert record["local_manipulation"] == "Tampered"
    candidate = {
        "verdict": "Real",
        "whole_image_generation": record["whole_image_generation"],
        "local_manipulation": "Tampered",
        "certainty": "High",
    }
    issues = coverage_issues(candidate, record["report"],
                             _evidence(record), SERVED)
    assert any(issue["code"] == "overall_real_with_fake_dimension"
               for issue in actionable(issues))


def test_midjourney_local_failure_uses_null_certainty_for_uncertain():
    record = _records()["genimage__Midjourney__144_midjourney_34__q95"]
    assert record["final_verdict"] == "Uncertain"
    value, issues = certainty_issues(
        {"verdict": "Uncertain", "certainty": None}, "Uncertain")
    assert value is None and issues == []


@pytest.mark.parametrize("variant_id", [
    "autosplice__39890__0__q95",
    "autosplice__39890__1__q95",
])
def test_single_expert_fake_with_uncertain_whole_is_legal(variant_id):
    record = _records()[variant_id]
    assert record["final_verdict"] == "Fake"
    assert record["whole_image_generation"] == "Uncertain"
    assert record["local_manipulation"] == "Tampered"
    candidate = {
        "verdict": "Fake",
        "whole_image_generation": "Uncertain",
        "local_manipulation": "Tampered",
        "certainty": "Medium",
    }
    assert actionable(coverage_issues(candidate, record["report"],
                                      _evidence(record), SERVED)) == []
    value, issues = certainty_issues(candidate, "Fake")
    assert value == "Medium" and issues == []


def test_uncertain_certainty_must_be_null():
    value, issues = certainty_issues(
        {"verdict": "Uncertain", "certainty": "High"}, "Uncertain")
    assert value is None
    assert any("null" in issue for issue in issues)
