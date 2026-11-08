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


def _certainty_records():
    batch = (Path(PROJECT_ROOT) / "sft_data/g8_acceptance"
             / "dual_20261108_certainty")
    return {r["variant_id"]: r for r in
            json.loads((batch / "records.json").read_text(encoding="utf-8"))}


@pytest.mark.parametrize("variant_id", [
    "genimage__SD14__045_sdv4_00039__q95",
    "genimage__SD15__042_sdv5_00003__q95",
    "genimage__Wukong__147_wukong_image128__q95",
    "genimage__Midjourney__144_midjourney_34__q95",
    "autosplice__53059__authentic__q95",
])
def test_real_uncertain_with_definite_fake_dimension_is_actionable(variant_id):
    record = _certainty_records()[variant_id]
    assert record["final_verdict"] == "Uncertain"
    assert record["whole_image_generation"] == "AI-generated"
    candidate = {
        "verdict": "Uncertain",
        "whole_image_generation": record["whole_image_generation"],
        "local_manipulation": record["local_manipulation"],
        "certainty": record.get("certainty"),
    }
    issues = coverage_issues(candidate, record["report"],
                             _evidence(record), SERVED)
    assert any(issue["code"] == "overall_uncertain_with_fake_dimension"
               for issue in actionable(issues))


def test_real_biggan_real_with_unmeasured_whole_is_actionable():
    record = _certainty_records()[
        "genimage__BigGAN__051_biggan_00020__q95"]
    assert record["final_verdict"] == "Real"
    assert record["whole_image_generation"] == "Uncertain"
    assert record["local_manipulation"] == "Not tampered"
    candidate = {
        "verdict": "Real",
        "whole_image_generation": record["whole_image_generation"],
        "local_manipulation": record["local_manipulation"],
        "certainty": record.get("certainty"),
    }
    issues = coverage_issues(candidate, record["report"],
                             _evidence(record), SERVED)
    assert any(issue["code"] == "overall_real_with_unmeasured_whole"
               for issue in actionable(issues))


@pytest.mark.parametrize("variant_id", [
    "genimage__Real__351a8be1-104f-49be-9634-7c045dee48a2__q95",
    "autosplice__42005__0__q95",
])
def test_real_uncertain_records_with_non_null_certainty_are_invalid(variant_id):
    record = _certainty_records()[variant_id]
    assert record["final_verdict"] == "Uncertain"
    value, issues = certainty_issues(
        {"verdict": "Uncertain", "certainty": "Low"}, "Uncertain")
    assert value is None and issues
    assert record.get("certainty") is None  # controller canonicalizes, keeps issue
