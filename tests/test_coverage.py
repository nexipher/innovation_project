"""Evidence-coverage checks for joint verdict fields."""

import pytest

from utils.coverage import actionable, coverage_issues, coverage_note

pytestmark = pytest.mark.core

SERVED = ["probe_dino_expert", "trufor_expert"]


def _evidence(*sources):
    return [{"source": source, "status": "measured",
             "evidence_id": f"E-{source}"} for source in sources]


def test_local_definite_without_trufor_is_a_defect():
    issues = coverage_issues(
        {"whole_image_generation": "Real", "local_manipulation": "Not tampered"},
        "未发现局部编辑证据。", _evidence("probe_dino_expert"), SERVED)
    assert [item["code"] for item in actionable(issues)] == [
        "local_manipulation_without_trufor"]


def test_whole_definite_without_probe_is_a_defect():
    issues = coverage_issues(
        {"whole_image_generation": "AI-generated",
         "local_manipulation": "Tampered"},
        "局部证据支持结论。", _evidence("trufor_expert"), SERVED)
    assert [item["code"] for item in actionable(issues)] == [
        "whole_image_without_probe"]


def test_single_expert_with_uncertain_dimension_is_legal():
    issues = coverage_issues(
        {"whole_image_generation": "Uncertain",
         "local_manipulation": "Tampered"},
        "TruFor 支持局部篡改，整图生成未测量。",
        _evidence("trufor_expert"), SERVED)
    assert actionable(issues) == []


def test_mutual_support_wording_is_flagged_but_negation_is_not():
    issues = coverage_issues(
        {"whole_image_generation": "AI-generated",
         "local_manipulation": "Tampered"},
        "两个专家的结果相互支持。", _evidence("probe_dino_expert", "trufor_expert"),
        SERVED)
    assert any(item["code"] == "cross_task_mutual_validation"
               for item in issues)
    negated = coverage_issues(
        {"whole_image_generation": "AI-generated",
         "local_manipulation": "Tampered"},
        "两者不互相验证。", _evidence("probe_dino_expert", "trufor_expert"),
        SERVED)
    assert not any(item["code"] == "cross_task_mutual_validation"
                   for item in negated)
    long_negation = coverage_issues(
        {"whole_image_generation": "AI-generated",
         "local_manipulation": "Tampered"},
        "两者结果一致，但不构成相互验证。",
        _evidence("probe_dino_expert", "trufor_expert"), SERVED)
    assert not any(item["code"] == "cross_task_mutual_validation"
                   for item in long_negation)


def test_edit_process_claim_is_actionable_but_negation_is_not():
    issues = coverage_issues(
        {"whole_image_generation": "Real", "local_manipulation": "Tampered"},
        "异常图表明文字是后期添加的。", _evidence("trufor_expert"), SERVED)
    item = next(item for item in issues if item["code"] == "edit_process_claim")
    assert item["severity"] == "wording"
    assert actionable([item]) == [item]
    negated = coverage_issues(
        {"whole_image_generation": "Real", "local_manipulation": "Tampered"},
        "不能断言文字是后期添加的。", _evidence("trufor_expert"), SERVED)
    assert not any(item["code"] == "edit_process_claim" for item in negated)


def test_correction_note_offers_uncertain_and_keeps_supported_parts():
    note = coverage_note([
        {"code": "local_manipulation_without_trufor", "value": "Not tampered"},
        {"code": "whole_image_without_probe", "value": "AI-generated"},
    ])
    assert "局部问题未测时应写 Uncertain" in note
    assert "整图生成分项应写 Uncertain" in note
    assert "不要伪造专家分数" in note


def test_real_35b3b70e_and_39890_0_coverage_failures():
    import json
    from pathlib import Path
    from config import PROJECT_ROOT

    batch = Path(PROJECT_ROOT) / "sft_data/g8_acceptance/dual_20261108_closing"
    records = {
        record["variant_id"]: record
        for record in json.loads((batch / "records.json").read_text(encoding="utf-8"))
    }
    for variant_id, expected in (
        ("genimage__Real__35b3b70e-4994-4618-8aaa-86467988d1d5__q95",
         "local_manipulation_without_trufor"),
        ("autosplice__39890__0__q95",
         "whole_image_without_probe"),
    ):
        record = records[variant_id]
        candidate = {
            "whole_image_generation": record["whole_image_generation"],
            "local_manipulation": record["local_manipulation"],
        }
        issues = coverage_issues(candidate, record["report"],
                                 record["evidence"], SERVED)
        assert expected in [item["code"] for item in issues], variant_id
