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
        {"code": "overall_uncertain_with_fake_dimension"},
        {"code": "overall_real_with_unmeasured_whole"},
    ])
    assert "局部问题未测时应写 Uncertain" in note
    assert "整图生成分项应写 Uncertain" in note
    assert "总体必须是 Fake" in note
    assert "局部“未发现篡改”只排除了一个方向" in note
    assert "不要伪造专家分数" in note


def test_local_tampered_with_only_candidate_wording_is_actionable():
    candidate = {"verdict": "Fake", "whole_image_generation": "Real",
                 "local_manipulation": "Tampered"}
    issues = coverage_issues(
        candidate, "TruFor 仅提供局部篡改候选，不构成证明。",
        _evidence("trufor_expert"), SERVED)
    assert any(item["code"] == "local_tampered_candidate_only"
               for item in actionable(issues))
    supported = coverage_issues(
        candidate, "TruFor 高响应且高可靠性，支持该区域存在内容变化。",
        _evidence("trufor_expert"), SERVED)
    assert not any(item["code"] == "local_tampered_candidate_only"
                   for item in supported)
    # "非证明" is an epistemic caveat, not "only a candidate"; the broad
    # regex used to turn a supported local Tampered answer into Uncertain.
    caveat = coverage_issues(
        candidate, "TruFor 高响应但非证明，具体内容变化仍需结合画面判断。",
        _evidence("trufor_expert"), SERVED)
    assert not any(item["code"] == "local_tampered_candidate_only"
                   for item in caveat)
    localised = coverage_issues(
        {**candidate, "suspicious_regions": [
            {"bbox": [100, 100, 500, 500], "reason": "同区域响应"}]},
        "TruFor 仅提供候选，但已定位该区域并说明内容。",
        _evidence("trufor_expert"), SERVED)
    assert not any(item["code"] == "local_tampered_candidate_only"
                   for item in localised)


def test_not_applicable_is_removed_for_all_whole_labels():
    for whole in ("Real", "AI-generated"):
        issues = coverage_issues(
            {"verdict": "Fake" if whole == "AI-generated" else "Real",
             "whole_image_generation": whole,
             "local_manipulation": "Not applicable"}, "", [], [])
        assert any(item["code"] == "local_not_applicable_removed"
                   for item in actionable(issues)), whole
    note = coverage_note([{"code": "local_not_applicable_removed"}])
    assert "局部编辑是两个独立问题" in note
    assert "Tampered、Not tampered 或 Uncertain" in note


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


def test_overall_label_consistency_with_dimensions():
    real_with_tamper = coverage_issues(
        {"verdict": "Real", "whole_image_generation": "Real",
         "local_manipulation": "Tampered"}, "", [], [])
    assert any(item["code"] == "overall_real_with_fake_dimension"
               for item in actionable(real_with_tamper))

    fake_without_support = coverage_issues(
        {"verdict": "Fake", "whole_image_generation": "Real",
         "local_manipulation": "Not tampered"}, "", [], [])
    assert any(item["code"] == "overall_fake_without_fake_dimension"
               for item in actionable(fake_without_support))

    uncertain_with_tamper = coverage_issues(
        {"verdict": "Uncertain", "whole_image_generation": "Real",
         "local_manipulation": "Tampered"}, "", [], [])
    assert any(item["code"] == "overall_uncertain_with_fake_dimension"
               for item in uncertain_with_tamper)
    assert any(item["code"] == "overall_uncertain_with_fake_dimension"
               for item in actionable(uncertain_with_tamper))


def test_overall_real_with_unmeasured_whole_is_actionable():
    for whole in ("Real", "Uncertain"):
        issues = coverage_issues(
            {"verdict": "Real", "whole_image_generation": whole,
             "local_manipulation": "Not tampered"},
            "整图生成尚未测量，无法排除 AI 生成。",
            _evidence("trufor_expert"), SERVED)
        assert any(item["code"] == "overall_real_with_unmeasured_whole"
                   for item in actionable(issues)), whole
    # A measured whole-image answer, or a local Tampered answer, does not
    # trigger this check.
    measured = coverage_issues(
        {"verdict": "Real", "whole_image_generation": "Real",
         "local_manipulation": "Not tampered"}, "",
        _evidence("probe_dino_expert", "trufor_expert"), SERVED)
    assert not any(item["code"] == "overall_real_with_unmeasured_whole"
                   for item in measured)
    local_fake = coverage_issues(
        {"verdict": "Fake", "whole_image_generation": "Uncertain",
         "local_manipulation": "Tampered"}, "",
        _evidence("trufor_expert"), SERVED)
    assert not any(item["code"] == "overall_real_with_unmeasured_whole"
                   for item in local_fake)
    # A failed PROBE cannot be called again, so the correction must not tell
    # the model to retry it; the Real verdict is left for human review.
    probe_failed = coverage_issues(
        {"verdict": "Real", "whole_image_generation": "Uncertain",
         "local_manipulation": "Not tampered"}, "",
        [{"source": "probe_dino_expert", "status": "failed"},
         {"source": "trufor_expert", "status": "measured"}], SERVED)
    assert not any(item["code"] == "overall_real_with_unmeasured_whole"
                   for item in probe_failed)


def test_local_uncertain_without_trufor_forces_measurement():
    issues = coverage_issues(
        {"verdict": "Fake", "whole_image_generation": "AI-generated",
         "local_manipulation": "Uncertain"},
        "整图 AI 生成，局部不确定。",
        _evidence("probe_dino_expert"), SERVED)
    assert any(item["code"] == "local_uncertain_without_trufor"
               for item in actionable(issues))


def test_generated_whole_does_not_change_the_local_answer():
    evidence = [
        {"source": "probe_dino_expert", "status": "measured",
         "evidence_id": "E-probe", "raw_metric": 0.99},
        {"source": "trufor_expert", "status": "measured",
         "evidence_id": "E-trufor", "raw_metric": 0.35},
    ]
    # A generated whole image may still have a local answer; low TruFor
    # supports Not tampered, and it must not be converted to Not applicable.
    for local in ("Not tampered", "Uncertain", "Tampered"):
        issues = coverage_issues(
            {"verdict": "Fake", "whole_image_generation": "AI-generated",
             "local_manipulation": local},
            "", evidence, SERVED)
        assert not any(item["code"] == "local_not_applicable_removed"
                       for item in issues)
        assert not any(item["code"].startswith("generated_whole")
                       for item in issues)
    # Whole=AI-generated + local Uncertain is still overall Fake (no issue).
    issues = coverage_issues(
        {"verdict": "Fake", "whole_image_generation": "AI-generated",
         "local_manipulation": "Uncertain"},
        "", evidence, SERVED)
    assert not any(item["code"] == "overall_uncertain_with_fake_dimension"
                   for item in issues)


def test_local_candidate_note_keeps_generated_overall_fake():
    note = coverage_note([{"code": "local_tampered_candidate_only",
                           "phrase": "仅提供候选"}])
    assert "总体必须保持 Fake" in note
    assert "不能改成 Uncertain" in note
