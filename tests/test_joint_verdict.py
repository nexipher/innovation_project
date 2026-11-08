"""Joint whole-image / local-manipulation verdict schema."""

import pytest

from utils.joint_verdict import extract, legal_model_fields

pytestmark = pytest.mark.core


def test_two_dimensions_are_kept_separate_without_a_mechanical_overall():
    fields, issues = extract({
        "whole_image_generation": "AI-generated",
        "local_manipulation": "Not applicable",
        "suspicious_regions": [],
    })
    assert issues == []
    assert fields["whole_image_generation"] == "AI-generated"
    assert fields["local_manipulation"] == "Not applicable"
    # The overall verdict belongs to the model; the schema module must not
    # derive it mechanically from the two dimensions.
    assert "overall_from_dimensions" not in fields
    _fields, issues = extract({
        "whole_image_generation": "Real",
        "local_manipulation": "Not applicable",
    })
    assert any("Not applicable" in issue for issue in issues)


def test_regions_require_bounds_reason_and_real_evidence():
    fields, issues = extract({
        "whole_image_generation": "Real",
        "local_manipulation": "Tampered",
        "suspicious_regions": [
            {"bbox": [100, 200, 300, 400], "reason": "copy-move seam",
             "evidence": ["E-abc"]},
        ],
    }, evidence_ids=["E-abc"])
    assert issues == []
    assert fields["suspicious_regions"][0]["bbox"] == [100, 200, 300, 400]
    assert fields["suspicious_regions"][0]["scope"] == "local_candidate"

    full, _ = extract({
        "whole_image_generation": "AI-generated",
        "local_manipulation": "Uncertain",
        "suspicious_regions": [
            {"bbox": [0, 0, 1000, 1000], "reason": "whole range",
             "evidence": ["E-abc"]},
        ],
    }, evidence_ids=["E-abc"])
    assert full["suspicious_regions"][0]["scope"] == "global_range"

    _fields, issues = extract({
        "whole_image_generation": "Real",
        "local_manipulation": "Tampered",
        "suspicious_regions": [
            {"bbox": [0, 0, 10, 10], "reason": "", "evidence": ["E-missing"]},
        ],
    }, evidence_ids=["E-abc"])
    assert any("reason" in issue for issue in issues)
    assert any("unknown evidence" in issue for issue in issues)

    _fields, issues = extract({
        "whole_image_generation": "Real",
        "local_manipulation": "Tampered",
        "suspicious_regions": [{"bbox": [0, 0, 0, 10], "reason": "x"}],
    })
    assert any("bbox" in issue for issue in issues)

    _fields, issues = extract({
        "whole_image_generation": "Real",
        "local_manipulation": "Tampered",
        "suspicious_regions": [
            {"bbox": [10, 10, 20, 20], "reason": "seam", "evidence": []},
        ],
    })
    assert any("evidence is empty" in issue for issue in issues)


def test_missing_fields_are_not_invented():
    fields, issues = extract({"verdict": "Fake", "report": "x"})
    assert fields == {}
    assert "partial" not in " ".join(issues)
    # legal_model_fields preserves only what the model wrote, plus report.
    preserved = legal_model_fields({"report": "x", "primary_evidence": ["E-1"]})
    assert preserved == {"report": "x", "primary_evidence": ["E-1"]}
