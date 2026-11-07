"""Joint whole-image / local-manipulation verdict schema."""

import pytest

from utils.joint_verdict import extract, legal_model_fields, overall_from_fields

pytestmark = pytest.mark.core


def test_two_dimensions_are_kept_separate_and_map_to_overall():
    fields, issues = extract({
        "whole_image_generation": "AI-generated",
        "local_manipulation": "Not applicable",
        "suspicious_regions": [],
    })
    assert issues == []
    assert fields["whole_image_generation"] == "AI-generated"
    assert fields["local_manipulation"] == "Not applicable"
    assert fields["overall_from_dimensions"] == "Fake"
    assert overall_from_fields("Real", "Not tampered") == "Real"
    assert overall_from_fields("Real", "Tampered") == "Fake"
    assert overall_from_fields("Uncertain", "Tampered") == "Fake"
    assert overall_from_fields("Real", "Uncertain") == "Uncertain"
    # Not applicable answers the local question only for a generated whole
    # image; on a Real image it is not a clean local measurement.
    assert overall_from_fields("Real", "Not applicable") == "Uncertain"
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
    assert fields["overall_from_dimensions"] == "Fake"

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
