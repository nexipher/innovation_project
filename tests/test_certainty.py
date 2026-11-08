"""certainty_v1: qualitative certainty replaces numeric confidence."""

import pytest

from utils.certainty import (
    CERTAINTY_VALUES,
    VERDICT_SCHEMA,
    canonical_certainty,
    certainty_issues,
)

pytestmark = pytest.mark.core


def test_canonical_values_accept_case_insensitive_but_not_numbers():
    assert canonical_certainty("High") == ("High", None)
    assert canonical_certainty("medium") == ("Medium", None)
    assert canonical_certainty("LOW") == ("Low", None)
    assert canonical_certainty(0.95)[0] is None
    assert canonical_certainty("Very sure")[0] is None


def test_real_and_fake_require_high_medium_or_low():
    value, issues = certainty_issues(
        {"verdict": "Fake", "certainty": "High"}, "Fake")
    assert value == "High" and issues == []
    value, issues = certainty_issues({"verdict": "Fake"}, "Fake")
    assert value is None and any("missing" in issue for issue in issues)
    value, issues = certainty_issues(
        {"verdict": "Fake", "certainty": 0.9}, "Fake")
    assert value is None and issues


def test_uncertain_requires_null_and_numeric_confidence_is_rejected():
    value, issues = certainty_issues(
        {"verdict": "Uncertain", "certainty": None}, "Uncertain")
    assert value is None and issues == []
    value, issues = certainty_issues(
        {"verdict": "Uncertain", "certainty": "High"}, "Uncertain")
    assert value is None and any("null" in issue for issue in issues)
    value, issues = certainty_issues(
        {"verdict": "Fake", "certainty": "High", "confidence": 0.9}, "Fake")
    assert value == "High"
    assert any("numeric confidence" in issue for issue in issues)


def test_schema_name_is_stable():
    assert VERDICT_SCHEMA == "certainty_v1"
    assert CERTAINTY_VALUES == ("High", "Medium", "Low")
