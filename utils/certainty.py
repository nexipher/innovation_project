"""Qualitative certainty for the overall verdict (certainty_v1).

The new schema replaces the model's numeric `confidence` with a qualitative
`certainty`:

* Real / Fake  -> "High" | "Medium" | "Low"
* Uncertain    -> null

The value is the model's own judgement about evidence quality, applicability
and unresolved issues.  It is never derived from expert scores, score bands,
tool counts or a numeric formula, and old numeric confidence is never
converted into it.
"""

from __future__ import annotations

from typing import Optional, Tuple

CERTAINTY_VALUES = ("High", "Medium", "Low")
_CANONICAL = {value.lower(): value for value in CERTAINTY_VALUES}
VERDICT_SCHEMA = "certainty_v1"


def canonical_certainty(value) -> Tuple[Optional[str], Optional[str]]:
    """Return (canonical value, error).  Numbers are not accepted."""
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return None, f"certainty must be High/Medium/Low or null, not {value!r}"
    if not isinstance(value, str):
        return None, f"certainty must be High/Medium/Low or null, not {value!r}"
    canonical = _CANONICAL.get(value.strip().lower())
    if canonical is None:
        return None, f"illegal certainty: {value!r}"
    return canonical, None


def certainty_issues(candidate: dict, overall_label: Optional[str]) -> Tuple[Optional[str], list]:
    """Validate certainty against the overall label.

    Returns the canonical certainty (None for Uncertain/invalid) and a list of
    issue strings.  It does not invent a value for a missing field.
    """
    candidate = candidate or {}
    issues = []
    if "confidence" in candidate:
        issues.append(
            "certainty_v1 does not accept numeric confidence; use certainty "
            "High/Medium/Low for Real/Fake and null for Uncertain")
    if "certainty" not in candidate:
        issues.append(
            "certainty is missing; use High/Medium/Low for Real/Fake and null "
            "for Uncertain")
        return None, issues
    value = candidate.get("certainty")
    if overall_label == "Uncertain":
        if value is not None:
            issues.append("certainty must be null when the overall verdict is Uncertain")
        return None, issues
    canonical, error = canonical_certainty(value)
    if error:
        issues.append(error)
        return None, issues
    return canonical, issues
