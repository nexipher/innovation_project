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


def certainty_note(issues) -> str:
    """One targeted correction for an invalid certainty field.

    It never supplies a label and never converts a value: the model decides
    whether to keep its overall verdict (and write null) or change the label
    (and give a certainty for Real/Fake).
    """
    details = "；".join(issues or [])
    return (
        "[System: 结论字段检查：certainty 与总体标签不一致。"
        f"{details}。\n"
        "- 总体 verdict=Uncertain 时，certainty 必须为 null；High/Medium/Low "
        "只用于 Real/Fake，不能表示“对弃权的把握”。\n"
        "- 请保留你的证据判断，只重新输出一个 <verdict>（不要再调用专家）："
        "若仍是 Uncertain，certainty 写 null；若改为 Real/Fake，再按该结论"
        "给出 High/Medium/Low。\n"
        "- 不要为了让字段合法而改变证据结论；也不要另写一个 confidence。]"
    )
