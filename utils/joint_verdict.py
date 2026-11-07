"""Joint whole-image generation / local-manipulation verdict fields.

The legacy ``verdict`` field answers one question.  The dual-expert loop needs
two answers that must not overwrite each other:

* ``whole_image_generation``: Real / AI-generated / Uncertain
* ``local_manipulation``: Tampered / Not tampered / Uncertain / Not applicable
* ``suspicious_regions``: bbox + reason + evidence ids, in original-image
  normalized [0,1000] coordinates.

This module only normalises and validates what the model wrote.  It never
invents a field or rewrites a label; callers can record the issues and keep the
model's own text.
"""

from __future__ import annotations

from typing import Iterable, Optional

WHOLE_IMAGE_VALUES = ("Real", "AI-generated", "Uncertain")
LOCAL_VALUES = ("Tampered", "Not tampered", "Uncertain", "Not applicable")

WHOLE_ALIASES = {
    "real": "Real",
    "ai-generated": "AI-generated",
    "ai generated": "AI-generated",
    "aigenerated": "AI-generated",
    "generated": "AI-generated",
    "uncertain": "Uncertain",
}
LOCAL_ALIASES = {
    "tampered": "Tampered",
    "manipulated": "Tampered",
    "not tampered": "Not tampered",
    "not-tampered": "Not tampered",
    "no tampering": "Not tampered",
    "untampered": "Not tampered",
    "uncertain": "Uncertain",
    "not applicable": "Not applicable",
    "not-applicable": "Not applicable",
    "n/a": "Not applicable",
    "none": "Not applicable",
}


def _canonical(value, aliases):
    if not isinstance(value, str):
        return None
    return aliases.get(value.strip().lower())


def overall_from_fields(whole: Optional[str], local: Optional[str]) -> str:
    """The legacy overall label implied by the two dimensions.

    ``Not applicable`` says local manipulation is not a meaningful question
    because the whole image is generated; it is not a substitute for a clean
    local measurement on a Real image.
    """
    if whole == "AI-generated" or local == "Tampered":
        return "Fake"
    if local == "Not applicable" and whole != "AI-generated":
        return "Uncertain"
    if whole == "Real" and local == "Not tampered":
        return "Real"
    return "Uncertain"


def _normalise_region(region, evidence_ids: Optional[Iterable[str]]):
    issues = []
    if not isinstance(region, dict):
        return None, ["suspicious_regions entry is not an object"]
    bbox = region.get("bbox")
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
            or any(not isinstance(value, (int, float)) for value in bbox)):
        return None, ["suspicious_regions.bbox must be four numbers"]
    values = [int(round(float(value))) for value in bbox]
    ymin, xmin, ymax, xmax = values
    if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
        return None, [f"suspicious_regions.bbox out of [0,1000] or empty: {values}"]
    reason = region.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        issues.append("suspicious_regions.reason is empty")
        reason = ""
    evidence = region.get("evidence")
    if evidence is None:
        evidence = []
    if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
        issues.append("suspicious_regions.evidence must be a list of strings")
        evidence = []
    if not evidence:
        # A region without a supporting token is an observation, not a
        # supported local finding; the report must carry the evidence basis.
        issues.append("suspicious_regions.evidence is empty")
    if evidence_ids is not None:
        known = set(evidence_ids)
        unknown = [item for item in evidence if item not in known]
        if unknown:
            issues.append(f"suspicious_regions cites unknown evidence ids: {unknown}")
    return {
        "bbox": values,
        "reason": reason.strip(),
        "evidence": list(evidence),
    }, issues


def extract(candidate: dict, evidence_ids: Optional[Iterable[str]] = None):
    """Return (legal joint fields, issues) from a model verdict JSON object.

    Missing fields are *not* invented.  A partial object returns what was
    present plus an issue, which is the caller's signal to record a missing
    joint answer rather than turn the old single label into two claims.
    """
    if not isinstance(candidate, dict):
        return {}, ["verdict is not an object"]
    fields = {}
    issues = []
    whole = candidate.get("whole_image_generation")
    local = candidate.get("local_manipulation")
    if whole is not None:
        canonical = _canonical(whole, WHOLE_ALIASES)
        if canonical is None:
            issues.append(f"illegal whole_image_generation: {whole!r}")
        else:
            fields["whole_image_generation"] = canonical
    if local is not None:
        canonical = _canonical(local, LOCAL_ALIASES)
        if canonical is None:
            issues.append(f"illegal local_manipulation: {local!r}")
        else:
            fields["local_manipulation"] = canonical
    if "suspicious_regions" in candidate:
        regions = candidate.get("suspicious_regions")
        if not isinstance(regions, list):
            issues.append("suspicious_regions must be a list")
        else:
            legal = []
            for index, region in enumerate(regions):
                normalised, region_issues = _normalise_region(region, evidence_ids)
                issues.extend(f"region[{index}]: {issue}" for issue in region_issues)
                if normalised is not None:
                    legal.append(normalised)
            fields["suspicious_regions"] = legal
    if ("whole_image_generation" in candidate) != ("local_manipulation" in candidate):
        issues.append("joint verdict is partial: both dimensions must be present")
    if (fields.get("local_manipulation") == "Not applicable"
            and fields.get("whole_image_generation") != "AI-generated"):
        issues.append(
            "local_manipulation = Not applicable requires "
            "whole_image_generation = AI-generated")
    if fields.get("whole_image_generation") and fields.get("local_manipulation"):
        fields["overall_from_dimensions"] = overall_from_fields(
            fields["whole_image_generation"], fields["local_manipulation"])
    return fields, issues


def legal_model_fields(candidate: dict, evidence_ids: Optional[Iterable[str]] = None):
    """Fields to preserve verbatim in the supervised answer.

    Only fields the model actually wrote, and only when they are legal.  The
    converter must not fill a missing field with a default.
    """
    fields, _issues = extract(candidate, evidence_ids)
    out = {}
    report = candidate.get("report") if isinstance(candidate, dict) else None
    if isinstance(report, str) and report.strip():
        out["report"] = report.strip()
    primary = candidate.get("primary_evidence") if isinstance(candidate, dict) else None
    if (isinstance(primary, list)
            and all(isinstance(item, str) for item in primary)):
        out["primary_evidence"] = list(primary)
    out.update(fields)
    return out
