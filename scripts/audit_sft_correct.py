#!/usr/bin/env python3
"""Audit the correct split and stamp dispositions on all legacy SFT splits.

G0 wrap-up (plan.md §4.7):

  1. Hard-reject structurally invalid *correct* traces into the rejection set:
       - RC1 duplicate evidence payload in one conversation;
       - RC2 coordinate-space drift (repeatedly shrinking bbox areas);
       - RC3 explicitly confirmed invalid session ids.
  2. Tag the remaining *correct* records with review_status=regenerate and a
     list of failure types (contaminated reasoning, single weak evidence with
     high confidence, verdict/evidence direction mismatch).
  3. Stamp explicit dispositions on the other legacy splits:
       - conflict (not already rejected) -> regenerate (awaiting expert calibration);
       - borderline -> regenerate (fixed-template synthetic);
       - format -> format_only (structure training only, facts not verified);
       - the superseded synthetic correct file under train/ -> superseded.
  4. Refresh metadata.json with disposition counts.

The script is idempotent: previously rejected records keep their audit block,
and re-running produces identical outputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

CORRECT_RULE = "correct_trace_integrity_v1"
DISPOSITION_RULE = "sft_disposition_v1"

# Sessions confirmed invalid by manual review (plan.md §4.7).
KNOWN_INVALID_IDS = {
    "session_20260721_111418_00eb3be4-a0cb-4682-96ea-0a074cd7efaa",  # coordinate re-conversion, duplicate evidence, fake convergence
    "session_20260721_111810_080862af-5aba-41b0-80ed-dab7b7683807",  # internal expert contradiction, reversed evidence, 0.99 uncalibrated
}

# Old (pre-fix) expert reasoning markers that claim generative artifacts
# regardless of the measured strength.
OLD_REASONING_MARKERS = (
    "consistent with upsampling / deconvolution grid artifacts",
    "Splicing, inpainting, or AI-based local editing disrupts this homogeneity",
    "These are classic digital forgery markers",
    "Multi-scale approach catches artifacts",
)


def support_direction(value: Any) -> str | None:
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized == "real":
        return "real"
    if normalized in {"fake", "ai-generated", "ai generated"}:
        return "fake"
    return None


def extract_evidence(record: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for turn in record.get("conversations", []):
        if turn.get("from") != "user":
            continue
        value = turn.get("value")
        if not isinstance(value, str) or not value.lstrip().startswith("{"):
            continue
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "source" in parsed and "support" in parsed:
            evidence.append(parsed)
    return evidence


def region_area(region: Any) -> int | None:
    match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", str(region))
    if not match:
        return None
    y1, x1, y2, x2 = (int(g) for g in match.groups())
    return max(1, y2 - y1) * max(1, x2 - x1)


def correct_rejection_reasons(record: dict[str, Any]) -> list[str]:
    """Hard structural failures that force the record into the rejection set."""
    if record.get("id") in KNOWN_INVALID_IDS:
        return ["confirmed_invalid_by_review"]

    evidence = extract_evidence(record)
    reasons = []

    keys = [
        (e.get("source"), round(float(e.get("strength", 0.0)), 4), e.get("region", ""))
        for e in evidence
    ]
    if len(keys) > len(set(keys)):
        reasons.append("duplicate_evidence")

    areas = [a for a in (region_area(e.get("region", "")) for e in evidence) if a]
    if len(areas) >= 3 and all(areas[i + 1] <= areas[i] * 0.5 for i in range(len(areas) - 1)):
        reasons.append("coordinate_drift")

    return reasons


def correct_soft_failures(record: dict[str, Any]) -> list[str]:
    """Quality flags that require regeneration but do not force rejection."""
    evidence = extract_evidence(record)
    failures = []

    if any(
        float(e.get("strength", 0.5)) < 0.3
        and any(marker in str(e.get("reasoning", "")) for marker in OLD_REASONING_MARKERS)
        for e in evidence
    ):
        failures.append("contaminated_reasoning_direction")

    verdict = (record.get("final_verdict") or {}).get("verdict", "")
    directions = {support_direction(e.get("support")) for e in evidence}
    if evidence and verdict == "Fake" and "fake" not in directions:
        failures.append("verdict_fake_without_fake_evidence")
    if evidence and verdict == "Real" and "real" not in directions:
        failures.append("verdict_real_without_real_evidence")

    confidence = float((record.get("final_verdict") or {}).get("confidence", 0.0))
    if len(evidence) == 1 and confidence >= 0.95:
        failures.append("single_evidence_high_confidence")

    return failures or ["pre_fix_expert_reasoning"]


def mark(record: dict[str, Any], status: str, rule_id: str, failures: list[str], notes: str) -> dict[str, Any]:
    marked = copy.deepcopy(record)
    marked["audit"] = {
        "review_status": status,
        "review_method": "automated_structural_rule",
        "rule_id": rule_id,
        "failure_types": failures,
        "reviewer_notes": notes,
    }
    return marked


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def partition_correct(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, rejected = [], []
    for record in records:
        reasons = correct_rejection_reasons(record)
        if reasons:
            rejected.append(
                mark(
                    record,
                    "reject",
                    CORRECT_RULE,
                    reasons,
                    "Trace is structurally invalid (duplicate evidence, coordinate drift or "
                    "confirmed manual finding); never use for training, keep for regression.",
                )
            )
            continue
        soft = correct_soft_failures(record)
        kept.append(
            mark(
                record,
                "regenerate",
                CORRECT_RULE,
                soft,
                "Generated on 2026-07-21 with pre-fix expert reasoning. Regenerate with the "
                "corrected experts under the G1 protocol before any training use.",
            )
        )
    return kept, rejected


def merge_rejected(previous: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve previously rejected records verbatim; add new ones by id."""
    merged: dict[str, dict[str, Any]] = {}
    for record in previous:
        merged[record.get("id")] = record
    for record in new:
        merged[record.get("id")] = record
    return list(merged.values())


def stamp_split(path: Path, status: str, failures: list[str], notes: str) -> int:
    """Add a disposition audit block to records that do not have one yet."""
    records = load_json(path, [])
    if not records:
        return 0
    stamped = []
    changed = 0
    for record in records:
        if "audit" in record:
            stamped.append(record)
            continue
        stamped.append(mark(record, status, DISPOSITION_RULE, list(failures), notes))
        changed += 1
    write_json(path, stamped)
    return changed


def update_metadata(final_dir: Path) -> dict[str, Any]:
    metadata_path = final_dir / "metadata.json"
    metadata = load_json(metadata_path, {})
    splits = ("correct", "conflict", "borderline", "format")

    counts: dict[str, int] = {}
    dispositions: dict[str, int] = {}
    for name in splits:
        records = load_json(final_dir / f"sft_{name}.json", [])
        counts[name] = len(records)
        for record in records:
            status = (record.get("audit") or {}).get("review_status", "unreviewed")
            dispositions[status] = dispositions.get(status, 0) + 1

    rejected = load_json(final_dir / "sft_rejected.json", [])
    metadata["total"] = sum(counts.values())
    metadata["breakdown"] = counts
    metadata["rejected"] = {
        "total": len(rejected),
        "file": "sft_rejected.json",
        "included_in_total": False,
        "rule_ids": sorted({(r.get("audit") or {}).get("rule_id", "?") for r in rejected}),
    }
    metadata["dispositions"] = dispositions
    metadata["audited_at"] = date.today().isoformat()
    write_json(metadata_path, metadata)
    return metadata


def audit_final(train_dir: Path) -> None:
    final_dir = train_dir / "final"

    # 1. correct hard-reject pass
    correct_path = final_dir / "sft_correct.json"
    rejected_path = final_dir / "sft_rejected.json"
    correct = load_json(correct_path, [])
    previous_rejected = load_json(rejected_path, [])

    kept, newly_rejected = partition_correct(correct)
    rejected = merge_rejected(previous_rejected, newly_rejected)

    write_json(correct_path, kept)
    write_json(rejected_path, rejected)
    print(f"correct: {len(correct)} -> kept {len(kept)} (regenerate), "
          f"newly rejected {len(newly_rejected)}; rejected set={len(rejected)}")

    # 2. disposition stamps on the other legacy splits
    counts = {
        "borderline": stamp_split(
            final_dir / "sft_borderline.json",
            "regenerate",
            ["fixed_template_synthetic", "verdict_copied_from_gt", "random_confidence"],
            "Fixed-template synthetic samples; not usable for reasoning, routing or confidence "
            "training. Regenerate after expert calibration (G2).",
        ),
        "format": stamp_split(
            final_dir / "sft_format.json",
            "format_only",
            ["content_not_verified"],
            "Structure-only training split; factual conclusions are not verified and must not be "
            "used for factual reasoning supervision.",
        ),
        "conflict": stamp_split(
            final_dir / "sft_conflict.json",
            "regenerate",
            ["pending_expert_calibration"],
            "Passed structural admission only; numbers, regions and content still require expert "
            "calibration before training use.",
        ),
    }
    for name, changed in counts.items():
        print(f"{name}: stamped {changed} records")

    # 3. superseded synthetic correct file under train/
    superseded_path = train_dir / "sft_correct.json"
    superseded = load_json(superseded_path, [])
    if superseded:
        write_json(
            superseded_path,
            [
                record if "audit" in record else mark(
                    record,
                    "superseded",
                    DISPOSITION_RULE,
                    ["synthetic_superseded_by_a_line"],
                    "Synthetic correct split superseded by the A-line filtered set used in final/; "
                    "kept for pipeline diagnostics only.",
                )
                for record in superseded
            ],
        )
        print(f"train/sft_correct.json: stamped {len(superseded)} records as superseded")

    metadata = update_metadata(final_dir)
    print(f"metadata: total={metadata['total']} dispositions={metadata['dispositions']} "
          f"rejected={metadata['rejected']['total']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sft_data" / "train",
    )
    args = parser.parse_args()
    audit_final(args.train_dir.resolve())


if __name__ == "__main__":
    main()
