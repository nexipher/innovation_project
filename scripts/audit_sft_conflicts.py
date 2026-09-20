#!/usr/bin/env python3
"""Move structurally invalid conflict SFT records into a rejection set."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import date
from pathlib import Path
from typing import Any


REJECTION_RULE = "conflict_requires_independent_opposed_evidence_v1"


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


def rejection_reasons(record: dict[str, Any]) -> list[str]:
    evidence = extract_evidence(record)
    if len(evidence) < 2:
        return ["missing_or_unparseable_evidence"]

    first, second = evidence[:2]
    reasons = []
    if first.get("source") == second.get("source"):
        reasons.append("duplicate_expert_call")

    directions = {support_direction(first.get("support")), support_direction(second.get("support"))}
    if directions != {"real", "fake"}:
        reasons.append("non_opposed_evidence")

    comparable_keys = {
        "evidence_name", "phenomenon", "reasoning", "strength",
        "source", "support", "interpretation_text",
    }
    if all(first.get(key) == second.get(key) for key in comparable_keys):
        reasons.append("duplicate_evidence_payload")
    return reasons


def mark_rejected(record: dict[str, Any], reasons: list[str]) -> dict[str, Any]:
    rejected = copy.deepcopy(record)
    rejected["audit"] = {
        "review_status": "reject",
        "review_method": "automated_structural_rule",
        "rule_id": REJECTION_RULE,
        "failure_types": reasons,
        "reviewer_notes": (
            "Conflict training requires two independent experts with opposed Real and "
            "AI-generated/Fake support; this record does not meet that minimum condition."
        ),
    }
    return rejected


def partition_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept, rejected = [], []
    for record in records:
        reasons = rejection_reasons(record)
        if reasons:
            rejected.append(mark_rejected(record, reasons))
        else:
            kept.append(record)
    return kept, rejected


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


def audit_file(conflict_path: Path, rejected_path: Path) -> tuple[int, int, int]:
    records = load_json(conflict_path, [])
    previous_rejected = load_json(rejected_path, [])
    kept, newly_rejected = partition_records(records)

    merged = {}
    for record in previous_rejected:
        original = copy.deepcopy(record)
        original.pop("audit", None)
        reasons = rejection_reasons(original)
        merged[original.get("id")] = mark_rejected(
            original,
            reasons or record.get("audit", {}).get("failure_types", ["previously_rejected"]),
        )
    for record in newly_rejected:
        merged[record.get("id")] = record
    rejected = list(merged.values())

    write_json(conflict_path, kept)
    write_json(rejected_path, rejected)
    return len(records), len(kept), len(rejected)


def update_metadata(final_dir: Path) -> None:
    metadata_path = final_dir / "metadata.json"
    metadata = load_json(metadata_path, {})
    trainable_names = ("correct", "conflict", "borderline", "format")
    breakdown = {
        name: len(load_json(final_dir / f"sft_{name}.json", []))
        for name in trainable_names
    }
    rejected_count = len(load_json(final_dir / "sft_rejected.json", []))
    metadata["total"] = sum(breakdown.values())
    metadata["breakdown"] = breakdown
    metadata["rejected"] = {
        "total": rejected_count,
        "file": "sft_rejected.json",
        "included_in_total": False,
        "rule_id": REJECTION_RULE,
    }
    metadata["audited_at"] = date.today().isoformat()
    write_json(metadata_path, metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sft_data" / "train",
    )
    args = parser.parse_args()

    train_dir = args.train_dir.resolve()
    final_dir = train_dir / "final"
    results = []
    for directory in (train_dir, final_dir):
        result = audit_file(directory / "sft_conflict.json", directory / "sft_rejected.json")
        results.append((directory, result))
    update_metadata(final_dir)

    for directory, (before, kept, rejected) in results:
        print(f"{directory}: conflict {before} -> {kept}; rejected set={rejected}")


if __name__ == "__main__":
    main()
