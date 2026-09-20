import json

import pytest

from scripts.audit_sft_conflicts import (
    audit_file,
    partition_records,
    rejection_reasons,
)
from scripts.build_sft_data import classify_samples, synthesize_conflict


def _record(record_id, first_source, first_support, second_source, second_support):
    def evidence(source, support, strength):
        return json.dumps({
            "evidence_name": f"{source}_evidence",
            "region": "patch_coordinates_[0, 0, 10, 10]",
            "phenomenon": f"{source} phenomenon",
            "reasoning": f"{source} reasoning",
            "strength": strength,
            "source": source,
            "support": support,
            "interpretation_text": f"{source} interpretation",
        })

    return {
        "id": record_id,
        "type": "conflict",
        "conversations": [
            {"from": "user", "value": "<image>"},
            {"from": "gpt", "value": "<call_a>"},
            {"from": "user", "value": evidence(first_source, first_support, 0.8)},
            {"from": "gpt", "value": "<call_b>"},
            {"from": "user", "value": evidence(second_source, second_support, 0.2)},
        ],
    }


def test_valid_conflict_requires_distinct_opposed_experts():
    record = _record("valid", "noise_expert", "AI-generated", "frequency_expert", "Real")
    assert rejection_reasons(record) == []


def test_same_direction_evidence_is_rejected():
    record = _record("same-direction", "jpeg_expert", "Real", "frequency_expert", "Real")
    assert "non_opposed_evidence" in rejection_reasons(record)


def test_duplicate_expert_is_rejected():
    record = _record("duplicate", "frequency_expert", "Real", "frequency_expert", "Real")
    reasons = rejection_reasons(record)
    assert "duplicate_expert_call" in reasons
    assert "non_opposed_evidence" in reasons


def test_partition_adds_audit_metadata():
    valid = _record("valid", "noise_expert", "Fake", "frequency_expert", "Real")
    invalid = _record("invalid", "noise_expert", "Real", "frequency_expert", "Real")
    kept, rejected = partition_records([valid, invalid])
    assert [item["id"] for item in kept] == ["valid"]
    assert [item["id"] for item in rejected] == ["invalid"]
    assert rejected[0]["audit"]["review_status"] == "reject"


def test_audit_file_is_idempotent(tmp_path):
    conflict_path = tmp_path / "sft_conflict.json"
    rejected_path = tmp_path / "sft_rejected.json"
    records = [
        _record("valid", "noise_expert", "AI-generated", "frequency_expert", "Real"),
        _record("invalid", "jpeg_expert", "Real", "frequency_expert", "Real"),
    ]
    conflict_path.write_text(json.dumps(records), encoding="utf-8")

    assert audit_file(conflict_path, rejected_path) == (2, 1, 1)
    assert audit_file(conflict_path, rejected_path) == (1, 1, 1)


def _raw_sample(fake_strength=0.65, real_strength=0.2):
    def expert(source, support, strength):
        return {
            "evidence_name": f"{source}_evidence",
            "phenomenon": "phenomenon",
            "reasoning": "reasoning",
            "strength": strength,
            "source": source,
            "support": support,
            "interpretation_text": "interpretation",
        }

    return {
        "image_path": "sample.jpg",
        "gt": "Real",
        "experts": {
            "freq": expert("frequency_expert", "Real", real_strength),
            "noise": expert("noise_expert", "AI-generated", fake_strength),
            "jpeg": expert("jpeg_expert", "Uncertain", 0.4),
        },
    }


def test_conflict_classifier_and_synthesizer_share_thresholds():
    sample = _raw_sample(fake_strength=0.65, real_strength=0.2)
    classified = classify_samples([sample])
    assert classified["conflict"] == [sample]
    record = synthesize_conflict(sample)
    assert "noise 专家强烈指出 AI 生成" in record["conversations"][-1]["value"]
    assert "freq 专家强烈指出真实相机" in record["conversations"][-1]["value"]
    evidence = [
        json.loads(turn["value"])
        for turn in record["conversations"]
        if turn["from"] == "user" and turn["value"].startswith("{")
    ]
    assert [item["source"] for item in evidence] == ["noise_expert", "frequency_expert"]


def test_synthesizer_rejects_missing_opposed_evidence():
    sample = _raw_sample(fake_strength=0.55, real_strength=0.2)
    with pytest.raises(ValueError, match="lacks opposed"):
        synthesize_conflict(sample)
