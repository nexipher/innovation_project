import json

from scripts.audit_sft_correct import (
    KNOWN_INVALID_IDS,
    audit_final,
    correct_rejection_reasons,
    correct_soft_failures,
    partition_correct,
)


def _evidence(source, support, strength, region="patch_coordinates_[100, 100, 200, 200]",
              reasoning="reasoning"):
    return json.dumps({
        "evidence_name": f"{source}_evidence",
        "region": region,
        "phenomenon": "phenomenon",
        "reasoning": reasoning,
        "strength": strength,
        "source": source,
        "support": support,
        "interpretation_text": "interpretation",
    })


def _record(record_id, evidence_json_list, verdict="Fake", confidence=0.8):
    conversations = [{"from": "user", "value": "<image>"}]
    for item in evidence_json_list:
        conversations.append({"from": "gpt", "value": "<call>"})
        conversations.append({"from": "user", "value": item})
    return {
        "id": record_id,
        "type": "correct_filtered",
        "conversations": conversations,
        "final_verdict": {"verdict": verdict, "confidence": confidence},
    }


def test_clean_record_reaches_regenerate_not_reject():
    record = _record("clean", [_evidence("noise_expert", "Real", 0.2)])
    assert correct_rejection_reasons(record) == []
    kept, rejected = partition_correct([record])
    assert rejected == []
    assert kept[0]["audit"]["review_status"] == "regenerate"


def test_duplicate_evidence_is_rejected():
    record = _record("dup", [
        _evidence("noise_expert", "Real", 0.23),
        _evidence("noise_expert", "Real", 0.23),
    ])
    assert "duplicate_evidence" in correct_rejection_reasons(record)
    kept, rejected = partition_correct([record])
    assert kept == []
    assert rejected[0]["audit"]["failure_types"] == ["duplicate_evidence"]


def test_same_expert_at_different_regions_is_not_a_duplicate():
    record = _record("two-regions", [
        _evidence("noise_expert", "Real", 0.23, region="patch_coordinates_[10, 10, 50, 50]"),
        _evidence("noise_expert", "Real", 0.23, region="patch_coordinates_[60, 60, 100, 100]"),
    ])
    assert "duplicate_evidence" not in correct_rejection_reasons(record)


def test_coordinate_drift_is_rejected():
    record = _record("drift", [
        _evidence("jpeg_expert", "Real", 0.7, region="patch_coordinates_[0, 0, 100, 100]"),
        _evidence("jpeg_expert", "Real", 0.6, region="patch_coordinates_[0, 0, 40, 40]"),
        _evidence("jpeg_expert", "Real", 0.5, region="patch_coordinates_[0, 0, 10, 10]"),
    ])
    assert "coordinate_drift" in correct_rejection_reasons(record)


def test_known_invalid_id_is_rejected():
    record = _record(next(iter(KNOWN_INVALID_IDS)), [_evidence("noise_expert", "Real", 0.2)])
    assert correct_rejection_reasons(record) == ["confirmed_invalid_by_review"]


def test_contaminated_low_strength_reasoning_is_flagged():
    record = _record("contaminated", [_evidence(
        "frequency_expert", "Real", 0.011,
        reasoning=(
            "This spectral pattern is mathematically consistent with upsampling / "
            "deconvolution grid artifacts common in GAN and Diffusion-based image synthesis."
        ),
    )])
    assert "contaminated_reasoning_direction" in correct_soft_failures(record)


def test_single_weak_evidence_high_confidence_is_flagged():
    record = _record("overconf", [_evidence("noise_expert", "Real", 0.23)], confidence=0.99)
    assert "single_evidence_high_confidence" in correct_soft_failures(record)


def test_verdict_without_matching_evidence_direction_is_flagged():
    record = _record("mismatch", [_evidence("jpeg_expert", "Real", 0.3)], verdict="Fake")
    assert "verdict_fake_without_fake_evidence" in correct_soft_failures(record)


def _write_split(path, records):
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


def test_audit_final_is_idempotent(tmp_path):
    train_dir = tmp_path / "train"
    final_dir = train_dir / "final"
    final_dir.mkdir(parents=True)

    valid = _record("valid", [_evidence("noise_expert", "Real", 0.2)])
    duplicate = _record("duplicate", [
        _evidence("noise_expert", "Real", 0.23),
        _evidence("noise_expert", "Real", 0.23),
    ])
    _write_split(final_dir / "sft_correct.json", [valid, duplicate])
    _write_split(final_dir / "sft_borderline.json", [{
        "id": "b1", "type": "borderline", "conversations": [],
        "final_verdict": {"verdict": "Real", "confidence": 0.65},
    }])
    _write_split(final_dir / "sft_format.json", [{
        "id": "f1", "type": "format", "conversations": [],
        "final_verdict": {"verdict": "Fake", "confidence": 0.9},
    }])
    _write_split(final_dir / "sft_conflict.json", [])
    _write_split(final_dir / "sft_rejected.json", [])
    _write_split(train_dir / "sft_correct.json", [])
    _write_split(final_dir / "metadata.json", {})

    audit_final(train_dir)
    first = (final_dir / "sft_correct.json").read_text(encoding="utf-8")
    first_rejected = (final_dir / "sft_rejected.json").read_text(encoding="utf-8")

    audit_final(train_dir)
    assert (final_dir / "sft_correct.json").read_text(encoding="utf-8") == first
    assert (final_dir / "sft_rejected.json").read_text(encoding="utf-8") == first_rejected

    kept = json.loads(first)
    rejected = json.loads(first_rejected)
    assert [record["id"] for record in kept] == ["valid"]
    assert kept[0]["audit"]["review_status"] == "regenerate"
    assert [record["id"] for record in rejected] == ["duplicate"]

    borderline = json.loads((final_dir / "sft_borderline.json").read_text(encoding="utf-8"))
    assert borderline[0]["audit"]["review_status"] == "regenerate"
    format_split = json.loads((final_dir / "sft_format.json").read_text(encoding="utf-8"))
    assert format_split[0]["audit"]["review_status"] == "format_only"

    metadata = json.loads((final_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["breakdown"]["correct"] == 1
    assert metadata["rejected"]["total"] == 1
