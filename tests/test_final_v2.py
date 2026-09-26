"""Tests for the final_v2 schema and validator (G4-d)."""

import json
import re

import pytest

from scripts.build_final_v2 import assert_not_frozen, bucket_for
from utils.final_v2 import (
    SECTIONS,
    parse_sections,
    render_answer,
    render_sample,
    validate_dataset,
    validate_sample,
)

APPLICABILITY = {
    "noise": {"png": 0.845, "jpeg_q70": 0.772},
    "jpeg": {"png": 0.972, "jpeg_q70": 0.431},
    "frequency_v2": {"png": 0.556},
}

SPLIT = {"hash": "h", "sources": {
    "ADM/x": {"split": "train", "resolution": [48, 64], "generator": "ADM"},
    "ADM/tested": {"split": "test", "resolution": [48, 64], "generator": "ADM"},
}}


def _token(evidence_id="E-1", source="noise_expert", support="AI-generated",
           likelihood=0.9, raw=1.0, region=(2, 2, 40, 40), scope="global"):
    direction = "AI-generated" if likelihood >= 0.6 else ("Real" if likelihood <= 0.4 else "Uncertain")
    return {
        "evidence_id": evidence_id,
        "source": source,
        "support": support,
        "direction": direction,
        "direction_source": "calibrated_likelihood",
        "strength": 0.9 if direction == "AI-generated" else (0.1 if direction == "Real" else 0.5),
        "measurement_scope": scope,
        "raw_metric": raw,
        "region_pixels": list(region),
        "region": f"patch_coordinates_{list(region)}",
        "calibrated_likelihood": {"Real": round(1 - likelihood, 3), "Fake": likelihood},
        "applicability": "inverted:high-metric-means-real",
        "applicability_conditions": "高 strength 统计上对应 Real，不得按 AI-generated 解读。",
        "interpretation_text": "calibration sentence",
        "counter_explanation": "重压缩也会改变该指标。",
    }


def _record(**overrides):
    record = {
        "trajectory_id": "noise__ADM_x_png",
        "policy": "noise",
        "tools_served": ["noise"],
        "variant_id": "ADM_x_png",
        "source_id": "ADM/x",
        "split": "train",
        "generator": "ADM",
        "treatment": "png",
        "container": "png",
        "ground_truth": "Fake",
        "final_verdict": "Fake",
        "confidence": 0.9,
        "posterior": 0.9,
        "conflict_score": 0.0,
        "policy_reasons": ["candidate_matches_posterior"],
        "counters": {"model_turns": 2, "expert_calls": 1, "weighted_cost": 2.0},
        "scores": {"posterior_brier": 0.01},
        "evidence": [_token()],
        "image_path": "sft_data/variants/ADM_x_png.png",
        "resolution": [48, 64],
    }
    record.update(overrides)
    return record


def _sample(**overrides):
    return render_sample(_record(**overrides), "positive",
                         split_entry=SPLIT["sources"]["ADM/x"])


def _validate(sample, **kwargs):
    return validate_sample(sample, split=SPLIT, applicability=APPLICABILITY, **kwargs)


class TestRendering:
    def test_all_four_sections_in_order(self):
        answer = render_answer(_record())
        order = re.findall(
            r"<(observation|forensic_evidence|reasoning|verdict)>", answer)
        assert order == list(SECTIONS)

    def test_evidence_block_carries_scope_measurement_and_calibration(self):
        answer = render_answer(_record())
        body = parse_sections(answer)["forensic_evidence"]
        assert "evidence_id: E-1" in body
        assert "measurement_scope: global" in body
        assert "raw_metric: 1.0" in body
        assert "calibrated_likelihood: Real=0.1, Fake=0.9" in body
        assert "applicability: inverted:high-metric-means-real" in body

    def test_a_tool_free_record_says_so(self):
        answer = render_answer(_record(policy="no-tool", tools_served=[], evidence=[]))
        assert "本轨迹未调用工具" in parse_sections(answer)["forensic_evidence"]

    def test_verdict_section_is_json(self):
        verdict = json.loads(parse_sections(render_answer(_record()))["verdict"])
        assert verdict == {"verdict": "Fake", "confidence": 0.9, "posterior": 0.9}

    def test_sample_shape_matches_the_frozen_format(self):
        sample = _sample()
        assert set(sample) >= {"id", "image_path", "ground_truth", "source_model",
                               "final_verdict", "conversations", "evidence_chain",
                               "metadata", "type", "source", "audit"}
        assert sample["conversations"][0]["value"].startswith("<image>")
        assert sample["audit"]["review_status"] == "pending_human"


class TestLeakage:
    def test_a_training_source_passes(self):
        assert not [p for p in _validate(_sample()) if p.startswith("leakage")]

    def test_a_test_source_is_caught(self):
        problems = _validate(_sample(source_id="ADM/tested",
                                     trajectory_id="noise__ADM_tested_png"))
        assert any("leakage" in p for p in problems)

    def test_an_unknown_source_is_caught(self):
        problems = _validate(_sample(source_id="ghost", trajectory_id="noise__ghost_png"))
        assert any("not found in the split" in p for p in problems)


class TestStructure:
    def test_missing_section(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            "<reasoning>", "").replace("</reasoning>", "")
        assert any("missing <reasoning>" in p for p in _validate(sample))

    def test_out_of_order_sections(self):
        sample = _sample()
        answer = sample["conversations"][1]["value"]
        # finditer, not findall: the pattern's group is the section name, so
        # findall would hand back the names instead of the blocks.
        blocks = [m.group(0) for m in re.finditer(
            r"<(observation|forensic_evidence|reasoning|verdict)>.*?</\1>", answer, re.S)]
        assert len(blocks) == 4
        shuffled = "\n\n".join([blocks[1], blocks[0], blocks[2], blocks[3]])
        sample["conversations"][1]["value"] = shuffled
        assert any("out of order" in p for p in _validate(sample))

    def test_invalid_verdict_json(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            '{"verdict": "Fake", "confidence": 0.9, "posterior": 0.9}', "{not json}")
        assert any("not valid JSON" in p for p in _validate(sample))

    def test_illegal_label_and_confidence(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            '{"verdict": "Fake", "confidence": 0.9, "posterior": 0.9}',
            '{"verdict": "Probably", "confidence": 1.9, "posterior": 0.9}')
        problems = _validate(sample)
        assert any("illegal label" in p for p in problems)
        assert any("confidence out of range" in p for p in problems)


class TestEvidenceReferences:
    def test_unknown_reference_is_caught(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            "evidence_id: E-1", "evidence_id: E-ghost")
        assert any("unknown evidence_id" in p for p in _validate(sample))

    def test_duplicate_reference_is_caught(self):
        sample = _sample()
        body = parse_sections(sample["conversations"][1]["value"])["forensic_evidence"]
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            body, body + "\n- evidence_id: E-1")
        assert any("referenced twice" in p for p in _validate(sample))


class TestScopeAndRegion:
    def test_a_region_scoped_measurement_is_caught(self):
        problems = _validate(_sample(evidence=[_token(scope="region")]))
        assert any("not a global measurement" in p for p in problems)

    def test_a_bbox_outside_the_image_is_caught(self):
        problems = _validate(_sample(evidence=[_token(region=(2, 2, 400, 400))]))
        assert any("bbox outside the image" in p for p in problems)


class TestDuplicates:
    def test_the_same_expert_twice_is_caught(self):
        problems = _validate(_sample(evidence=[_token(), _token(evidence_id="E-2")]))
        assert any("repeated expert calls" in p for p in problems)


class TestDirection:
    def test_support_contradicting_the_calibration_is_caught(self):
        problems = _validate(_sample(evidence=[_token(support="Real", likelihood=0.9)]))
        assert any("the calibration gives AI-generated" in p for p in problems)

    def test_a_consistent_token_passes(self):
        assert not [p for p in _validate(_sample()) if p.startswith("direction")]


class TestApplicability:
    def test_a_retired_expert_is_caught(self):
        problems = _validate(_sample(evidence=[_token(source="ela_expert")]))
        assert any("retired expert" in p for p in problems)

    def test_a_carrying_expert_below_the_bar_is_caught(self):
        problems = _validate(_sample(treatment="jpeg_q70", tools_served=["jpeg"],
                                     evidence=[_token(source="jpeg_expert")]))
        assert any("0.431" in p for p in problems)

    def test_corroboration_below_the_bar_with_a_label_is_caught(self):
        problems = _validate(_sample(
            policy="frequency", tools_served=["freq"], treatment="png",
            evidence=[_token(source="frequency_expert_v2", support="AI-generated")]))
        assert any("0.556" in p for p in problems)

    def test_corroboration_below_the_bar_with_an_abstention_is_kept(self):
        """Weak evidence ending in silence is the lesson, not a defect."""
        problems = _validate(_sample(
            policy="frequency", tools_served=["freq"], treatment="png",
            final_verdict="Uncertain", confidence=0.5, posterior=0.5,
            evidence=[_token(source="frequency_expert_v2", support="Real", likelihood=0.48)]))
        assert not [p for p in problems if p.startswith("applicability")]


class TestVerdictAgainstPosterior:
    def test_a_label_contradicting_the_posterior_is_caught(self):
        problems = _validate(_sample(final_verdict="Real", posterior=0.9))
        assert any("contradicts posterior" in p for p in problems)

    def test_abstaining_with_a_decisive_posterior_is_caught(self):
        problems = _validate(_sample(final_verdict="Uncertain", posterior=0.95,
                                     confidence=0.5))
        assert any("abstained although" in p for p in problems)


class TestConfidenceSupport:
    def test_a_confident_label_without_evidence_is_caught(self):
        problems = _validate(_sample(evidence=[], tools_served=[], policy="no-tool"))
        assert any("no evidence supports it" in p for p in problems)

    def test_a_confident_label_with_admissible_evidence_passes(self):
        assert not [p for p in _validate(_sample()) if p.startswith("confidence")]


class TestContainerReason:
    def test_a_container_used_as_the_reason_is_caught(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            "<observation>", "<observation>\n这张图是 PNG 格式，因此判定为 AI 生成。")
        problems = _validate(sample)
        assert any("container_reason" in p for p in problems)

    def test_a_denied_container_inference_passes(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            "<observation>",
            "<observation>\n容器格式与真伪无关，不得据此判断 AI 生成。")
        assert not [p for p in _validate(sample) if p.startswith("container_reason")]

    def test_a_bare_mention_without_an_inference_passes(self):
        sample = _sample()
        sample["conversations"][1]["value"] = sample["conversations"][1]["value"].replace(
            "<observation>", "<observation>\n图像为 JPEG 容器。")
        assert not [p for p in _validate(sample) if p.startswith("container_reason")]


class TestDatasetValidation:
    def test_it_reports_only_the_failing_samples(self):
        findings = validate_dataset([_sample(), _sample(final_verdict="Real", posterior=0.9)],
                                    split=SPLIT, applicability=APPLICABILITY)
        assert len(findings) == 1


class TestBuildGuards:
    def test_writing_into_the_frozen_baseline_is_refused(self):
        from scripts.build_final_v2 import FROZEN_DIR

        with pytest.raises(ValueError, match="frozen baseline"):
            assert_not_frozen(FROZEN_DIR)
        with pytest.raises(ValueError):
            assert_not_frozen(FROZEN_DIR + "/nested")

    def test_a_final_v2_directory_is_fine(self, tmp_path):
        assert_not_frozen(str(tmp_path / "final_v2"))

    def test_bucketing_separates_tool_from_baseline_samples(self):
        assert bucket_for({"tools_served": ["noise"]}, "positive") == "positive"
        assert bucket_for({"tools_served": []}, "positive") == "no_tool_positive"
        assert bucket_for({"tools_served": ["noise"]}, "honest_abstention") == \
            "honest_abstention"
