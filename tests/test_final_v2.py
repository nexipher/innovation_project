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

pytestmark = pytest.mark.core

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

OBSERVATION = "模型视觉观察：眼睛周围的皮肤纹理不一致，可能存在合成痕迹。"

def _sample(**overrides):
    return render_sample(_record(**overrides), "positive",
                         observation_text=OBSERVATION,
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

    def test_a_tool_free_reasoning_does_not_claim_a_posterior(self):
        """
        The 0.5 in a tool-free record is the prior the session started from;
        writing it as "后验 P(Fake)=0.5" claims a measurement that never ran.
        """
        answer = render_answer(_record(policy="no-tool", tools_served=[], evidence=[]))
        reasoning = parse_sections(answer)["reasoning"]
        assert "未获得工具证据" in reasoning
        assert "后验 P(Fake)" not in reasoning      # no posterior is claimed
        assert "P(Fake)=0.5" not in reasoning

    def test_a_tool_record_still_reports_its_posterior(self):
        reasoning = parse_sections(render_answer(_record()))["reasoning"]
        assert "后验 P(Fake)" in reasoning

    def test_verdict_section_is_json(self):
        """The record here is a v2 one, so the label *was* the posterior's."""
        verdict = json.loads(parse_sections(render_answer(_record()))["verdict"])
        assert verdict == {"verdict": "Fake", "confidence": 0.9, "posterior": 0.9,
                           "reference_posterior": 0.9, "verdict_status": "policy_label",
                           "basis": "evidence_posterior"}

    def test_a_v3_record_says_the_label_is_the_models_judgement(self):
        """§4.17 R5: the posterior is a reference, never the label's source."""
        record = _record(policy_version="v3", verdict_status="model_concluded",
                         reference={"recommendation": "Real", "advisory": []})
        verdict = json.loads(parse_sections(render_answer(record))["verdict"])
        assert verdict["basis"] == "model_judgement"
        assert verdict["verdict_status"] == "model_concluded"
        assert verdict["reference_recommendation"] == "Real"
        assert verdict["reference_posterior"] == 0.9
        assert "posterior" not in verdict

    def test_an_unparseable_conclusion_is_not_written_as_an_abstention(self):
        record = _record(policy_version="v3", verdict_status="no_parseable_conclusion",
                         verdict="Uncertain")
        verdict = json.loads(parse_sections(render_answer(record))["verdict"])
        assert verdict["verdict_status"] == "no_parseable_conclusion"

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
        import re

        sample["conversations"][1]["value"] = re.sub(
            r"<verdict>.*?</verdict>", "<verdict>{not json}</verdict>",
            sample["conversations"][1]["value"], flags=re.S)
        assert any("not valid JSON" in p for p in _validate(sample))

    def test_illegal_label_and_confidence(self):
        sample = _sample()
        import re

        sample["conversations"][1]["value"] = re.sub(
            r"<verdict>.*?</verdict>",
            '<verdict>{"verdict": "Probably", "confidence": 1.9, '
            '"basis": "evidence_posterior", "posterior": 0.9}</verdict>',
            sample["conversations"][1]["value"], flags=re.S)
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
    def test_a_confident_label_with_inadmissible_evidence_is_caught(self):
        """Evidence came back, but it carries no direction to stand on."""
        token = _token(support="Uncertain", likelihood=0.5)
        problems = _validate(_sample(evidence=[token], tools_served=["noise"],
                                     policy="noise"))
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

class TestGroundingChecks:
    """
    The four checks the review demanded (plan.md §4.11 C1-C4): they exist to
    catch the defects the validator used to wave through.
    """

    def _with_freq_claim(self, text):
        sample = _sample()
        turns = sample["conversations"]
        turns.insert(-1, {"from": "gpt", "value": text})
        return sample

    def test_citing_an_expert_that_never_ran_is_caught(self):
        """The real case: noise ran, the turn reported frequency results."""
        sample = self._with_freq_claim(
            "<reasoning>此外，<call_freq>专家的多尺度频率分析结果表明，图像的频率分布与"
            "生成图像的统计特征相符，但其分离度仅为0.56-0.63。</reasoning>")
        assert any("frequency_expert_v2 results" in p for p in _validate(sample))

    def test_a_call_shaped_mention_is_an_action_not_a_claim(self):
        sample = self._with_freq_claim(
            "<planning>Expert Target & Hypothesis: <call_freq>[0, 0, 100, 100]</planning>")
        assert not [p for p in _validate(sample) if p.startswith("grounding")]

    def test_naming_a_delivered_expert_is_fine(self):
        sample = _sample()
        problem = [p for p in _validate(sample) if p.startswith("grounding")]
        assert problem == []          # the closing answer names noise_expert, which ran

    def test_a_real_verdict_must_carry_p_real(self):
        sample = _sample(final_verdict="Real", confidence=0.1, posterior=0.1)
        assert any("confidence: Real reported at 0.1" in p for p in _validate(sample))

    def test_a_real_verdict_with_p_real_passes(self):
        sample = _sample(final_verdict="Real", confidence=0.9, posterior=0.1)
        assert not [p for p in _validate(sample) if p.startswith("confidence")]

    def test_a_global_measurement_described_as_local_is_caught(self):
        token = _token()
        token["phenomenon"] = ("Localised noise variance at the level the calibration "
                               "set associates with generated imagery (variance collapse).")
        sample = _sample()
        sample["evidence_chain"] = [token]
        assert any("global measurement" in p for p in _validate(sample))

    def test_a_global_measurement_kept_global_passes(self):
        sample = _sample()
        assert not [p for p in _validate(sample) if p.startswith("scope:")]

    def test_an_observation_without_picture_content_is_caught(self):
        sample = _sample()
        answer = sample["conversations"][-1]["value"]
        sample["conversations"][-1]["value"] = answer.replace(
            OBSERVATION, "")
        assert any(p.startswith("observation:") for p in _validate(sample))

class TestNoToolSamples:
    """A tool-free answer rests on perception, not on an evidence posterior."""

    def _no_tool(self, **overrides):
        return render_sample(
            _record(policy="no-tool", tools_served=[], evidence=[],
                    model_candidate="Real", final_verdict="Real", confidence=0.95,
                    posterior=0.5, **overrides),
            "no_tool_positive", observation_text=OBSERVATION,
            split_entry=SPLIT["sources"]["ADM/x"])

    def test_the_verdict_section_names_its_basis(self):
        verdict = json.loads(parse_sections(
            self._no_tool()["conversations"][1]["value"])["verdict"])
        assert verdict["basis"] == "model_perception"
        assert "posterior" not in verdict

    def test_tool_samples_name_the_evidence_basis(self):
        verdict = json.loads(parse_sections(
            _sample()["conversations"][1]["value"])["verdict"])
        assert verdict["basis"] == "evidence_posterior"
        assert verdict["posterior"] == 0.9

    def test_a_correct_no_tool_answer_passes(self):
        assert _validate(self._no_tool()) == []

    def test_a_prior_posterior_is_not_held_against_it(self):
        """The 0.5 posterior is the prior, not a judgement to contradict."""
        assert not [p for p in _validate(self._no_tool()) if p.startswith("verdict")]

    def test_confidence_without_evidence_is_fine_without_tools(self):
        assert not [p for p in _validate(self._no_tool()) if p.startswith("confidence")]

    def test_claiming_a_conclusion_the_model_did_not_reach_is_caught(self):
        sample = self._no_tool()
        sample["metadata"]["model_candidate"] = "Fake"
        assert any("but the model said" in p for p in _validate(sample))

    def test_tool_samples_still_need_admissible_evidence(self):
        problems = _validate(_sample(evidence=[], tools_served=["noise"]))
        assert any("no evidence supports it" in p for p in problems)

class TestTrainingConversation:
    """
    A tool sample cites evidence ids, so its conversation must contain the
    turns that delivered them — otherwise training teaches the model to cite
    evidence it never received, which is indistinguishable from fabricating it.
    """

    TRACE = {"conversations": [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱。"},
        {"from": "gpt", "value": "<planning>Suspected Region: [1, 2, 3, 4]\n</planning>\n"
                                  "<call_noise>[1, 2, 3, 4]</call_noise>"},
        {"from": "user", "value": '{"evidence_id": "E-raw", "evidence_name": "raw"}'},
        {"from": "user", "value": "[System: 预算已耗尽，请立即输出 <verdict>。]"},
        {"from": "gpt", "value": "旧结案文本"},
    ]}

    def _sample(self):
        return render_sample(_record(), "positive", trace=self.TRACE,
                             split_entry=SPLIT["sources"]["ADM/x"])

    def test_the_tool_call_is_preserved(self):
        turns = self._sample()["conversations"]
        calls = [t for t in turns if t["from"] == "gpt" and "<call_noise>" in t["value"]]
        assert len(calls) == 1

    def test_the_evidence_turn_is_rebuilt_from_the_rectified_token(self):
        turns = self._sample()["conversations"]
        evidence_turns = [t for t in turns
                          if t["from"] == "user" and t["value"].lstrip().startswith("{")]
        assert len(evidence_turns) == 1
        payload = json.loads(evidence_turns[0]["value"])
        assert payload["evidence_id"] == "E-1"
        assert payload["measurement_scope"] == "global"

    def test_the_correction_turn_is_replayed_and_the_old_conclusion_replaced(self):
        """
        The "[System: ...]" correction stays: without it the two conclusions
        sit next to each other with nothing to explain the revision.
        """
        values = " ".join(t["value"] for t in self._sample()["conversations"])
        assert "[System:" in values
        assert "旧结案文本" not in values

    def test_a_rejected_conclusion_keeps_the_correction_that_followed_it(self):
        trace = {"conversations": [
            {"from": "user", "value": "<image>\n看图。"},
            {"from": "gpt", "value": "<planning>…</planning><call_noise>[1, 2, 3, 4]</call_noise>"},
            {"from": "user", "value": '{"evidence_id": "E-raw"}'},
            {"from": "gpt", "value": "<reasoning>我先判 Uncertain。</reasoning>\n<verdict>{}</verdict>"},
            {"from": "user", "value": "[System: 你的结论与后验不一致，请继续取证。]"},
            {"from": "gpt", "value": "<observation>o</observation>\n<verdict>{}</verdict>"},
        ]}
        sample = render_sample(_record(), "positive", observation_text=OBSERVATION,
                               trace=trace, split_entry=SPLIT["sources"]["ADM/x"])
        turns = sample["conversations"]
        roles = [t["from"] for t in turns]
        # 中间结论 → 纠正 → 结案，三者都在且顺序不变
        assert "[System:" in turns[4]["value"]
        assert roles == ["user", "gpt", "user", "gpt", "user", "gpt"]
        assert turns[-1]["value"].startswith("<observation>")
        assert _validate(sample) == [] or all("never given" not in p for p in _validate(sample))

    def test_turns_after_the_closing_answer_are_dropped(self):
        """The sample's verdict is the one derived from the *last* answer."""
        trace = {"conversations": [
            {"from": "user", "value": "<image>\n看图。"},
            {"from": "gpt", "value": "<observation>o</observation>\n<verdict>{}</verdict>"},
            {"from": "user", "value": "[System: 再说一次。]"},
            {"from": "gpt", "value": "<observation>o2</observation>\n<verdict>{}</verdict>"},
            {"from": "user", "value": "[System: 尾巴。]"},
            {"from": "gpt", "value": "这是结案之后的杂音"},
        ]}
        sample = render_sample(_record(), "positive", observation_text=OBSERVATION,
                               trace=trace, split_entry=SPLIT["sources"]["ADM/x"])
        values = " ".join(t["value"] for t in sample["conversations"])
        assert "杂音" not in values
        assert sample["conversations"][-1]["value"].startswith("<observation>")

    def test_a_trailing_reasoning_turn_is_replaced_by_the_rendered_answer(self):
        """
        Real traces never record the four-section answer — the pipeline
        synthesises it — so the last assistant turn is the one it replaces.
        """
        trace = {"conversations": [
            {"from": "user", "value": "<image>\n看图。"},
            {"from": "gpt", "value": "<planning>…</planning><call_noise>[1, 2, 3, 4]</call_noise>"},
            {"from": "user", "value": '{"evidence_id": "E-raw"}'},
            {"from": "user", "value": "[System: 请撰写最终报告。]"},
            {"from": "gpt", "value": "<reasoning>我倾向 Uncertain，<call_noise> 的结果说明…</reasoning>\n<verdict>{\"verdict\": \"Uncertain\"}</verdict>"},
        ]}
        sample = render_sample(_record(), "positive", observation_text=OBSERVATION,
                               trace=trace, split_entry=SPLIT["sources"]["ADM/x"])
        turns = sample["conversations"]
        assert [t["from"] for t in turns] == ["user", "gpt", "user", "user", "gpt"]
        assert turns[-1]["value"].startswith("<observation>")
        # 中间那个结案轮（正文提到 <call_noise>）不再被当成调用轮留下
        assert sum(1 for t in turns if t["from"] == "gpt") == 2
        assert _validate(sample) == []

    def test_the_answer_is_the_final_turn(self):
        turns = self._sample()["conversations"]
        assert turns[-1]["from"] == "gpt"
        assert "<forensic_evidence>" in turns[-1]["value"]

    def test_a_tool_free_sample_has_no_evidence_turns(self):
        sample = render_sample(_record(policy="no-tool", tools_served=[], evidence=[]),
                               "no_tool_positive", trace=self.TRACE,
                               split_entry=SPLIT["sources"]["ADM/x"])
        evidence_turns = [t for t in sample["conversations"]
                          if t["from"] == "user" and t["value"].lstrip().startswith("{")]
        assert evidence_turns == []

class TestConversationValidation:
    def _sample_with_trace(self):
        return render_sample(_record(), "positive", observation_text=OBSERVATION,
                             trace=TestTrainingConversation.TRACE,
                             split_entry=SPLIT["sources"]["ADM/x"])

    def test_a_conversation_that_delivers_its_evidence_passes(self):
        assert _validate(self._sample_with_trace()) == []

    def test_an_answer_citing_evidence_that_was_never_delivered_is_caught(self):
        sample = self._sample_with_trace()
        sample["conversations"] = [t for t in sample["conversations"]
                                   if not (t["from"] == "user"
                                           and t["value"].lstrip().startswith("{"))]
        problems = _validate(sample)
        assert any("never given" in p for p in problems)

    def test_a_fabricated_evidence_id_in_the_conversation_is_caught(self):
        sample = self._sample_with_trace()
        for turn in sample["conversations"]:
            if turn["from"] == "user" and turn["value"].lstrip().startswith("{"):
                turn["value"] = turn["value"].replace("E-1", "E-ghost")
        assert any("unknown evidence_id" in p for p in _validate(sample))

    def test_a_tool_free_sample_carrying_evidence_turns_is_caught(self):
        # Rendered with a trace, so the conversation does carry evidence turns;
        # the policy then claims no tools were used, which cannot both be true.
        sample = self._sample_with_trace()
        sample["metadata"]["tools_served"] = []
        problems = _validate(sample)
        assert any("tool-free sample carries evidence turns" in p for p in problems)

    def test_the_shape_is_checked(self):
        sample = _sample()
        sample["conversations"] = list(reversed(sample["conversations"]))
        problems = _validate(sample)
        assert any("must open with a user turn" in p for p in problems)

class TestObservationExtraction:
    """
    G7 entry 4: where the model's visual description is read from.

    The first batch's no-tool sessions put it in the reasoning (or in a
    planning block nobody asked for), the reader looked only at the planning
    line, and 30 of 56 sessions became metadata-only observations — rejected by
    the validator, which is correct given what they contained.  The training
    baseline prompt now asks for an explicit section, and the reader takes it.
    """

    def _trace(self, tmp_path, assistant_turn):
        trace = {"conversations": [{"from": "user", "value": "x"},
                                   {"from": "gpt", "value": assistant_turn}]}
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        return str(path)

    def test_an_explicit_observation_section_is_used(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        path = self._trace(tmp_path, "<observation>\n画面右侧有重复的纹理\n</observation>\n"
                                      "<reasoning>因此…</reasoning>")
        text = extract_observation({"trace_path": path})
        assert "重复的纹理" in text
        assert "因此" not in text, "结论不应混进观察段"

    def test_the_planning_line_still_works_for_the_tool_arms(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        path = self._trace(tmp_path, "<planning>\nVisual Anomalies: 边缘过度平滑\n</planning>")
        assert "边缘过度平滑" in extract_observation({"trace_path": path})

    def test_the_explicit_section_wins_when_both_are_present(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        path = self._trace(tmp_path,
                           "<planning>\nVisual Anomalies: 旧写法\n</planning>"
                           "<observation>\n新写法\n</observation>")
        assert "新写法" in extract_observation({"trace_path": path})

    def test_an_empty_observation_section_falls_back(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        path = self._trace(tmp_path, "<observation>\n\n</observation>"
                                      "<planning>\nVisual Anomalies: 内容\n</planning>")
        assert "内容" in extract_observation({"trace_path": path})

    def test_nothing_to_read_gives_an_empty_string(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        path = self._trace(tmp_path, "<reasoning>只有结论</reasoning>")
        assert extract_observation({"trace_path": path}) == ""

class TestTheTrainingBaseline:
    def test_the_training_prompt_asks_for_an_observation_section(self):
        from mllm.message_builder import TRAINING_BASELINE_PROMPT

        assert "<observation>" in TRAINING_BASELINE_PROMPT
        assert "no verdict" in TRAINING_BASELINE_PROMPT.lower()

    def test_the_historical_baseline_is_untouched(self):
        """The G2-d/G3-e arms were measured under this exact prompt."""
        import hashlib

        from mllm.message_builder import BASELINE_SYSTEM_PROMPT

        assert "<observation>" not in BASELINE_SYSTEM_PROMPT
        assert hashlib.sha256(BASELINE_SYSTEM_PROMPT.encode()).hexdigest()[:16] == \
            "a81f43b97d1b3714"

class TestTheValidatorKnowsWhichPolicyWroteTheSample:
    """§4.17 R5: under v3 the posterior is a reference, not the label's source."""

    def _v3_sample(self, **overrides):
        record = _record(policy_version="v3", verdict_status="model_concluded",
                         final_verdict="Real", confidence=0.8,
                         reference={"recommendation": "Fake",
                                    "advisory": ["reference_disagrees"]})
        record.update(overrides)
        sample = render_sample(record, "positive", observation_text=OBSERVATION,
                               split_entry=SPLIT["sources"]["ADM/x"])
        return sample

    def test_a_model_led_label_may_disagree_with_the_reference(self):
        problems = _validate(self._v3_sample())
        assert not any("contradicts posterior" in p for p in problems), problems
        assert not any("confidence:" in p for p in problems), problems

    def test_an_undisclosed_disagreement_is_still_a_problem(self):
        sample = self._v3_sample(reference={"recommendation": "Fake", "advisory": []})
        assert any("reference.advisory" in p for p in _validate(sample))

    def test_an_unparsed_conclusion_may_not_be_trained_on(self):
        sample = self._v3_sample(verdict_status="no_parseable_conclusion")
        assert any("不可解析" in p for p in _validate(sample))

    def test_a_v2_sample_still_has_to_agree_with_its_posterior(self):
        sample = render_sample(_record(final_verdict="Real", confidence=0.8),
                               "positive", observation_text=OBSERVATION,
                               split_entry=SPLIT["sources"]["ADM/x"])
        assert any("contradicts posterior" in p for p in _validate(sample))

class TestTheObservationSectionKeepsWhatIsKnown:
    """A conversion that says "None" is worse than one that says nothing."""

    def test_the_input_size_comes_from_the_trace_when_the_field_is_empty(self):
        record = _record(resolution=None)
        sample = render_sample(record, "positive", observation_text="模型视觉观察：纹理不一致。",
                               split_entry=None,
                               trace={"metadata": {"image_size": [256, 256]},
                                      "conversations": []})
        observation = parse_sections(sample["conversations"][-1]["value"])["observation"]
        assert "resolution: [256, 256]" in observation
        assert "输入图像尺寸（模型实际所见）" in observation
        assert "None" not in observation

    def test_an_unknown_size_is_written_as_unrecorded(self):
        from utils.final_v2 import render_observation

        assert "未记录" in render_observation({"container": "png", "treatment": "png"})

    def test_the_models_own_words_are_not_written_by_the_converter(self):
        """No visual description is invented: the section carries what it got."""
        from utils.final_v2 import render_observation

        rendered = render_observation({"container": "png", "treatment": "png",
                                       "resolution": [10, 10]})
        assert "模型视觉观察" not in rendered

    def test_the_new_planning_line_is_read(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        trace = {"conversations": [{"from": "gpt", "value":
                 "<planning>\nVisual Observation: 一只猫坐在窗台上，毛发清晰。\n"
                 "Suspected Anomalies: 未观察到明确异常，需要统计证据\n</planning>"}]}
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        text = extract_observation({"trace_path": str(path)})
        assert "一只猫坐在窗台上" in text

    def test_the_old_planning_line_still_reads(self, tmp_path):
        from scripts.build_final_v2 import extract_observation

        trace = {"conversations": [{"from": "gpt", "value":
                 "<planning>\nVisual Anomalies: 眼睛周围纹理不一致\n</planning>"}]}
        path = tmp_path / "trace.json"
        path.write_text(json.dumps(trace), encoding="utf-8")
        assert "眼睛周围纹理不一致" in extract_observation({"trace_path": str(path)})
