"""
Tests for the evidence ledger (G7 §4.17 R3).

The ledger exists because 45 sentences in the first batch reported results for
experts that never ran, and one fabricated a whole calibrated_likelihood field.
What has to hold: a claim about a measurement either matches the ledger or it is
flagged, and advice, absence and hypotheses are never flagged.
"""

import json

import pytest

from utils.evidence_ledger import Ledger, correction_note

pytestmark = pytest.mark.core

def _token(source="probe_dino_expert", raw=0.9838, p_fake=1.0, evidence_id="E-1"):
    return {"source": source, "evidence_id": evidence_id, "raw_metric": raw,
            "strength": raw, "measurement_scope": "global",
            "calibrated_likelihood": {"Real": round(1 - p_fake, 3), "Fake": p_fake}}

class TestTheLedgerItself:
    def test_it_records_what_ran_and_what_was_served(self):
        ledger = Ledger.from_session([_token()], served=["probe_dino_expert", "noise_expert"])
        assert ledger.ran("probe_dino_expert") and not ledger.ran("noise_expert")
        assert "probe_dino_expert" in ledger.summary()
        assert "noise_expert" in ledger.summary()

    def test_the_summary_names_skipped_repeats(self):
        ledger = Ledger.from_session([_token()], skipped_sources=["noise_expert"],
                                     served=["probe_dino_expert", "noise_expert"])
        assert "被跳过" in ledger.summary()

    def test_every_number_in_the_tokens_is_permitted(self):
        ledger = Ledger.from_session([_token(raw=3.091196, p_fake=0.564)])
        for value in (3.091196, 0.564, 0.436):
            assert any(abs(value - known) < 1e-6 for known in ledger.permitted_numbers())

class TestFabricatedExpert:
    def test_a_result_for_an_unrun_expert_is_flagged(self):
        ledger = Ledger.from_session([_token()], served=["probe_dino_expert", "noise_expert"])
        findings = ledger.findings("微噪声残留专家也显示了较高的微噪声水平，支持这一结论。")
        assert [f["kind"] for f in findings] == ["fabricated_expert_result"]
        assert findings[0]["source"] == "noise_expert"

    def test_a_result_for_an_expert_that_ran_is_fine(self):
        ledger = Ledger.from_session([_token("noise_expert", 4.2, 0.11)])
        assert ledger.findings("微噪声残留专家显示了 4.2 的水平。") == []

    def test_advice_absence_and_hypothesis_are_not_claims(self):
        ledger = Ledger.from_session([_token()])
        for sentence in ("建议调用噪声残留专家进一步验证。",
                         "本会话未调用噪声残留专家。",
                         "若调用噪声残留专家，可能会得到较低的水平。",
                         "尚未获得噪声专家的测量结果。"):
            assert ledger.findings(sentence) == [], sentence

class TestFabricatedValue:
    def test_a_number_from_nowhere_is_flagged(self):
        ledger = Ledger.from_session([_token(raw=0.9838)])
        findings = ledger.findings("探针专家的得分为 0.551，指向真实图像。")
        assert [f["kind"] for f in findings] == ["fabricated_value"]

    def test_a_rounded_real_value_is_not_flagged(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.091196, p_fake=0.564)])
        assert ledger.findings("噪声残留专家的水平为 3.09，校准概率 0.56。") == []

    def test_coordinates_and_booleans_are_not_values(self):
        ledger = Ledger.from_session([_token()])
        assert ledger.findings("探针专家在 [0, 0, 1000, 1000] 上测量，结果为 0.98。") == []

class TestSkippedRepeat:
    def test_a_skipped_request_reported_as_new_measurement_is_flagged(self):
        ledger = Ledger.from_session([_token()], skipped_sources=["noise_expert"])
        findings = ledger.findings("噪声残留专家再次测量显示微噪声水平较低。")
        assert [f["kind"] for f in findings] == ["skipped_request_as_new_evidence"]

    def test_without_a_skip_nothing_is_flagged(self):
        ledger = Ledger.from_session([_token()])
        assert ledger.findings("噪声残留专家再次测量显示微噪声水平较低。") != []  # 无 token 时仍是虚构
        ledger2 = Ledger.from_session([_token("noise_expert", 4.0, 0.1)])
        assert ledger2.findings("噪声残留专家再次测量显示 4.0 的水平。") == []

class TestTheCorrectionNote:
    def test_it_states_the_ledger_and_forbids_the_defect(self):
        ledger = Ledger.from_session([_token()], served=["probe_dino_expert", "noise_expert"])
        findings = ledger.findings("微噪声残留专家显示较高的水平。")
        note = correction_note(ledger, findings)
        assert "probe_dino_expert" in note and "noise_expert" in note
        assert "编造" in note or "不存在的证据" in note
        # it must not suggest a label
        for word in ("Real", "Fake", "伪造", "真实图像"):
            assert word not in note

    def test_it_quotes_the_offending_sentence(self):
        ledger = Ledger.from_session([_token()])
        findings = ledger.findings("JPEG 压缩专家给出了 blockiness 2.5 的结果。")
        note = correction_note(ledger, findings)
        assert "JPEG" in note

class TestFieldBinding:
    """A number belongs to one expert and one field, not to a shared pool."""

    def test_a_number_from_another_expert_is_not_available(self):
        """The review's defect: PROBE's value quoted as the noise expert's."""
        ledger = Ledger.from_session([
            _token("probe_dino_expert", raw=0.9838, p_fake=1.0, evidence_id="E-P"),
            _token("noise_expert", raw=4.2, p_fake=0.11, evidence_id="E-N"),
        ])
        findings = ledger.findings("噪声残留专家的水平为 0.9838，支持伪造。")
        assert [f["kind"] for f in findings] == ["misattributed_value"]
        assert findings[0]["source"] == "probe_dino_expert"
        assert findings[0]["field"] == "raw_metric"
        assert findings[0]["evidence_id"] == "E-P"

    def test_a_raw_metric_called_a_calibrated_probability_is_flagged(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        findings = ledger.findings("噪声残留专家的校准概率为 3.09。")
        assert [f["kind"] for f in findings] == ["misattributed_field"]
        assert findings[0]["field"] == "raw_metric"

    def test_p_real_quoted_as_p_fake_is_flagged(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        findings = ledger.findings("噪声残留专家的校准概率 P(Real) 为 0.564。")
        assert [f["kind"] for f in findings] == ["misattributed_field"]
        assert findings[0]["field"] == "calibrated_likelihood.Fake"

    def test_the_right_probability_is_not_flagged(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        assert ledger.findings("噪声残留专家的校准概率 P(Fake) 为 0.564。") == []
        assert ledger.findings("噪声残留专家的校准概率 P(Real) 为 0.436。") == []

    def test_a_number_from_the_experts_own_text_is_a_candidate(self):
        """Prose numbers may be calibration statistics, not measurements."""
        token = {**_token("jpeg_expert", raw=0.7, p_fake=0.8),
                 "reasoning": "blockiness 2.5 偏高，提示重复压缩。"}
        ledger = Ledger.from_session([token])
        findings = ledger.findings("JPEG 压缩专家显示 blockiness 为 2.5。")
        assert [f["kind"] for f in findings] == ["text_value_as_measurement"]
        assert findings[0]["severity"] == "candidate"

    def test_a_structured_field_is_not_shadowed_by_the_text_pool(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        assert ledger.findings("噪声残留专家的水平为 3.09。") == []

    def test_the_summary_prints_the_right_values_for_a_rewrite(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564,
                                             evidence_id="E-N")])
        summary = ledger.summary()
        assert "raw_metric=3.0912" in summary and "E-N" in summary
        # the note must not put a label name into the model's mouth
        assert "Fake" not in summary and "Real" not in summary

class TestToleranceAndUnits:
    """Tolerance follows the field class and the displayed precision."""

    def test_a_fabricated_certainty_is_flagged(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        for sentence in ("噪声残留专家的校准概率 P(Fake) 为 1.0。",
                         "噪声残留专家的校准概率 P(Fake) 为 0。"):
            assert [f["kind"] for f in ledger.findings(sentence)] == ["fabricated_value"], sentence

    def test_display_precision_sets_the_tolerance(self):
        ledger = Ledger.from_session([_token("probe_dino_expert", raw=0.9838, p_fake=1.0)])
        assert ledger.findings("探针专家的原始分数为 0.98。") == []
        assert [f["kind"] for f in ledger.findings("探针专家的原始分数为 0.9。")] == [
            "fabricated_value"]

    def test_percentages_are_understood(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        assert ledger.findings("噪声残留专家的校准概率 P(Fake) 为 56.4%。") == []

    def test_a_bare_thousand_is_no_longer_exempt(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        assert [f["kind"] for f in ledger.findings("噪声残留专家的水平为 1000。")] == [
            "fabricated_value"]

    def test_coordinates_and_resolutions_stay_out_of_the_check(self):
        ledger = Ledger.from_session([_token()])
        assert ledger.findings(
            "探针专家在 [0, 0, 1000, 1000] 上测量，原图为 1000×1000，结果为 0.98。") == []

    def test_the_models_own_confidence_is_not_an_expert_value(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)])
        assert ledger.findings("噪声残留专家显示 3.09，我给出置信度 0.77。") == []

class TestNegationScope:
    """Negation protects the clause it is in, not the whole sentence."""

    def test_a_caveat_does_not_excuse_the_claim_before_it(self):
        ledger = Ledger.from_session([_token()], served=["probe_dino_expert", "noise_expert"])
        findings = ledger.findings("噪声残留专家显示低噪声水平，但不能单独确定真假。")
        assert [f["kind"] for f in findings] == ["fabricated_expert_result"]

    def test_a_negated_remoteasurement_is_not_a_new_measurement(self):
        ledger = Ledger.from_session([_token()], skipped_sources=["noise_expert"])
        assert ledger.findings("噪声残留专家重复请求已跳过，没有再次测量得到新结果。") == []

class TestSystemNumbers:
    """Numbers the system itself injected may be cited — as system numbers."""

    SYSTEM = ["[System: 你的结论与当前校准后验不一致（P(Fake)=0.63，冲突度=0.40），暂不能结案。]"]

    def _ledger(self):
        return Ledger.from_session([_token("noise_expert", raw=3.0912, p_fake=0.564)],
                                   system_texts=self.SYSTEM)

    def test_a_system_number_is_citable_beside_an_expert_mention(self):
        assert self._ledger().findings("噪声残留专家显示 3.09，系统参考后验显示 0.63。") == []

    def test_an_expert_cannot_borrow_the_system_number(self):
        findings = self._ledger().findings("噪声残留专家的水平为 0.63。")
        assert [f["kind"] for f in findings] == ["fabricated_value"]

class TestDirectionConsistency:
    """
    A real number read in the wrong direction (§4.17 B 段第 1.6 步 A).

    The reproduced case: PROBE raw=0.3431 with a near-chance calibration
    (P(Fake)=0.52), reported as "表明图像更可能真实".  Attribution was
    perfect, so every previous check passed.
    """

    TOKEN = {"source": "probe_dino_expert", "evidence_id": "E-P", "raw_metric": 0.3431,
             "strength": 0.3431, "calibrated_likelihood": {"Real": 0.48, "Fake": 0.52}}

    def _ledger(self, likelihood=None, **overrides):
        token = {**self.TOKEN, **overrides}
        if likelihood is not None:
            token["calibrated_likelihood"] = likelihood
        return Ledger.from_session([token])

    def test_the_reproduced_case_is_flagged(self):
        findings = self._ledger().findings("PROBE 原始分数 0.3431，表明图像更可能真实。")
        assert [f["kind"] for f in findings] == ["direction_overread"]
        # near chance → a person decides, not the correction loop
        assert findings[0]["severity"] == "candidate"

    def test_a_clear_direction_contradicted_is_a_defect(self):
        ledger = self._ledger(likelihood={"Real": 0.05, "Fake": 0.95})
        findings = ledger.findings("PROBE 原始分数 0.3431，表明图像更可能真实。")
        assert [f["kind"] for f in findings] == ["direction_conflict"]
        assert findings[0]["severity"] == "defect"

    def test_the_direction_the_calibration_gives_is_not_flagged(self):
        ledger = self._ledger(likelihood={"Real": 0.05, "Fake": 0.95})
        assert ledger.findings("PROBE 原始分数 0.3431，表明图像更可能伪造。") == []

    def test_a_hedged_weak_value_may_lean(self):
        """Entry 4's policy: a near-chance number may lean, not decide."""
        assert self._ledger().findings(
            "PROBE 原始分数 0.3431 接近随机，不足以指向真实。") == []

    def test_the_direction_check_needs_a_real_citation(self):
        """A fabricated number is already its own defect."""
        findings = self._ledger().findings("PROBE 得分 0.999，表明图像更可能真实。")
        assert [f["kind"] for f in findings] == ["fabricated_value"]

    def test_a_direction_in_the_next_sentence_is_caught(self):
        findings = self._ledger().findings("PROBE 原始分数 0.3431。这表明图像更可能真实。")
        assert [f["kind"] for f in findings] == ["direction_overread"]

    def test_the_expert_name_is_not_a_direction_assertion(self):
        """"AI 生成探测器" names the tool; it does not assert a direction."""
        ledger = self._ledger(likelihood={"Real": 0.05, "Fake": 0.95})
        findings = ledger.findings("AI生成探测器给出了 0.3431，表明图像更可能真实。")
        assert [f["kind"] for f in findings] == ["direction_conflict"]

    def test_an_uncalibrated_support_label_conflict_is_a_candidate(self):
        token = {"source": "noise_expert", "evidence_id": "E-N", "raw_metric": 4.2,
                 "strength": 0.9, "support": "AI-generated", "semantics_aligned": True}
        ledger = Ledger.from_session([token])
        findings = ledger.findings("噪声残留专家显示 4.2 的高水平，表明图像更可能真实。")
        assert [f["kind"] for f in findings] == ["direction_conflict"]
        assert findings[0]["severity"] == "candidate"

class TestNegativeResultsAreStillClaims:
    """"No noise found" is a measurement; "not measured" is not (§4.17 1.6 B)."""

    def test_a_clean_result_from_an_unrun_expert_is_flagged(self):
        ledger = Ledger.from_session([_token()], served=["probe_dino_expert", "noise_expert"])
        findings = ledger.findings("noise 专家显示没有相机噪声，支持生成。")
        assert [f["kind"] for f in findings] == ["fabricated_expert_result"]
        assert findings[0]["source"] == "noise_expert"

    def test_a_real_negative_result_is_checked_and_passes(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=0.4, p_fake=0.2)])
        assert ledger.findings("噪声残留专家显示没有明显的相机噪声，水平为 0.4。") == []

    def test_a_missing_result_is_still_not_a_claim(self):
        ledger = Ledger.from_session([_token()], served=["probe_dino_expert", "noise_expert"])
        assert ledger.findings("本会话未获得噪声残留专家的测量结果。") == []

class TestAttributionUsesTheNearestExpert:
    """Two experts in one sentence must not licence each other's numbers."""

    def _ledger(self):
        return Ledger.from_session([
            _token("probe_dino_expert", raw=0.9838, p_fake=1.0, evidence_id="E-P"),
            _token("noise_expert", raw=4.2, p_fake=0.11, evidence_id="E-N"),
        ])

    def test_a_value_that_belongs_to_the_other_expert_is_flagged(self):
        findings = self._ledger().findings(
            "噪声残留专家与探针专家相比，噪声残留专家的水平为 0.9838。")
        assert [f["kind"] for f in findings] == ["misattributed_value"]
        assert findings[0]["source"] == "probe_dino_expert"

    def test_a_comparison_of_two_real_values_is_not_flagged(self):
        assert self._ledger().findings(
            "噪声残留专家的水平为 4.2，探针专家的原始分数为 0.9838。") == []

class TestPerCentPrecision:
    def test_the_displayed_precision_survives_the_conversion(self):
        ledger = Ledger.from_session([_token("noise_expert", raw=3.0, p_fake=0.9387)])
        assert ledger.findings("噪声残留专家的校准概率 P(Fake) 为 93.9%。") == []
        assert [f["kind"] for f in ledger.findings(
            "噪声残留专家的校准概率 P(Fake) 为 95%。")] == ["fabricated_value"]

class TestProseNumbersHaveARole:
    """
    An expert's prose carries sub-metrics, calibration statistics and
    thresholds.  Quoting a sub-metric *with its name* is quoting the expert;
    lifting a bare number out of the prose is a role question for a person.
    """

    def test_a_named_submetric_quoted_with_its_name_is_the_experts_own(self):
        token = {**_token("jpeg_expert", raw=0.7, p_fake=0.8),
                 "reasoning": "blockiness=0.8779 与 DCT anomaly=4.6279 均偏高。"}
        ledger = Ledger.from_session([token])
        assert ledger.findings("JPEG 压缩专家显示 blockiness=0.8779 的结构特征。") == []

    def test_a_calibration_statistic_cited_as_a_measurement_is_a_candidate(self):
        """The real trace: the model cited the calibration set's medians."""
        token = {**_token("noise_expert", raw=3.0912, p_fake=0.564),
                 "reasoning": "校准集中生成图像中位数为2.0，真实图像中位数为3.6。"}
        ledger = Ledger.from_session([token])
        findings = ledger.findings(
            "噪声残留专家给出的校准集中位数为 2.0，真实图像中位数为 3.6。")
        assert [f["kind"] for f in findings] == ["text_value_as_measurement"] * 2
        assert {f["value"] for f in findings} == {2.0, 3.6}
        assert all(f["severity"] == "candidate" for f in findings)


class TestAVisualObservationIsNotAnExpertClaim:
    """
    b2 实测（第 1 条会话）：模型按 RULES 8 描述"未发现压缩块、噪点或边缘异常"，
    账本把"压缩"当成 jpeg 专家名、又把主语带进下一个断言子句，于是判成
    "声称 jpeg 专家的结果"——**惩罚了遵守提示词的行为**，还触发了一轮纠正。

    两处机制都修了：非断言子句不更新主语；物证词（压缩/噪声/频域）单独出现
    不算点名专家（工具原名与"××专家"仍算）。
    """

    SENTENCE = ("视觉观察中未发现压缩块、噪点或边缘异常，"
                "但该专家的高分表明图像整体结构或语义存在AI生成特征。")

    def test_the_b2_sentence_raises_nothing(self):
        from utils.evidence_ledger import Ledger

        token = {"evidence_id": "E-1", "source": "probe_dino_expert",
                 "evidence_name": "aigi_discriminator_score", "strength": 0.99997,
                 "raw_metric": 0.99997, "support": "AI-generated",
                 "phenomenon": "整图判别分数 0.99997。", "reasoning": "x",
                 "interpretation_text": "y", "measurement_scope": "global",
                 "model_delivered": "slim", "status": "measured"}
        ledger = Ledger.from_session([token], served=["probe_dino_expert"])
        assert ledger.findings(self.SENTENCE) == []

    def test_an_artefact_word_alone_does_not_name_the_jpeg_expert(self):
        from utils.evidence_ledger import _named_sources

        assert _named_sources("视觉观察中未发现压缩块、噪点或边缘异常") == []
        assert _named_sources("噪点明显，压缩痕迹很重") == []

    def test_the_expert_marker_still_names_it(self):
        from utils.evidence_ledger import _named_sources

        assert "jpeg_expert" in _named_sources("压缩专家给出 0.9")
        assert "jpeg_expert" in _named_sources("jpeg 专家给出 0.9")
        assert "noise_expert" in _named_sources("噪声残留专家给出 3.1")
        assert "frequency_expert_v2" in _named_sources("频域专家给出 0.2")

    def test_a_real_claim_about_an_unserved_expert_is_still_a_defect(self):
        """修的是误报，不是把这道检查关掉。"""
        from utils.evidence_ledger import Ledger

        ledger = Ledger.from_session([], served=["probe_dino_expert"])
        kinds = [finding["kind"] for finding in ledger.findings("频域专家给出的结果是 0.9。")]
        assert kinds == ["fabricated_expert_result"]


class TestTheTwoFalsePositivesTheBatchFound:
    """
    b1 小批（32 会话）里有两条候选，逐句核过——**都是账本的假阳性**，不是模型的
    问题。这里把两种形态钉住，免得下次又被记成"交人工裁决"的噪音。
    """

    def test_a_negated_generation_claim_is_not_a_fake_assertion(self):
        """'未被外部判别器识别为AI生成' 是在说**没有**生成证据。"""
        from utils.evidence_ledger import _asserted_direction

        assert _asserted_direction(
            "probe_dino_expert 的测量结果为 0.00514，方向为 Real，"
            "表明该图像在全局层面未被外部判别器识别为AI生成") is None

    def test_quoting_the_delivered_threshold_is_a_clean_citation(self):
        """包里有 `score.threshold`：写"低于阈值 0.5"是在引用它看到过的东西。"""
        from utils.evidence_ledger import Ledger

        token = {"evidence_id": "E-1", "source": "probe_dino_expert",
                 "evidence_name": "aigi_discriminator_score", "strength": 0.1396,
                 "raw_metric": 0.1396, "support": "Real",
                 "phenomenon": "整图判别分数 0.1396。", "reasoning": "x",
                 "interpretation_text": "y", "measurement_scope": "global",
                 "model_delivered": "slim", "status": "measured"}
        ledger = Ledger.from_session([token], served=["probe_dino_expert"])
        assert ledger.findings(
            "图像经外部判别器检测得分为0.1396，低于阈值0.5。") == []

    def test_a_number_that_was_never_delivered_is_still_a_defect(self):
        from utils.evidence_ledger import Ledger

        token = {"evidence_id": "E-1", "source": "probe_dino_expert",
                 "evidence_name": "aigi_discriminator_score", "strength": 0.1396,
                 "raw_metric": 0.1396, "support": "Real",
                 "phenomenon": "整图判别分数 0.1396。", "reasoning": "x",
                 "interpretation_text": "y", "measurement_scope": "global",
                 "model_delivered": "slim", "status": "measured"}
        ledger = Ledger.from_session([token], served=["probe_dino_expert"])
        kinds = [f["kind"] for f in ledger.findings(
            "图像经外部判别器检测得分为0.77，低于阈值0.5。")]
        assert kinds == ["fabricated_value"]


class TestTheDeliveredFieldRule:
    """
    阶段 4.3：引用"没投递给模型的字段"不算忠实引用。

    用户的场景：noise 的 strength（[0,1]，高=偏 Real）与 raw_metric（SRM 不一致度，
    无上界）方向不同、量纲不同。模型只拿到一个；引用另一个的数值，即便数字本身是真的，
    也是在讨论它没见过的量。
    """

    def _slim_token(self, source, delivered_field, delivered_value, other_field, other_value):
        return {"source": source, "evidence_id": "E-1", "model_delivered": "slim",
                delivered_field: delivered_value, other_field: other_value}

    def test_quoting_the_delivered_field_is_fine(self):
        from utils.evidence_ledger import Ledger

        token = self._slim_token("noise_expert", "strength", 0.71, "raw_metric", 2.5)
        ledger = Ledger.from_session([token])
        # 交付集合＝登记表指定的那个字段 **+ 阈值**（包里有 `score.threshold`，
        # 模型据此写"低于阈值 0.5"是忠实引用，不该算自由文本里的数字）。
        assert ledger.measured["noise_expert"].delivered == {"strength": 0.71,
                                                            "threshold": 0.5}

    def test_quoting_the_undelivered_field_is_a_defect(self):
        from utils.evidence_ledger import Ledger

        token = self._slim_token("noise_expert", "strength", 0.71, "raw_metric", 2.5)
        ledger = Ledger.from_session([token])
        finding = ledger._check_number(2.5, 1, "noise_expert", ["noise_expert"],
                                       strict=False, label=None, system_cue=False,
                                       clause="噪声残留专家报告了 2.5")
        assert finding is not None
        assert finding["kind"] == "undelivered_field_reference"
        assert finding["severity"] == "defect"
        assert "raw_metric" in finding["why"]

    def test_a_token_without_the_stamp_is_treated_as_fully_delivered(self):
        """旧批次的 token 没有戳，当时投递的就是完整 token——不能倒扣它。"""
        from utils.evidence_ledger import Ledger

        token = {"source": "noise_expert", "evidence_id": "E-2",
                 "strength": 0.71, "raw_metric": 2.5}
        ledger = Ledger.from_session([token])
        assert ledger.measured["noise_expert"].delivered == {"strength": 0.71,
                                                             "raw_metric": 2.5}

    def test_a_none_delivery_delivers_nothing(self):
        from utils.evidence_ledger import Ledger

        token = {"source": "probe_dino_expert", "evidence_id": "E-3",
                 "raw_metric": 0.93, "model_delivered": "none"}
        ledger = Ledger.from_session([token])
        assert ledger.measured["probe_dino_expert"].delivered == {}

