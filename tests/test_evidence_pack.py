"""
The slim evidence pack (stage 4.3, proposal §5).

The model is shown one score with its scale, direction, threshold and the
conditions under which it may be read; the full token stays in the trace for
the ledger, the converter and the reviewer.  The split exists because a single
token used to carry five numbers that could disagree — and did.

These tests lock the field set, the single-score rule, and the fact that the
conditions travel with the score rather than living in a prompt nobody updates.
"""

import json

import pytest

from state_machine.evidence_tokenizer import EvidenceTokenizer

pytestmark = pytest.mark.core

# 模型可见的必有字段（visual_artifacts 只在有产物时出现）
REQUIRED = ("evidence_id", "expert", "measurement", "status", "scope", "score", "reading")
# 只能在后台出现的字段：任何一项出现在模型可见包里，都是这次拆分白做了
BACKGROUND_ONLY = ("strength", "raw_metric", "support", "support_raw",
                   "calibrated_likelihood", "reliability", "applicability",
                   "applicability_conditions", "calibration", "reasoning",
                   "interpretation_text", "counter_explanation", "direction_source",
                   "semantics_aligned", "region_pixels", "condition_metadata")


class _Result:
    def __init__(self, source, evidence_name, strength, raw_metric, support="Fake"):
        self.source = source
        self.evidence_name = evidence_name
        self.strength = strength
        self.raw_metric = raw_metric
        self.support = support
        # 散文里的数字跟着 raw_metric 走：写死一个值会让"散文与 score 一致"
        # 这类检查在夹具上就已经失真（它曾经写死 0.93，而 token 是 0.02）
        self.phenomenon = f"整图测量：分数 {raw_metric:.4f}。"
        self.reasoning = "长段论证，模型不需要看到全部。"
        self.interpretation_text = "解释句。"


def _token(source, evidence_name, raw, strength=None):
    return EvidenceTokenizer.tokenize(
        _Result(source, evidence_name, strength if strength is not None else raw, raw),
        bbox=[10, 20, 30, 40], image_shape=(100, 100), measurement_scope="global")


class TestTheFieldSetIsFixed:
    def test_exactly_the_promised_fields_are_shown(self):
        pack = EvidenceTokenizer.model_visible(_token("probe_dino_expert",
                                                      "aigi_discriminator_score", 0.93))
        assert set(REQUIRED) <= set(pack)
        assert set(pack) <= set(REQUIRED) | {"visual_artifacts", "artifact_meaning"}

    @pytest.mark.parametrize("field", BACKGROUND_ONLY)
    def test_no_background_field_leaks_into_the_pack(self, field):
        pack = EvidenceTokenizer.model_visible(_token("probe_dino_expert",
                                                      "aigi_discriminator_score", 0.93))
        assert field not in pack, f"{field} 不该出现在模型可见包里"

    def test_the_session_trace_keeps_the_full_token(self):
        """拆分只在投递处发生：证据链里仍是完整 token（账本与转换要读它）。"""
        token = _token("probe_dino_expert", "aigi_discriminator_score", 0.93)
        assert "strength" in token and "raw_metric" in token and "reasoning" in token


class TestOneScoreWithItsSemantics:
    def test_the_score_carries_scale_direction_and_threshold(self):
        pack = EvidenceTokenizer.model_visible(_token("probe_dino_expert",
                                                      "aigi_discriminator_score", 0.93))
        score = pack["score"]
        assert score["value"] == 0.93
        assert score["range"] == [0.0, 1.0]
        assert score["direction"] == "higher_is_fake"
        assert score["threshold"] == 0.5
        assert score["is_calibrated_probability"] is False, \
            "sigmoid 分数不是本地校准概率，这一条不能省"

    def test_the_value_comes_from_the_registrys_field_not_whichever_is_present(self):
        """noise 的 strength 与 raw_metric 量纲不同：必须按登记表选字段。"""
        pack = EvidenceTokenizer.model_visible(
            _token("noise_expert", "noise_residual_inconsistency", 2.5, strength=0.71))
        assert pack["score"]["value"] == 0.71          # strength，不是 raw_metric=2.5
        assert pack["score"]["direction"] == "higher_is_real"

    def test_an_unregistered_expert_says_so_instead_of_borrowing_a_direction(self):
        pack = EvidenceTokenizer.model_visible(
            _token("some_offline_expert", "whatever_metric", 0.5))
        assert "未登记" in pack["score"]["direction"]
        assert "range" not in pack["score"]


class TestNoInternalNoteReachesTheModel:
    """
    登记表里有两类文字：给模型的（`prompt_note`）与给我们自己的
    （`usage_conditions`/`limits`，含 bootstrap 数值与写法要求）。
    曾经把后者拼进了模型输入——这里机械地挡住它再次发生。
    """

    INTERNAL_MARKERS = ("bootstrap", "CI[", "置信区间", "适用条件写", "不写",
                        "登记表", "本轮 val")

    def _packs(self):
        return [EvidenceTokenizer.model_visible(_token("npr_expert", "npr_score", 0.02)),
                EvidenceTokenizer.model_visible(_token("univfd_expert", "u", 0.9)),
                EvidenceTokenizer.model_visible(_token("probe_dino_expert", "p", 0.5))]

    def test_no_pack_carries_an_internal_marker(self):
        for pack in self._packs():
            blob = json.dumps(pack, ensure_ascii=False)
            for marker in self.INTERNAL_MARKERS:
                assert marker not in blob, f"{marker!r} 泄漏进了模型输入：{pack['reading'][:120]}"

    def test_the_model_facing_note_is_the_registrys_prompt_note(self):
        from utils.score_semantics import entry

        pack = EvidenceTokenizer.model_visible(_token("npr_expert", "npr_score", 0.02))
        assert entry("npr_expert")["prompt_note"] in pack["reading"]


class TestTheConditionsTravelWithTheScore:
    def test_nprs_png_condition_and_its_jpeg_caveat_are_in_the_pack(self):
        """条件随分数走：PNG 是它被验证的编码，JPEG 下低分不是 Real 证据。"""
        pack = EvidenceTokenizer.model_visible(_token("npr_expert", "npr_score", 0.02))
        assert "PNG" in pack["reading"]
        assert "not evidence of a real photograph" in pack["reading"].lower()

    def test_univfds_centre_crop_scope_is_visible(self):
        pack = EvidenceTokenizer.model_visible(_token("univfd_expert", "univfd_score", 0.9))
        assert pack["scope"] == "global"          # 测量范围以 token 为准（本次调用是整图）
        assert "centre crop" in pack["reading"] or "center crop" in pack["reading"]

    def test_a_low_npr_score_says_it_carries_no_direction(self):
        """用户 2026-10-15 的边界：JPEG 低分不等于强 Real——包必须这么说。"""
        pack = EvidenceTokenizer.model_visible(_token("npr_expert", "npr_score", 0.02))
        assert pack["score"]["direction_for_this_value"] == "Uncertain"
        assert "does not indicate a real photograph" in pack["reading"]

    def test_a_high_npr_score_says_generated(self):
        pack = EvidenceTokenizer.model_visible(_token("npr_expert", "npr_score", 0.98))
        assert pack["score"]["direction_for_this_value"] == "AI-generated"

    def test_probes_whole_image_scope_is_visible(self):
        pack = EvidenceTokenizer.model_visible(_token("probe_dino_expert",
                                                      "aigi_discriminator_score", 0.9))
        assert pack["scope"] == "global"
        assert pack["score"]["direction"] == "higher_is_fake"


class TestTheProseAndTheScoreAgree:
    """
    包里有**两处**数字：`score.value`，以及 phenomenon 散文里的那一个（外部判别器
    把原始分写进句子）。两处不一致，正是 G2-e 那类"同一个 token 里两个数字互相
    矛盾"的老毛病——而且散文那处更容易被模型引用，因为它读起来像结论。

    当前工具组的三个专家的登记字段都是 `raw_metric`，与散文同源，所以必须一致。
    噪声专家（已退役）散文写原始残差量、登记字段是 `strength`，是唯一的不一致
    案例；它不在新主线上，若回归必须先解决这一点。
    """

    def test_every_number_in_the_reading_is_the_score(self):
        import re

        from experts.external_base import score_phenomenon

        for source, evidence in (("probe_dino_expert", "aigi_discriminator_score"),
                                 ("npr_expert", "npr_upsampling_artefact_score"),
                                 ("univfd_expert", "univfd_score")):
            token = EvidenceTokenizer.tokenize(
                _Result(source, evidence, 0.93, 0.93),
                bbox=[0, 0, 100, 100], image_shape=(100, 100), measurement_scope="global")
            token["phenomenon"] = score_phenomenon(0.93, "resize(224,224)")
            pack = EvidenceTokenizer.model_visible(token)
            # 带小数点的数字：预处理里的 "224" 不是
            numbers = [float(n) for n in re.findall(r"\d+\.\d+", pack["reading"])]
            assert numbers, pack["reading"]
            for number in numbers:
                assert number == pytest.approx(pack["score"]["value"], abs=1e-6), \
                    f"{source} 的散文数字 {number} 与 score.value 不一致"


class TestStatusIsExplicit:
    def test_a_measured_token_says_measured(self):
        pack = EvidenceTokenizer.model_visible(_token("probe_dino_expert",
                                                      "aigi_discriminator_score", 0.9))
        assert pack["status"] == "measured"

    def test_a_failed_token_can_say_failed(self):
        """执行失败不得变成中性测量：状态跟着 token 走。"""
        token = _token("probe_dino_expert", "aigi_discriminator_score", 0.5)
        token["status"] = "failed"
        assert EvidenceTokenizer.model_visible(token)["status"] == "failed"


class TestArtifactsComeWithTheirMeaning:
    def test_artifacts_are_listed_with_a_note_that_they_are_not_evidence(self):
        token = _token("probe_dino_expert", "aigi_discriminator_score", 0.9)
        token["visual_artifacts"] = ["traces/evidence/x/region_E-1.png"]
        pack = EvidenceTokenizer.model_visible(token)
        assert pack["visual_artifacts"] == ["traces/evidence/x/region_E-1.png"]
        assert "不是模型的判据" in pack["artifact_meaning"]
