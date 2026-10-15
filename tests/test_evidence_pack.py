"""
The slim evidence pack (stage 4.3, proposal §5).

The model is shown one score with its scale, direction, threshold and the
conditions under which it may be read; the full token stays in the trace for
the ledger, the converter and the reviewer.  The split exists because a single
token used to carry five numbers that could disagree — and did.

These tests lock the field set, the single-score rule, and the fact that the
conditions travel with the score rather than living in a prompt nobody updates.
"""

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
        self.phenomenon = "整图测量：分数 0.93。"
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


class TestTheConditionsTravelWithTheScore:
    def test_nprs_png_condition_and_its_jpeg_caveat_are_in_the_pack(self):
        pack = EvidenceTokenizer.model_visible(_token("npr_expert", "npr_score", 0.02))
        assert "PNG" in pack["reading"]
        assert "不得解释成强 Real 证据" in pack["reading"]

    def test_univfds_centre_crop_scope_is_visible(self):
        pack = EvidenceTokenizer.model_visible(_token("univfd_expert", "univfd_score", 0.9))
        assert pack["scope"] == "global"          # 测量范围以 token 为准（本次调用是整图）
        assert "中心裁剪" in pack["reading"] or "center_crop" in pack["score"]["direction"] \
            or "裁剪" in pack["reading"]

    def test_probes_whole_image_scope_is_visible(self):
        pack = EvidenceTokenizer.model_visible(_token("probe_dino_expert",
                                                      "aigi_discriminator_score", 0.9))
        assert pack["scope"] == "global"
        assert pack["score"]["direction"] == "higher_is_fake"


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
