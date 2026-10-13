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
