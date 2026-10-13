"""
Faithful conversion (§4.17 R6).

The converter used to write a clean `<reasoning>` section out of token
metadata, whatever the model had actually argued — the audit found answers
whose rendered reasoning was better than the session's.  And the runtime
check covered only the adopted conclusion, while the training conversation
keeps every earlier assistant turn as written.

So: the model's own argument is carried verbatim, the whole supervised
dialogue is audited by the same ledger, and a defect in it refuses the
trajectory instead of being rendered away.
"""

import json

import pytest

from utils.final_v2 import adopted_reasoning, audit_dialogue, render_sample


def _trace(turns):
    return {"conversations": turns}


def _record(**overrides):
    from tests.test_final_v2 import _record as build

    record = build(**overrides)
    record["trajectory_id"] = "noise__ADM_x_png"
    return record


def _sample(record, trace):
    return render_sample(record, "positive",
                         observation_text="模型视觉观察：纹理不一致。",
                         split_entry=None, trace=trace)


CLOSING = ('<reasoning>噪声残留专家给出的水平为 3.0900，属于中间带，'
           '不足以单独支撑判定，因此保持不确定。</reasoning>\n'
           '<verdict>{"verdict": "Uncertain", "confidence": 0.4,'
           ' "primary_evidence": [], "report": "证据有限。"}</verdict>')


class TestTheModelArgumentSurvives:
    def test_the_reasoning_block_carries_the_models_own_words(self):
        trace = _trace([
            {"from": "user", "value": "分析这张图像。"},
            {"from": "gpt", "value": CLOSING},
        ])
        sample = _sample(_record(), trace)
        answer = sample["conversations"][-1]["value"]
        assert "噪声残留专家给出的水平为 3.0900" in answer
        assert sample["fidelity"]["model_reasoning_verbatim"] is True
        assert sample["fidelity"]["reasoning_source"] == "model+bookkeeping"

    def test_the_bookkeeping_is_labelled_as_bookkeeping(self):
        trace = _trace([{"from": "gpt", "value": CLOSING}])
        answer = _sample(_record(), trace)["conversations"][-1]["value"]
        assert "证据簿记" in answer
        assert "未经改写" in answer

    def test_a_trace_without_the_block_says_so_instead_of_pretending(self):
        trace = _trace([{"from": "gpt", "value": '<verdict>{"verdict": "Fake",'
                                               ' "confidence": 0.9}</verdict>'}])
        sample = _sample(_record(), trace)
        assert sample["fidelity"]["model_reasoning_verbatim"] is False
        assert sample["fidelity"]["reasoning_source"] == "bookkeeping_only"

    def test_adopted_reasoning_reads_the_closing_turn(self):
        assert "不足以单独支撑判定" in adopted_reasoning(_trace([
            {"from": "gpt", "value": CLOSING}]))


class TestTheWholeDialogueIsAudited:
    FABRICATED_TURN = ('<reasoning>JPEG 压缩专家显示 blockiness 为 2.5，支持伪造。'
                       '</reasoning><verdict>{"verdict": "Fake", "confidence": 0.9}'
                       '</verdict>')

    def test_a_fabrication_in_an_earlier_turn_is_found(self):
        trace = _trace([
            {"from": "user", "value": "分析这张图像。"},
            {"from": "gpt", "value": self.FABRICATED_TURN},
            {"from": "user", "value": "[System: 你的结论引用了不存在的证据。]"},
            {"from": "gpt", "value": CLOSING},
        ])
        audit = audit_dialogue({**_record(), "_trace": trace})
        assert audit["turns"] == 1                       # the closing turn is replaced
        assert [finding["kind"] for finding in audit["defects"]] == [
            "fabricated_expert_result"]
        assert audit["defects"][0]["turn"] == 1

    def test_the_replaced_turn_is_not_audited_again(self):
        trace = _trace([{"from": "gpt", "value": self.FABRICATED_TURN}])
        assert audit_dialogue({**_record(), "_trace": trace})["defects"] == []

    def test_the_sample_records_what_the_audit_found(self):
        trace = _trace([
            {"from": "gpt", "value": self.FABRICATED_TURN},
            {"from": "gpt", "value": CLOSING},
        ])
        sample = _sample(_record(), trace)
        assert sample["fidelity"]["dialogue_audit"]["defect_count"] == 1
        assert sample["fidelity"]["dialogue_audit"]["defect_kinds"] == [
            "fabricated_expert_result"]

    def test_conversion_refuses_a_trajectory_with_a_defect(self, tmp_path):
        from scripts.build_final_v2 import build

        record = _record()
        trace = _trace([
            {"from": "gpt", "value": self.FABRICATED_TURN},
            {"from": "gpt", "value": CLOSING},
        ])
        trajectory_dir = tmp_path / "trajectories"
        trajectory_dir.mkdir()
        (trajectory_dir / "t.json").write_text(json.dumps(record), encoding="utf-8")
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        record["trace_path"] = str(trace_path)
        (trajectory_dir / "t.json").write_text(json.dumps(record), encoding="utf-8")
        admission = tmp_path / "admission.json"
        admission.write_text(json.dumps({"decisions": [
            {"trajectory_id": record["trajectory_id"], "category": "positive"}]}),
            encoding="utf-8")

        metadata = build(str(trajectory_dir), str(admission), str(tmp_path / "out"),
                         "g4_trial", variants_manifest=str(tmp_path / "none.json"))
        assert metadata["total"] == 0
        assert metadata["gate_reasons"] == {"supervised_turn_fabrication": 1}
        gated = json.loads((tmp_path / "out" / "sft_gated.json").read_text(encoding="utf-8"))
        assert gated[0]["kinds"] == ["fabricated_expert_result"]
        # the trajectory itself is untouched
        assert json.loads((trajectory_dir / "t.json").read_text(encoding="utf-8"))


class TestTheScreenerAsksTheLedger:
    def _sample_with(self, answer, tokens):
        record = _record()
        sample = render_sample(record, "positive", observation_text="观察。",
                               split_entry=None, trace=_trace([{"from": "gpt", "value": answer}]))
        sample["evidence_chain"] = tokens
        sample["conversations"] = [{"from": "gpt", "value": answer}]
        return sample

    def _token(self, source="noise_expert", raw=3.0912, p_fake=0.564, scope="global"):
        return {"source": source, "evidence_id": "E-1", "raw_metric": raw,
                "strength": raw, "measurement_scope": scope,
                "calibrated_likelihood": {"Real": round(1 - p_fake, 3), "Fake": p_fake}}

    def test_score_layer_mixing(self):
        from utils.weak_evidence import ledger_flags

        sample = self._sample_with("噪声残留专家的校准概率为 3.09。", [self._token()])
        findings = ledger_flags(sample)
        assert [f["kind"] for f in findings] == ["misattributed_field"]
        assert "分数层次混用" in findings[0]["reason"]

    def test_a_fabricated_tool(self):
        from utils.weak_evidence import ledger_flags

        sample = self._sample_with("JPEG 压缩专家显示 blockiness 为 2.5。", [self._token()])
        assert [f["kind"] for f in ledger_flags(sample)] == ["fabricated_expert_result"]

    def test_a_direction_read_the_wrong_way(self):
        from utils.weak_evidence import ledger_flags

        sample = self._sample_with("噪声残留专家显示 3.09，表明图像更可能真实。",
                                   [self._token(p_fake=0.95)])
        kinds = [f["kind"] for f in ledger_flags(sample)]
        assert kinds == ["direction_conflict"]

    def test_global_evidence_used_as_a_local_finding(self):
        from utils.weak_evidence import localisation_flags

        sample = self._sample_with("噪声残留专家在该区域显示出异常。", [self._token()])
        assert [f["kind"] for f in localisation_flags(sample)] == [
            "global_evidence_localised"]

    def test_a_label_that_contradicts_its_own_report(self):
        from utils.weak_evidence import label_report_flags

        answer = ('<reasoning>综合来看，噪声水平符合相机特征，图像是真实的。</reasoning>\n'
                  '<verdict>{"verdict": "Fake", "confidence": 0.9}</verdict>')
        sample = self._sample_with(answer, [self._token()])
        sample["final_verdict"] = {"verdict": "Fake", "confidence": 0.9}
        assert [f["kind"] for f in label_report_flags(sample)] == ["label_report_conflict"]

    def test_a_consistent_sample_is_not_flagged(self):
        from utils.weak_evidence import flags

        answer = ('<reasoning>噪声残留专家显示 3.09，属于中间带，不足以支撑判定。</reasoning>\n'
                  '<verdict>{"verdict": "Uncertain", "confidence": 0.4}</verdict>')
        sample = self._sample_with(answer, [self._token(p_fake=0.52)])
        sample["final_verdict"] = {"verdict": "Uncertain", "confidence": 0.4}
        assert flags(sample) == []


class TestRepeatRequestTurnsAreMarked:
    """A re-request is a no-progress action, not a model behaviour to copy."""

    TRACE = _trace([
        {"from": "user", "value": "分析。"},
        {"from": "gpt", "value": "<call_noise></call_noise>"},
        {"from": "user", "value": "[System: noise_expert 已在本会话对整图测量过"
                                  "（E-1），重复调用不会产生新的测量，已跳过执行。]"},
        {"from": "gpt", "value": CLOSING},
    ])

    def test_the_audit_names_the_turn(self):
        audit = audit_dialogue({**_record(), "_trace": self.TRACE})
        assert audit["repeat_request_turns"] == [1]

    def test_the_sample_counts_them(self):
        sample = _sample(_record(), self.TRACE)
        assert sample["fidelity"]["dialogue_audit"]["repeat_request_turns"] == [1]

    def test_the_screener_flags_the_sample_for_a_person(self):
        from utils.weak_evidence import repeat_request_flags

        sample = _sample(_record(), self.TRACE)
        findings = repeat_request_flags(sample)
        assert [f["kind"] for f in findings] == ["repeat_request_turn"]
        assert findings[0]["severity"] == "candidate"
