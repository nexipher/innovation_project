"""
The g8 acceptance review store (stage 4.6).

The rules here decide **whose review a decision belongs to** — get them wrong and
hours of someone's work silently attaches to a different batch, or a rebuilt page
quietly inherits decisions made about different content.  So the invariants live
in Python with tests, and the page's JS is only a draft client.
"""

import json

import pytest

from utils.review_store import (
    CATEGORIES,
    SCHEMA,
    Decision,
    append_ledger,
    check_import,
    export_payload,
    latest,
    load_ledger,
    progress,
    sample_fingerprint,
)

pytestmark = pytest.mark.core

RECORD = {
    "trajectory_id": "g8_default__MJ_1_png", "variant_id": "MJ_1_png",
    "treatment": "png", "container": "png", "generator": "Midjourney",
    "source_id": "Midjourney/1", "split": "train", "ground_truth": "Fake",
    "final_verdict": "Fake", "model_candidate": "Fake",
    "verdict_status": "model_concluded", "confidence": 0.8,
    "halting_reason": "model_concluded", "policy_reasons": ["model_concluded"],
    "report": "分数指向生成。", "elapsed_s": 31.2, "record_schema": "g8_r1",
    "counters": {"expert_calls": 1, "model_turns": 3, "weighted_cost": 2.5},
    "tools_served": ["probe_dino_expert"],
    "evidence_check": {"status": "pass", "findings": [], "candidates": []},
    "image_path": "dataset/GenImage_Test/Midjourney/0_midjourney_169.png",
    "evidence": [{"evidence_id": "E-1", "source": "probe_dino_expert",
                  "raw_metric": 0.99, "model_delivered": "slim"}],
}

TRACE = {
    "metadata": {"image_size": [512, 512], "expert_call_count": 1,
                 "model_turn_count": 3, "halting_reason": "model_concluded",
                 "verdict_status": "model_concluded"},
    "conversations": [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性。"},
        {"from": "gpt", "value": "<planning>…</planning>\n<call_probe>[0,0,1000,1000]</call_probe>"},
        {"from": "user", "value": '{"evidence_id": "E-1", "expert": "probe_dino_expert"}'},
        {"from": "gpt", "value": '<reasoning>…</reasoning>\n<verdict>{"verdict": "Fake"}</verdict>'},
    ],
}


class TestTheFingerprint:
    def test_the_same_content_gives_the_same_fingerprint(self):
        """重渲染（换页面、换排版、重新生成 HTML）不该让谁的决定失效。"""
        assert sample_fingerprint(RECORD, TRACE) == sample_fingerprint(
            dict(RECORD), json.loads(json.dumps(TRACE)))

    def test_the_page_version_does_not_enter_the_fingerprint(self):
        """指纹只覆盖样本内容；页面自己的版本号不在其中（改排版不算改内容）。"""
        blob = json.dumps({"record": RECORD, "trace": TRACE}, ensure_ascii=False)
        assert "page_version" not in blob

    @pytest.mark.parametrize("change", [
        {"final_verdict": "Real"},
        {"halting_reason": "budget_exhausted"},
        {"report": "另一段报告。"},
        {"image_path": "dataset/Real/other.jpg"},
    ])
    def test_a_changed_fact_changes_the_fingerprint(self, change):
        assert sample_fingerprint({**RECORD, **change}, TRACE) != \
            sample_fingerprint(RECORD, TRACE)

    def test_a_changed_conversation_changes_the_fingerprint(self):
        other = json.loads(json.dumps(TRACE))
        other["conversations"][1]["value"] = "改了模型的一句话"
        assert sample_fingerprint(RECORD, other) != sample_fingerprint(RECORD, TRACE)

    def test_a_changed_background_evidence_field_changes_the_fingerprint(self):
        other = {**RECORD, "evidence": [{**RECORD["evidence"][0], "raw_metric": 0.5}]}
        assert sample_fingerprint(other, TRACE) != sample_fingerprint(RECORD, TRACE)


class TestTheDecisionRules:
    def _decision(self, **overrides):
        base = dict(trajectory_id="g8_default__MJ_1_png",
                    fingerprint="abc123", decision="pass", reviewer="hj")
        base.update(overrides)
        return Decision(**base)

    def test_a_pass_needs_no_reason(self):
        assert self._decision().validate() == []

    @pytest.mark.parametrize("decision", ["revise", "reject"])
    def test_revise_and_reject_need_a_reason(self, decision):
        problems = self._decision(decision=decision, note="").validate()
        assert any("必须填写原因" in problem for problem in problems)

    def test_a_reason_alone_is_enough(self):
        assert self._decision(decision="revise", note="视觉描述是套话").validate() == []

    def test_an_anonymous_decision_is_refused(self):
        assert any("审核人" in problem for problem in
                   self._decision(reviewer="  ").validate())

    def test_an_unknown_category_is_refused(self):
        assert any("未知问题分类" in problem for problem in
                   self._decision(category="我瞎写的").validate())

    def test_every_category_is_a_string(self):
        assert all(isinstance(name, str) and name for name in CATEGORIES)


class TestImportRejectsWhatItCannotOwn:
    def _payload(self, **overrides):
        payload = export_payload("g8", "b1", "hj", {
            "g8_default__MJ_1_png": Decision(
                trajectory_id="g8_default__MJ_1_png",
                fingerprint=sample_fingerprint(RECORD, TRACE),
                decision="pass", reviewer="hj")})
        payload.update(overrides)
        return payload

    def _fingerprints(self):
        return {"g8_default__MJ_1_png": sample_fingerprint(RECORD, TRACE)}

    def test_a_matching_import_is_accepted(self):
        accepted, rejected = check_import(self._payload(), "g8", "b1", self._fingerprints())
        assert [d.trajectory_id for d in accepted] == ["g8_default__MJ_1_png"]
        assert rejected == [] and accepted[0].stale is False

    def test_another_generation_is_refused_whole(self):
        accepted, rejected = check_import(self._payload(), "g9", "b1", self._fingerprints())
        assert accepted == []
        assert "跨代次" in rejected[0]["reason"]

    def test_another_batch_is_refused_whole(self):
        accepted, rejected = check_import(self._payload(), "g8", "b2", self._fingerprints())
        assert accepted == []
        assert "跨批次" in rejected[0]["reason"]

    def test_an_unknown_sample_is_refused_by_name(self):
        payload = self._payload()
        payload["decisions"]["nobody__knows"] = payload["decisions"]["g8_default__MJ_1_png"]
        accepted, rejected = check_import(payload, "g8", "b1", self._fingerprints())
        assert [d.trajectory_id for d in accepted] == ["g8_default__MJ_1_png"]
        assert rejected[0]["trajectory_id"] == "nobody__knows"

    def test_a_changed_sample_marks_the_decision_stale_not_inherited(self):
        moved = {**RECORD, "report": "报告被改过。"}
        accepted, rejected = check_import(self._payload(), "g8", "b1",
                                          {"g8_default__MJ_1_png":
                                           sample_fingerprint(moved, TRACE)})
        assert rejected == []
        assert accepted[0].stale is True
        assert "不自动继承" in accepted[0].stale_reason

    def test_a_wrong_schema_is_refused(self):
        accepted, rejected = check_import(self._payload(schema="something/else"),
                                          "g8", "b1", self._fingerprints())
        assert accepted == [] and SCHEMA in rejected[0]["reason"]

    def test_a_revise_without_a_reason_is_refused_at_import(self):
        payload = self._payload()
        payload["decisions"]["g8_default__MJ_1_png"]["decision"] = "revise"
        payload["decisions"]["g8_default__MJ_1_png"]["note"] = ""
        accepted, rejected = check_import(payload, "g8", "b1", self._fingerprints())
        assert accepted == [] and "必须填写原因" in rejected[0]["reason"]

    def test_a_training_flag_in_the_file_is_refused(self):
        """验收不是训练：文件里出现"可训练"这类用途声明，一律拒收。"""
        accepted, rejected = check_import(self._payload(scope="trainable"),
                                          "g8", "b1", self._fingerprints())
        assert accepted == [] and "未知用途" in rejected[0]["reason"]


class TestTheLedgerIsAppendOnly:
    def test_writing_and_reading_back(self, tmp_path):
        path = str(tmp_path / "dispositions.jsonl")
        first = Decision(trajectory_id="a", fingerprint="f1", decision="revise",
                         reviewer="hj", category="visual_grounding", note="套话")
        append_ledger("g8", "b1", [first], path=path)
        second = Decision(trajectory_id="a", fingerprint="f1", decision="pass",
                          reviewer="hj", note="改好了")
        append_ledger("g8", "b1", [second], path=path)
        loaded = load_ledger("g8", "b1", path=path)
        assert [d.decision for d in loaded] == ["revise", "pass"]
        assert latest(loaded)["a"].decision == "pass"

    def test_a_line_carries_the_batch_identity(self, tmp_path):
        path = str(tmp_path / "dispositions.jsonl")
        append_ledger("g8", "b1", [Decision(trajectory_id="a", fingerprint="f1",
                                            decision="pass", reviewer="hj")], path=path)
        line = json.loads(open(path, encoding="utf-8").read().strip())
        assert line["generation"] == "g8" and line["batch"] == "b1"
        assert "trainable" not in json.dumps(line)


class TestProgress:
    def test_counts_and_stale_are_reported_separately(self):
        decisions = [
            Decision(trajectory_id="a", fingerprint="f", decision="pass", reviewer="hj"),
            Decision(trajectory_id="b", fingerprint="f", decision="reject",
                     reviewer="hj", category="protocol", note="两个动作标签"),
            Decision(trajectory_id="c", fingerprint="old", decision="pass",
                     reviewer="hj", stale=True, stale_reason="内容已变"),
        ]
        state = progress(decisions, total=4)
        assert state["decided"] == 2 and state["undecided"] == 2
        assert state["stale"] == 1
        assert state["by_decision"] == {"pass": 1, "revise": 0, "reject": 1}
        assert state["by_category"] == {"protocol": 1}
