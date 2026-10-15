"""
The minimal review view for a batch (stage 4.5).

plan.md §4.5 asks for the source image, the evidence pack the model actually
received, the whole raw conversation, the conclusion and the external control
reason — before any larger generation.  Two properties are pinned here, both
learned from failures this project has already had:

* the view reads the **record + trace**, not the training pipeline: admission is
  the "may this teach the model" gate, and running it as an acceptance gate
  rejects everything (it knows the old tag names and demands a no-tool
  baseline), leaving the reviewer nothing to look at;
* the evidence round is rendered **as delivered** — the pack, field for field,
  not the background token it came from.
"""

import json
import os

import pytest

from config import PROJECT_ROOT
from scripts.g8_review_view import build, session_page

pytestmark = pytest.mark.core

IMAGE = "dataset/GenImage_Test/Midjourney/0_midjourney_169.png"

RECORD = {
    "trajectory_id": "g8_default__MJ_1_png",
    "variant_id": "MJ_1_png", "treatment": "png", "generator": "Midjourney",
    "source_id": "Midjourney/1", "ground_truth": "Fake",
    "final_verdict": "Fake", "model_candidate": "Fake",
    "verdict_status": "model_concluded", "confidence": 0.8,
    "halting_reason": "model_concluded", "policy_reasons": ["model_concluded"],
    "report": "整图判别分数指向生成。", "elapsed_s": 31.2,
    "counters": {"expert_calls": 1, "model_turns": 3, "weighted_cost": 2.5},
    "tools_served": ["npr_expert", "probe_dino_expert", "univfd_expert"],
    "evidence_check": {"status": "pass", "findings": [], "candidates": []},
    "image_path": IMAGE,
    "evidence": [{
        "evidence_id": "E-1", "source": "probe_dino_expert",
        "evidence_name": "aigi_discriminator_score", "strength": 0.9936,
        "raw_metric": 0.9936, "support": "AI-generated", "phenomenon": "整图判别分数 0.9936。",
        "reasoning": "后台长论证", "interpretation_text": "后台解释",
        "measurement_scope": "global", "status": "measured",
        "model_delivered": "slim",
    }],
}

PACK = ('{"evidence_id": "E-1", "expert": "probe_dino_expert", "measurement": '
        '"aigi_discriminator_score", "status": "measured", "scope": "global", '
        '"score": {"name": "aigi_discriminator_score", "value": 0.993582, '
        '"direction": "higher_is_fake", "direction_for_this_value": "AI-generated"}, '
        '"reading": "整图判别分数 0.9936。"}')

TRACE = {"conversations": [
    {"from": "user", "value": "<image>\n请分析这张图像的真实性。"},
    {"from": "gpt", "value": "<planning>Visual Observation: 平滑。\n</planning>\n"
                              "<call_probe>[0,0,1000,1000]</call_probe>"},
    {"from": "user", "value": PACK},
    {"from": "user", "value": "[System: 本会话状态 —— 已执行的专家：\n"
                              "  - probe_dino_expert（E-1）：aigi_discriminator_score=0.993582"
                              "（该值的方向：AI-generated）\n尚可调用：npr_expert。]"},
    {"from": "gpt", "value": '<reasoning>分数很高。</reasoning>\n'
                              '<verdict>{"verdict": "Fake", "confidence": 0.8}</verdict>'},
]}


class TestThePageShowsWhatWasDelivered:
    def test_the_evidence_round_is_the_pack_not_the_token(self, tmp_path):
        page = session_page(RECORD, TRACE, str(tmp_path), "index.html")
        assert "direction_for_this_value" in page
        # 后台字段不进页面：这正是精简包存在的理由
        for background in ("后台长论证", "后台解释", "interpretation_text"):
            assert background not in page, background

    def test_the_whole_conversation_is_there(self, tmp_path):
        page = session_page(RECORD, TRACE, str(tmp_path), "index.html")
        assert "&lt;image&gt;" in page            # 首轮任务提示（转义后原样）
        assert "本会话状态" in page               # 每次测量后的状态小结
        assert "call_probe" in page               # 调用轮

    def test_the_conclusion_and_the_control_reason_are_there(self, tmp_path):
        page = session_page(RECORD, TRACE, str(tmp_path), "index.html")
        assert "判定 Fake" in page
        assert "停止原因 model_concluded" in page
        assert "整图判别分数指向生成。" in page     # 模型自己的报告

    def test_the_image_is_referenced_relatively(self, tmp_path):
        import re

        page = session_page(RECORD, TRACE, str(tmp_path), "index.html")
        src = re.search(r'<img src="([^"]+)"', page).group(1)
        assert src.startswith("..")               # 相对引用，不拷图
        resolved = os.path.normpath(os.path.join(str(tmp_path), src))
        assert resolved == os.path.join(PROJECT_ROOT, IMAGE)


class TestTheBatchBuild:
    def _batch(self, tmp_path):
        """记录与 trace 都写在 tmp 里：测试不往仓库的批次目录写一个字节。"""
        records_dir = os.path.join(str(tmp_path), "records")
        os.makedirs(records_dir, exist_ok=True)
        trace_dir = os.path.join(str(tmp_path), "sessions")
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, "trace.json")
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(TRACE, handle, ensure_ascii=False)
        record = {**RECORD,
                  "trace_path": os.path.relpath(trace_path, PROJECT_ROOT)}
        with open(os.path.join(records_dir, "MJ_1_png__g8_default.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
        return records_dir

    def test_it_writes_an_index_and_a_page_per_session(self, tmp_path):
        records_dir = self._batch(tmp_path)
        out = str(tmp_path / "view")
        report = build("g8", "b_review_test", out, records_dir=records_dir)
        assert report["sessions"] == 1
        assert os.path.exists(os.path.join(out, "index.html"))
        assert os.path.exists(os.path.join(out, "MJ_1_png__g8_default.html"))

    def test_a_record_without_a_trace_is_reported_not_dropped_silently(self, tmp_path):
        records_dir = self._batch(tmp_path)
        with open(os.path.join(records_dir, "orphan__g8_default.json"),
                  "w", encoding="utf-8") as handle:
            json.dump({"trajectory_id": "orphan"}, handle)
        report = build("g8", "b_review_test", str(tmp_path / "view2"),
                       records_dir=records_dir)
        assert report["missing_trace"] == ["orphan__g8_default.json"]
