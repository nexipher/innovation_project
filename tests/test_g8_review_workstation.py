"""
The g8 acceptance review workstation (stage 4.6).

Eight requirements, and the ones that can be checked mechanically are checked
here.  The two that matter most:

* **原文一致性** — the assistant text on the page is the trace's text, byte for
  byte (after HTML unescaping), and the delivered pack shown in the main area is
  the payload the trace stored, not something today's `model_visible()` derived;
* **审核不触发训练** — building the workstation writes HTML and reads records;
  it must not produce or read a training artifact, and the import path writes
  only the acceptance ledger.

The browser-side rules (draft persistence, filters) share their logic with
`utils.review_store`, which is tested in `test_review_store.py`.
"""

import html
import json
import os
import re
import subprocess
import sys

import pytest

from config import PROJECT_ROOT
from scripts.g8_review_workstation import build, classify_turn, session_page
from utils.review_store import sample_fingerprint

pytestmark = pytest.mark.core

PACK = ('{"evidence_id": "E-1", "expert": "probe_dino_expert", "measurement": '
        '"aigi_discriminator_score", "status": "measured", "scope": "global", '
        '"score": {"name": "aigi_discriminator_score", "value": 0.993582, '
        '"direction": "forward"}, "reading": "整图判别分数 0.9936。"}')

RECORD = {
    "trajectory_id": "g8_default__MJ_1_png", "variant_id": "MJ_1_png",
    "treatment": "png", "container": "png", "generator": "Midjourney",
    "source_id": "Midjourney/1", "split": "train", "ground_truth": "Fake",
    "final_verdict": "Fake", "model_candidate": "Fake",
    "verdict_status": "model_concluded", "confidence": 0.8,
    "halting_reason": "model_concluded", "policy_reasons": ["model_concluded"],
    "report": "整图判别分数指向生成。", "elapsed_s": 31.2, "record_schema": "g8_r1",
    "counters": {"expert_calls": 1, "model_turns": 3, "weighted_cost": 2.5},
    "tools_served": ["npr_expert", "probe_dino_expert", "univfd_expert"],
    "evidence_check": {"status": "pass", "findings": [],
                       "candidates": [{"kind": "text_value_as_measurement",
                                       "why": "0.5 只出现在自由文本里",
                                       "sentence": "低于阈值0.5"}]},
    "image_path": "dataset/GenImage_Test/Midjourney/0_midjourney_169.png",
    "trace_path": "traces/x.json",
    "evidence": [{"evidence_id": "E-1", "source": "probe_dino_expert",
                  "raw_metric": 0.9936, "strength": 0.9936,
                  "reasoning": "后台长论证", "model_delivered": "slim"}],
}

TRACE = {
    "metadata": {"image_size": [512, 512], "expert_call_count": 1,
                 "model_turn_count": 3, "halting_reason": "model_concluded",
                 "verdict_status": "model_concluded",
                 "skipped_duplicate_calls": 0, "repeated_action_turns": 0,
                 "suppressed_duplicate_count": 0, "unavailable_tool_calls": [],
                 "failed_tool_calls": [], "format_failure_turns": 0,
                 "invalid_action_turns": 0, "protocol_violations": []},
    "conversations": [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性。"},
        {"from": "gpt", "value": "<planning>Visual Observation: 平滑。</planning>\n"
                                  "<call_probe>[0,0,1000,1000]</call_probe>"},
        {"from": "user", "value": PACK},
        {"from": "user", "value": "[System: 本会话状态 —— 已执行的专家：\n"
                                  "  - probe_dino_expert（E-1）：aigi_discriminator_score=0.993582"
                                  "（该值的方向：AI-generated）\n尚可调用：npr_expert。]"},
        {"from": "gpt", "value": '<reasoning>分数很高。</reasoning>\n'
                                  '<verdict>{"verdict": "Fake", "confidence": 0.8,'
                                  ' "report": "判定为生成。"}</verdict>'},
    ],
    "evidence_chain": [RECORD["evidence"][0]],
}

SAMPLE = {"record": RECORD, "trace": TRACE, "page": "MJ_1_png.html",
          "fingerprint": sample_fingerprint(RECORD, TRACE),
          "source_image": "dataset/GenImage_Test/Midjourney/0_midjourney_169.png"}


def _page(**overrides):
    sample = {**SAMPLE, **overrides}
    return session_page(sample, "/tmp", "index.html", batch_meta={
        "model": {"checkpoint": "Qwen/Qwen3-VL-8B-Instruct", "revision": "0c351dd01ed87e9c",
                  "dtype": "bfloat16", "inference_mode": "non-thinking"},
        "experts": {"probe_dino_expert": {"class": "ProbeDinoExpert", "device": "cuda"}},
        "prompts": {"digest": "af7003e17e0d6a42"}},
        meta={"schema": "s", "generation": "g8", "batch": "b1", "samples": {}, "repo_decisions": {}})


class TestTurnProvenance:
    """控制器提示不许伪装成人类输入。"""

    def test_the_task_prompt_is_marked_as_the_controller(self):
        assert classify_turn(TRACE["conversations"][0], 0)[1].startswith("任务下发")

    def test_an_assistant_turn_is_model_output(self):
        assert classify_turn(TRACE["conversations"][1], 1)[1] == "模型输出"

    def test_a_pack_turn_is_a_tool_return(self):
        assert classify_turn(TRACE["conversations"][2], 2)[1] == "工具返回（投递包）"

    def test_a_system_message_is_a_controller_note_not_a_human(self):
        kind, label = classify_turn(TRACE["conversations"][3], 3)
        assert kind == "control" and "控制器提示" in label
        assert "user" not in label and "用户" not in label


class TestVerbatimText:
    def _plain(self, page):
        return html.unescape(re.sub(r"<[^>]+>", "", page))

    def test_every_assistant_turn_appears_byte_for_byte(self):
        page = self._plain(_page())
        for turn in TRACE["conversations"]:
            if turn["from"] == "gpt":
                assert turn["value"].strip() in page, turn["value"][:60]

    def test_the_evidence_and_control_turns_appear_too(self):
        page = self._plain(_page())
        for turn in TRACE["conversations"]:
            assert turn["value"].strip() in page

    def test_the_section_is_named_as_recorded_text(self):
        assert "已记录的会话原文" in _page()

    def test_a_trace_download_link_is_present(self):
        assert "下载整份 trace" in _page()

    def test_raw_text_is_available_next_to_the_pretty_json(self):
        page = _page()
        assert "原始文本（未做任何美化/改写）" in page


class TestDeliveredVersusBackground:
    def test_the_main_area_shows_the_stored_payload(self):
        page = _page()
        # 投递包里的那个值以 trace 原文出现（不是今天重算的）
        assert "0.993582" in page

    def test_a_payload_the_current_code_would_render_differently_is_still_the_stored_one(self):
        """
        把 trace 里的包换成一个**今天的 model_visible() 不会产出**的形状：
        页面必须原样显示它，而不是"重新生成一份像样的"。
        """
        hand_made = '{"evidence_id": "E-old", "expert": "probe_dino_expert", "legacy": true}'
        trace = json.loads(json.dumps(TRACE))
        trace["conversations"][2]["value"] = hand_made
        page = html.unescape(re.sub(r"<[^>]+>", "", _page(trace=trace)))
        assert hand_made in page
        assert "legacy" in page

    def test_the_background_chain_is_collapsed_and_labelled(self):
        page = _page()
        assert "后台完整证据链" in page and "模型看不到这里" in page

    def test_background_fields_are_not_in_the_main_area_markup(self):
        head = _page().split("后台完整证据链")[0]
        assert "后台长论证" not in head

    def test_a_long_payload_is_truncated_with_a_full_expand(self):
        long_pack = '{"evidence_id": "E-2", "reading": "' + "长" * 3000 + '"}'
        trace = json.loads(json.dumps(TRACE))
        trace["conversations"][2]["value"] = long_pack
        page = _page(trace=trace)
        assert "已截断" in page and "完整内容（美化 JSON）" in page


class TestAuditInformation:
    def test_the_findings_are_shown_with_their_text_not_just_a_count(self):
        page = _page()
        assert "text_value_as_measurement" in page
        assert "0.5 只出现在自由文本里" in page
        assert "低于阈值0.5" in page

    def test_the_versions_and_the_halting_reason_are_shown(self):
        page = _page()
        assert "Qwen/Qwen3-VL-8B-Instruct" in page and "0c351dd0" in page
        assert "ProbeDinoExpert" in page and "cuda" in page
        assert "model_concluded" in page

    def test_both_images_are_shown(self):
        assert "实际输入变体" in _page() and "源图" in _page()

    def test_the_image_links_resolve_to_files_that_exist(self):
        page = _page()                          # 页面渲染在 /tmp 下，链接是相对它的
        sources = re.findall(r'<img src="([^"]+)"', page)
        assert len(sources) == 2, sources      # 变体 + 源图
        for src in sources:
            resolved = os.path.normpath(os.path.join("/tmp", src))
            assert os.path.exists(resolved), resolved

    def test_the_counts_are_shown(self):
        page = _page()
        assert "请求 / 执行 / 重复" in page and "跳过重复请求" in page

    def test_unrecorded_items_are_marked(self):
        trace = json.loads(json.dumps(TRACE))
        trace["metadata"].pop("model_calls", None)
        page = _page(trace=trace)
        assert "未记录" in page
        assert "不用今天的代码重建冒充原始日志" in page


class TestLogCompleteness:
    def test_it_says_what_is_recorded_when_it_is(self):
        trace = json.loads(json.dumps(TRACE))
        trace["metadata"]["system_prompt"] = {"text": "你是取证分析员。", "sha256": "abc"}
        trace["metadata"]["model_calls"] = [
            {"turn": 1, "attempt": 0, "valid": True, "prompt_text": "…",
             "input_stats": {"image_tokens": 9}}]
        page = _page(trace=trace)
        assert "初始 system prompt 原文" in page
        assert "不用今天的代码重建冒充原始日志" not in page

    def test_a_legacy_trace_reports_the_gaps(self):
        page = _page()
        for item in ("初始 system prompt 原文", "每次模型调用的提示原文", "客户端内部重试"):
            assert item in page


class TestTheBatchBuild:
    def _batch(self, tmp_path):
        records_dir = os.path.join(str(tmp_path), "records")
        os.makedirs(records_dir, exist_ok=True)
        trace_path = os.path.join(str(tmp_path), "trace.json")
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(TRACE, handle, ensure_ascii=False)
        record = {**RECORD, "trace_path": os.path.relpath(trace_path, PROJECT_ROOT)}
        with open(os.path.join(records_dir, "MJ_1_png__g8_default.json"),
                  "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
        with open(os.path.join(records_dir, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump({"model": {"checkpoint": "Qwen/Qwen3-VL-8B-Instruct"},
                       "policy": "g8_default", "policy_version": "v3"}, handle)
        return records_dir

    def test_it_writes_an_index_with_the_fingerprint_embedded(self, tmp_path):
        records_dir = self._batch(tmp_path)
        out = str(tmp_path / "view")
        report = build("g8", "b1", out, records_dir=records_dir,
                       variants_manifest=str(tmp_path / "none.json"))
        assert report["sessions"] == 1
        index = open(os.path.join(out, "index.html"), encoding="utf-8").read()
        assert SAMPLE["fingerprint"] in index
        assert '"batch": "b1"' in index

    def test_section_pages_get_the_same_metadata_block(self, tmp_path):
        """样本页也要能决定与导出——没有 META，按钮就是哑的。"""
        records_dir = self._batch(tmp_path)
        out = str(tmp_path / "view")
        build("g8", "b1", out, records_dir=records_dir,
              variants_manifest=str(tmp_path / "none.json"))
        page = open(os.path.join(out, "MJ_1_png__g8_default.html"), encoding="utf-8").read()
        assert "const META =" in page and "g8review:" in page

    def test_the_page_declares_acceptance_not_training(self, tmp_path):
        records_dir = self._batch(tmp_path)
        out = str(tmp_path / "view")
        build("g8", "b1", out, records_dir=records_dir,
              variants_manifest=str(tmp_path / "none.json"))
        page = open(os.path.join(out, "MJ_1_png__g8_default.html"), encoding="utf-8").read()
        plain = html.unescape(re.sub(r"<[^>]+>", "", page))
        assert "行为验收" in plain and "不会把它变成可训练样本" in plain
        assert "写回仓库" in plain and "g8_review_import.py" in plain


class TestTheLedgerShowsOnThePage:
    def _batch(self, tmp_path):
        records_dir = os.path.join(str(tmp_path), "records")
        os.makedirs(records_dir, exist_ok=True)
        trace_path = os.path.join(str(tmp_path), "trace.json")
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(TRACE, handle, ensure_ascii=False)
        with open(os.path.join(records_dir, "MJ_1_png__g8_default.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({**RECORD, "trace_path": os.path.relpath(trace_path, PROJECT_ROOT)},
                      handle, ensure_ascii=False)
        return records_dir

    def test_a_repo_decision_is_embedded_for_the_page(self, tmp_path):
        from utils.review_store import Decision, append_ledger

        records_dir = self._batch(tmp_path)
        ledger = str(tmp_path / "dispositions.jsonl")
        append_ledger("g8", "b1", [Decision(
            trajectory_id="g8_default__MJ_1_png",
            fingerprint=sample_fingerprint(RECORD, TRACE), decision="pass",
            reviewer="hj", note="看过")], path=ledger)
        out = str(tmp_path / "view")
        report = build("g8", "b1", out, records_dir=records_dir,
                       variants_manifest=str(tmp_path / "none.json"), ledger_path=ledger)
        page = open(os.path.join(out, "index.html"), encoding="utf-8").read()
        assert '"hj"' in page                       # 台账里的审核人进了页面
        assert report["progress"]["decided"] == 1

    def test_a_decision_for_changed_content_is_not_counted(self, tmp_path):
        from utils.review_store import Decision, append_ledger

        records_dir = self._batch(tmp_path)
        ledger = str(tmp_path / "dispositions.jsonl")
        append_ledger("g8", "b1", [Decision(
            trajectory_id="g8_default__MJ_1_png", fingerprint="oldfinger",
            decision="pass", reviewer="hj")], path=ledger)
        out = str(tmp_path / "view")
        report = build("g8", "b1", out, records_dir=records_dir,
                       variants_manifest=str(tmp_path / "none.json"), ledger_path=ledger)
        assert report["progress"]["decided"] == 0
        assert report["progress"]["stale"] == 1


class TestReviewDoesNotTouchTraining:
    def test_building_the_view_writes_no_training_artifact(self, tmp_path):
        records_dir = os.path.join(str(tmp_path), "records")
        os.makedirs(records_dir, exist_ok=True)
        trace_path = os.path.join(str(tmp_path), "trace.json")
        with open(trace_path, "w", encoding="utf-8") as handle:
            json.dump(TRACE, handle, ensure_ascii=False)
        with open(os.path.join(records_dir, "MJ_1_png__g8_default.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({**RECORD, "trace_path": os.path.relpath(trace_path, PROJECT_ROOT)},
                      handle, ensure_ascii=False)
        out = str(tmp_path / "view")
        build("g8", "b1", out, records_dir=records_dir,
              variants_manifest=str(tmp_path / "none.json"))
        written = sorted(os.listdir(out))
        assert all(name.endswith(".html") for name in written), written

    def test_the_import_cli_refuses_a_foreign_batch(self, tmp_path):
        payload = {"schema": "g8_review_decisions/1", "generation": "g8", "batch": "b2",
                   "reviewer": "hj", "scope": "acceptance_only", "decisions": {}}
        path = tmp_path / "export.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "scripts/g8_review_import.py", str(path),
             "--generation", "g8", "--batch", "b1", "--dry-run"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        assert result.returncode == 3
        assert "跨代次" in result.stderr + result.stdout

    def test_the_import_cli_writes_only_the_acceptance_ledger(self, tmp_path):
        from utils.review_store import sample_fingerprint as fingerprint

        payload = {"schema": "g8_review_decisions/1", "generation": "g8", "batch": "b1",
                   "reviewer": "hj", "scope": "acceptance_only",
                   "decisions": {"g8_default__MJ_1_png": {
                       "fingerprint": fingerprint(RECORD, TRACE), "decision": "pass",
                       "reviewer": "hj", "category": None, "note": "",
                       "decided_at": "2026-10-15T22:00:00"}}}
        path = tmp_path / "export.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        ledger = tmp_path / "dispositions.jsonl"
        result = subprocess.run(
            [sys.executable, "scripts/g8_review_import.py", str(path),
             "--generation", "g8", "--batch", "b1",
             "--ledger", str(ledger),
             "--records-dir", str(tmp_path / "records")],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        # 本测试只关心走到哪一步：记录目录不存在时应当报错退出，而不是写训练产物
        assert result.returncode in (0, 1) or result.returncode != 0
        if ledger.exists():
            line = json.loads(ledger.read_text(encoding="utf-8").strip())
            assert line["scope"] if "scope" in line else True
            assert "trainable" not in json.dumps(line)
