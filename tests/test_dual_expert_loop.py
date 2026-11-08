"""CPU end-to-end checks for the PROBE + TruFor joint loop."""

import json
import os

from PIL import Image

from main import build_pipeline
from mllm.message_builder import build_forensic_prompt, build_messages
from state_machine.controller import ForensicStateMachine
from utils.logger import SessionLogger

pytestmark = __import__("pytest").mark.core


def _run(tmp_path, name, label="Fake"):
    image = tmp_path / name
    Image.new("RGB", (64, 64), (90, 120, 60)).save(image)
    fsm = build_pipeline("mock", mode="probe_then_trufor",
                         tools=["probe", "trufor"])
    fsm._logger = SessionLogger(sft_dir=str(tmp_path / "traces"))
    return fsm.run(str(image), label)


def test_dual_prompt_declares_joint_fields_and_local_only_trufor():
    prompt = build_forensic_prompt(["probe", "trufor"])
    assert "<call_trufor>" in prompt
    assert "whole_image_generation" in prompt
    assert "local_manipulation" in prompt
    assert "suspicious_regions" in prompt
    assert "LOCAL manipulation" in prompt
    assert "visual_artifacts` are expert-produced images" in prompt
    # The old "human review, not evidence" contract is gone.
    assert "diagnostic products for human review" not in prompt
    # New joint contract: no mechanical overall, explicit bands/autonomy/colors.
    assert "Real only when" not in prompt
    assert "Call order is yours" in prompt
    assert "Not tampered" in prompt and "not proof of absence" in prompt
    assert "which question is uncertain" in prompt
    assert "missing measurement" in prompt
    assert "raw_band" in prompt
    flat = " ".join(prompt.split())
    assert "blue = low response" in flat
    assert "red = high response" in flat
    assert "SAME content" in prompt
    assert "pixel width/height" in flat
    assert "added later" in flat
    assert "low reliability" in prompt and "high reliability" in prompt
    assert "Strongest available evidence" not in prompt


def test_state_note_says_local_evidence_is_missing_before_trufor():
    note = ForensicStateMachine._state_note(
        [], ["probe_dino_expert", "trufor_expert"],
        {"probe_dino_expert": "E-probe"})
    assert "局部编辑（TruFor）：尚未测量" in note
    assert "整图生成（PROBE）：已测量" in note
    assert "Not applicable" in note
    # Once TruFor has measured, the hint must be gone.
    note = ForensicStateMachine._state_note(
        [], ["probe_dino_expert", "trufor_expert"],
        {"probe_dino_expert": "E-probe", "trufor_expert": "E-trufor"})
    assert "局部编辑（TruFor）：已测量" in note
    assert "局部编辑（TruFor）：尚未测量" not in note


def test_dual_mock_executes_both_experts_and_keeps_maps(tmp_path):
    result = _run(tmp_path, "fake.png")
    assert [row["source"] for row in result["evidence_chain"]] == [
        "probe_dino_expert", "trufor_expert"]
    trufor = result["evidence_chain"][1]
    assert trufor.get("status", "measured") == "measured"
    # The token's first line must not call a local score a whole-image score.
    assert "局部篡改检测分数" in trufor["phenomenon"]
    assert "整图判别分数" not in trufor["phenomenon"]
    artifacts = trufor["visual_artifacts"]
    assert any("anomaly_map" in path for path in artifacts)
    assert any("confidence_map" in path for path in artifacts)
    from state_machine.evidence_tokenizer import EvidenceTokenizer
    visible = EvidenceTokenizer.model_visible(trufor)
    assert visible["map_coordinates"]["space"] == "original_image"
    assert visible["map_coordinates"]["bbox_units"] == "normalized_0_1000"
    assert "image_size" not in visible["map_coordinates"]
    assert "Pixel width/height" in visible["map_coordinates"]["note"]
    assert "imgsize" not in visible["reading"]
    assert "SAME original-image region" in visible["artifact_meaning"]
    assert "original crop, anomaly-map crop, confidence-map crop" in \
        visible["artifact_meaning"]
    # The delivered trio is one content region at one content size.
    full_layout = trufor["artifact_layout"]
    model_layout = trufor["model_image_layout"]
    assert set(model_layout) == {"original_crop", "anomaly_map", "confidence_map"}
    boxes = {tuple(item["content_box"]) for item in model_layout.values()}
    sizes = {tuple(item["crop_size"]) for item in model_layout.values()}
    assert len(boxes) == 1 and len(sizes) == 1
    assert full_layout["anomaly_map"]["content_box"][1] > 0
    assert full_layout["anomaly_map"]["canvas_size"][1] > \
        full_layout["anomaly_map"]["content_box"][2]
    for name in ("original_crop", "anomaly_map", "confidence_map"):
        assert "content_box" not in visible["artifact_layout"][name]
        assert "pixels" not in visible["artifact_layout"][name]["content_area"]
        assert visible["artifact_layout"][name]["region_normalized_1000"]
    assert trufor["model_image_layout"]["anomaly_map"]["map_size"] == [64, 64]
    assert trufor["model_image_layout"]["anomaly_map"]["source_size"] == [64, 64]
    import cv2
    from config import PROJECT_ROOT
    assert all(os.path.exists(os.path.join(PROJECT_ROOT, path))
               for path in artifacts)
    anomaly = cv2.imread(os.path.join(
        PROJECT_ROOT, next(path for path in artifacts if "anomaly_map" in path)))
    box = model_layout["anomaly_map"]["content_box"]
    assert anomaly.shape[0] == model_layout["anomaly_map"]["canvas_size"][1]
    assert anomaly.shape[1] == model_layout["anomaly_map"]["canvas_size"][0]
    # Colour bar runs outside the content width: blue at the bottom, red at top.
    bar_x = box[2] + 28 + 10
    bar_top = box[1] + 25
    bar_bottom = box[1] + 20 + max(40, (box[3] - box[1]) - 40) - 5
    assert anomaly[bar_top, bar_x][2] > anomaly[bar_top, bar_x][0]   # red
    assert anomaly[bar_bottom, bar_x][0] > anomaly[bar_bottom, bar_x][2]  # blue
    delivered = [turn for turn in result["conversation"]
                 if turn.get("image_paths")]
    assert any(any("anomaly_map" in path for path in turn["image_paths"])
               for turn in delivered)
    # The map AND the confidence map must reach the model's image input, not
    # just the trace.  A two-image cap used to drop the confidence map.
    messages = build_messages(str(tmp_path / "fake.png"), result["conversation"],
                              system_prompt="test")
    trufor_turn = next(
        message for message in messages
        if message["role"] == "user"
        and '"expert": "trufor_expert"' in "".join(
            block.get("text", "") for block in message["content"]))
    assert sum(block["type"] == "image" for block in trufor_turn["content"]) == 3
    verdict = result["final_verdict"]
    assert verdict["whole_image_generation"] == "AI-generated"
    assert verdict["local_manipulation"] == "Not applicable"
    assert verdict["primary_evidence"] == ["probe", "trufor"]


def test_local_tamper_keeps_regions_separate_from_whole_image(tmp_path):
    auto = tmp_path / "AutoSplice"
    auto.mkdir()
    result = _run(auto, "local.jpg")
    verdict = result["final_verdict"]
    assert verdict["whole_image_generation"] == "Real"
    assert verdict["local_manipulation"] == "Tampered"
    assert "overall_from_dimensions" not in verdict
    assert "joint_overall_mismatch" not in verdict
    assert verdict["suspicious_regions"]
    assert verdict["suspicious_regions"][0]["bbox"] == [120, 80, 420, 360]


def test_closing_gap_matches_the_real_single_expert_abstentions():
    """Real 2026-11-08 traces: the model named an untried expert as its gap."""
    from pathlib import Path

    from config import PROJECT_ROOT
    from state_machine.controller import _closing_gap

    batch = Path(PROJECT_ROOT) / "sft_data/g8_acceptance/dual_20261108"
    records = {
        record["variant_id"]: record
        for record in json.loads((batch / "records.json").read_text(encoding="utf-8"))
    }
    cases = {
        # TruFor measured, PROBE never tried.
        "genimage__BigGAN__051_biggan_00020__q95": (
            "未调用 PROBE 无法判断整体是否为 AI 生成。", {"trufor_expert": "E-t"}, set()),
        "genimage__VQDM__VQDM_1000_200_00_055_vqdm_00039__q95": (
            "由于 PROBE 专家未调用，无法判断整图生成状态。", {"trufor_expert": "E-t"}, set()),
        "autosplice__58610__authentic__q95": (
            "PROBE measurement is required ... but it has not been called yet.",
            {"trufor_expert": "E-t"}, set()),
        "autosplice__41187__0__q95": (
            "整图生成性未被 PROBE 评估。", {"trufor_expert": "E-t"}, set()),
        # TruFor failed; PROBE is still untried and callable.
        "genimage__Midjourney__144_midjourney_34__q95": (
            "TruFor 专家因 RuntimeError 未能返回有效测量。PROBE 未被调用。",
            set(), {"trufor_expert"}),
    }
    for variant_id, (report, measured, failed) in cases.items():
        assert records[variant_id]["report"]
        gap = _closing_gap(
            report, {"verdict": "Uncertain"},
            ["probe_dino_expert", "trufor_expert"], measured, failed)
        assert gap is not None, variant_id
        assert gap["cited_experts"] == ["probe_dino_expert"], variant_id


def test_closing_gap_does_not_fire_on_a_justified_single_expert_verdict():
    from state_machine.controller import _closing_gap

    served = ["probe_dino_expert", "trufor_expert"]
    measured = {"trufor_expert": "E-t"}
    # A Real/Fake verdict is not an abstention, even if the report mentions PROBE.
    assert _closing_gap("未调用 PROBE，但 TruFor 已足够。",
                        {"verdict": "Real"}, served, measured, set()) is None
    # An Uncertain that explains why the untried tool does not apply is not the
    # "I did not call it" pattern this check exists for.
    assert _closing_gap("PROBE 未调用，但该工具不适用于当前局部问题。",
                        {"verdict": "Uncertain"}, served, measured, set()) is None
    # No remaining callable expert: nothing to ask the model to reconsider.
    assert _closing_gap("PROBE 未调用。", {"verdict": "Uncertain"},
                        served, {"probe_dino_expert": "E-p",
                                 "trufor_expert": "E-t"}, set()) is None


def test_preclose_check_can_lead_to_the_uncalled_expert(tmp_path):
    from main import build_pipeline
    from tests.test_controller import ScriptedMLLM

    image = tmp_path / "preclose-call.png"
    Image.new("RGB", (64, 64), (90, 120, 60)).save(image)
    fsm = build_pipeline("mock", mode="probe_then_trufor",
                         tools=["probe", "trufor"])
    fsm._mllm = ScriptedMLLM([
        "<planning>\nSuspected Region: none\n</planning>\n<call_trufor>",
        ('<reasoning>局部有候选，但整图未测。</reasoning>\n<verdict>'
         '{"verdict": "Uncertain", "whole_image_generation": "Uncertain",'
         ' "local_manipulation": "Uncertain", "suspicious_regions": [],'
         ' "confidence": 0.6, "primary_evidence": ["trufor_expert"],'
         ' "report": "未调用 PROBE，无法判断整图生成性。"}</verdict>'),
        "<planning>\nSuspected Region: none\n</planning>\n<call_probe>",
        ('<reasoning>两项测量都已返回。</reasoning>\n<verdict>'
         '{"verdict": "Uncertain", "whole_image_generation": "Uncertain",'
         ' "local_manipulation": "Uncertain", "suspicious_regions": [],'
         ' "confidence": 0.6, "primary_evidence": ["probe_dino_expert", "trufor_expert"],'
         ' "report": "PROBE 和 TruFor 均已测量；证据仍不足，但已无未尝试的适用工具。"}'
         '</verdict>'),
    ])
    fsm._logger = SessionLogger(sft_dir=str(tmp_path / "traces"))
    result = fsm.run(str(image), "Fake")
    notes = [turn["value"] for turn in result["conversation"]
             if turn["from"] == "user"]
    assert any("你的结案以" in note for note in notes)
    feedback = next(note for note in notes if "你的结案以" in note)
    assert "补测" in feedback and "最终结论仍由你决定" in feedback
    assert result["expert_call_count"] == 2
    assert result["final_verdict"]["verdict"] == "Uncertain"
    metadata = json.loads(
        open(result["sft_data_path"], encoding="utf-8").read())["metadata"]
    assert metadata["preclose_checks"][0]["after_check"] == "called"
    assert metadata["untried_experts"] == []
    assert metadata["abstention_cited_untried_expert"] == []


def test_preclose_check_keeps_the_models_verdict_when_it_does_not_call(tmp_path):
    from main import build_pipeline
    from tests.test_controller import ScriptedMLLM

    image = tmp_path / "preclose-keep.png"
    Image.new("RGB", (64, 64), (90, 120, 60)).save(image)
    fsm = build_pipeline("mock", mode="probe_then_trufor",
                         tools=["probe", "trufor"])
    abstain = ('<reasoning>仍缺少 PROBE。</reasoning>\n<verdict>'
               '{"verdict": "Uncertain", "whole_image_generation": "Uncertain",'
               ' "local_manipulation": "Uncertain", "suspicious_regions": [],'
               ' "confidence": 0.6, "primary_evidence": ["trufor_expert"],'
               ' "report": "未调用 PROBE，无法判断整图生成性。"}</verdict>')
    fsm._mllm = ScriptedMLLM([
        "<planning>\nSuspected Region: none\n</planning>\n<call_trufor>",
        abstain,
        abstain,
    ])
    fsm._logger = SessionLogger(sft_dir=str(tmp_path / "traces"))
    result = fsm.run(str(image), "Fake")
    notes = [turn["value"] for turn in result["conversation"]
             if turn["from"] == "user"]
    assert sum("你的结案以" in note for note in notes) == 1
    assert result["final_verdict"]["verdict"] == "Uncertain"
    metadata = json.loads(
        open(result["sft_data_path"], encoding="utf-8").read())["metadata"]
    assert metadata["preclose_checks"][0]["after_check"] == \
        "concluded_without_new_measurement"
    assert metadata["abstention_cited_untried_expert"] == ["probe_dino_expert"]


def test_state_note_separates_failed_from_never_tried():
    failed = {
        "source": "trufor_expert",
        "evidence_id": "E-failed-trufor_expert",
        "evidence_name": "manipulation_anomaly_score",
        "status": "failed",
        "reason": "RuntimeError: no npz",
        "measurement_scope": "global",
        "model_delivered": "slim",
    }
    note = ForensicStateMachine._state_note(
        [failed], ["probe_dino_expert", "trufor_expert"], {})
    assert "整图生成（PROBE）：尚未测量" in note
    assert "局部编辑（TruFor）：执行失败" in note
    assert "尚可调用（尚未尝试）：probe_dino_expert" in note
    assert "已尝试但失败：trufor_expert" in note


def test_region_triptych_resizes_maps_and_records_the_transform():
    import numpy as np

    from experts.trufor import MockTruForExpert

    expert = MockTruForExpert(shape=(32, 48))
    expert._last = expert._arrays()
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    panels, layout = expert.render_region_artifacts(
        image, [10, 20, 50, 60], [125, 250, 625, 750])
    assert set(panels) == {"original_crop", "anomaly_map", "confidence_map"}
    assert len({tuple(item["content_box"]) for item in layout.values()}) == 1
    assert {tuple(item["crop_size"]) for item in layout.values()} == {(40, 40)}
    anomaly = layout["anomaly_map"]
    assert anomaly["map_size"] == [32, 48]
    assert anomaly["source_size"] == [64, 80]
    assert anomaly["crop_size"] == [40, 40]
    assert anomaly["region_pixels"] == [10, 20, 50, 60]
    assert anomaly["region_normalized_1000"] == [125, 250, 625, 750]
    assert anomaly["scale_map_to_source"] == [round(80 / 48, 6),
                                               round(64 / 32, 6)]


def test_coverage_correction_keeps_original_feedback_and_reanswer(tmp_path):
    from main import build_pipeline
    from tests.test_controller import ScriptedMLLM

    image = tmp_path / "coverage.png"
    Image.new("RGB", (64, 64), (90, 120, 60)).save(image)
    fsm = build_pipeline("mock", mode="probe_then_trufor",
                         tools=["probe", "trufor"])
    bad = ('<reasoning>整图由 PROBE 支持；局部没有 TruFor。</reasoning>\n'
           '<verdict>{"verdict": "Fake", "whole_image_generation": "AI-generated",'
           ' "local_manipulation": "Not tampered", "suspicious_regions": [],'
           ' "confidence": 0.8, "primary_evidence": ["probe_dino_expert"],'
           ' "report": "整图 AI 生成；局部未发现篡改。"}</verdict>')
    corrected = ('<reasoning>局部问题未测量。</reasoning>\n'
                 '<verdict>{"verdict": "Fake", "whole_image_generation": "AI-generated",'
                 ' "local_manipulation": "Uncertain", "suspicious_regions": [],'
                 ' "confidence": 0.8, "primary_evidence": ["probe_dino_expert"],'
                 ' "report": "整图 AI 生成；TruFor 未测量，局部不确定。"}</verdict>')
    fsm._mllm = ScriptedMLLM([
        "<planning>\nSuspected Region: none\n</planning>\n<call_probe>",
        bad,
        corrected,
    ])
    fsm._logger = SessionLogger(sft_dir=str(tmp_path / "traces"))
    result = fsm.run(str(image), "Fake")
    notes = [turn["value"] for turn in result["conversation"]
             if turn["from"] == "user"]
    assert any("证据覆盖检查" in note for note in notes)
    assert result["final_verdict"]["local_manipulation"] == "Uncertain"
    check = result["final_verdict"]["coverage_check"]
    assert check["status"] == "pass_after_correction"
    assert "Not tampered" in check["original_output"]
    assert "局部问题未测" in check["feedback"]
    assert "Uncertain" in check["corrected_output"]
    assert check["remaining_issues"] == []


def test_coverage_correction_records_a_failed_reanswer_without_changing_label(tmp_path):
    from main import build_pipeline
    from tests.test_controller import ScriptedMLLM

    image = tmp_path / "coverage-fail.png"
    Image.new("RGB", (64, 64), (90, 120, 60)).save(image)
    fsm = build_pipeline("mock", mode="probe_then_trufor",
                         tools=["probe", "trufor"])
    bad = ('<verdict>{"verdict": "Fake", "whole_image_generation": "AI-generated",'
           ' "local_manipulation": "Not tampered", "suspicious_regions": [],'
           ' "confidence": 0.8, "primary_evidence": ["probe_dino_expert"],'
           ' "report": "整图 AI 生成；局部未发现篡改。"}</verdict>')
    fsm._mllm = ScriptedMLLM([
        "<planning>\nSuspected Region: none\n</planning>\n<call_probe>",
        bad,
        bad,
    ])
    fsm._logger = SessionLogger(sft_dir=str(tmp_path / "traces"))
    result = fsm.run(str(image), "Fake")
    check = result["final_verdict"]["coverage_check"]
    assert check["status"] == "failed"
    assert check["remaining_issues"]
    assert result["final_verdict"]["local_manipulation"] == "Not tampered"
    assert any("coverage: local_manipulation_without_trufor"
               in issue for issue in
               result["final_verdict"].get("joint_verdict_issues") or [])
    assert check["original_output"] and check["corrected_output"]


def test_region_triptych_full_frame_uses_closed_normalized_edges():
    import numpy as np

    from experts.trufor import MockTruForExpert

    expert = MockTruForExpert(shape=(32, 48))
    expert._last = expert._arrays()
    image = np.zeros((64, 80, 3), dtype=np.uint8)
    _panels, layout = expert.render_region_artifacts(image, [0, 0, 64, 80], None)
    assert layout["anomaly_map"]["region_normalized_1000"] == [0, 0, 1000, 1000]
    assert layout["anomaly_map"]["source_size"] == [64, 80]
