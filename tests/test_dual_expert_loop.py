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
    assert "blue = low response" in prompt
    assert "low reliability" in prompt and "high reliability" in prompt
    assert "Strongest available evidence" not in prompt


def test_state_note_says_local_evidence_is_missing_before_trufor():
    note = ForensicStateMachine._state_note(
        [], ["probe_dino_expert", "trufor_expert"],
        {"probe_dino_expert": "E-probe"})
    assert "TruFor 尚未测量" in note
    assert "Not applicable" in note
    # Once TruFor has measured, the hint must be gone.
    note = ForensicStateMachine._state_note(
        [], ["probe_dino_expert", "trufor_expert"],
        {"probe_dino_expert": "E-probe", "trufor_expert": "E-trufor"})
    assert "TruFor 尚未测量" not in note


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
    assert visible["map_coordinates"]["image_size"] == [64, 64]
    assert visible["map_coordinates"]["space"] == "original_image"
    assert "diagnostic region crop, anomaly_map, confidence_map" in \
        visible["artifact_meaning"]
    layout = trufor["artifact_layout"]
    assert layout["anomaly_map"]["content_box"][1] > 0
    assert layout["anomaly_map"]["canvas_size"][1] > layout["anomaly_map"]["content_box"][2]
    assert visible["artifact_layout"]["anomaly_map"]["content_box"] == \
        layout["anomaly_map"]["content_box"]
    import cv2
    from config import PROJECT_ROOT
    assert all(os.path.exists(os.path.join(PROJECT_ROOT, path))
               for path in artifacts)
    anomaly = cv2.imread(os.path.join(
        PROJECT_ROOT, next(path for path in artifacts if "anomaly_map" in path)))
    box = layout["anomaly_map"]["content_box"]
    assert anomaly.shape[0] == layout["anomaly_map"]["canvas_size"][1]
    assert anomaly.shape[1] == layout["anomaly_map"]["canvas_size"][0]
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
