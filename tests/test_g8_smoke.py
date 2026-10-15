"""
The g8 smoke's five checks, run on CPU with the mock client.

This is the protocol half of stage 2 (plan.md §3b): before any GPU time is
spent on Qwen3-VL, the *plumbing* has to be provably right — the picture enters
the message as a picture, one legal call is executed, the number comes back in
a later turn, the conclusion parses with the image still in context, and the
raw reply / adopted reply / control reason are three separate records.

The GPU half (the same five checks against the real model, plus VRAM) cannot
run here; the test asserts the authorization gate that guards it instead.
"""

import json
import os
import subprocess
import sys

import pytest
from PIL import Image

from config import PROJECT_ROOT

# Protocol and message construction: core-tier material (TESTING.md), and
# fast — the mock client loads no weights.
pytestmark = pytest.mark.core


def _image(tmp_path):
    """A real PNG: the smoke loads it, and a generated one keeps the test
    independent of the dataset being mounted."""
    path = tmp_path / "smoke_input.png"
    Image.new("RGB", (96, 96), (120, 90, 200)).save(path)
    return path


@pytest.fixture(scope="module")
def smoke_report(tmp_path_factory):
    from scripts.g8_smoke import run

    work = tmp_path_factory.mktemp("g8_smoke")
    image = _image(work)
    return run(str(image), "mock", ["probe", "npr-stub"],
               inference_mode="non-thinking", greedy=True, out_dir=str(work))


class TestTheFiveChecks:
    def test_every_check_passes_on_the_mock(self, smoke_report):
        failed = {name: check["detail"] for name, check in smoke_report["checks"].items()
                  if check.get("applicable", True) and not check["ok"]}
        assert failed == {}, failed

    def test_the_visual_account_check_is_not_applicable_to_the_mock(self, smoke_report):
        """Only the real model makes a visual account; the mock's line is a
        template, so the check reports "not applicable" instead of passing."""
        assert smoke_report["checks"]["6_first_turn_has_a_specific_visual_account"][
            "applicable"] is False
        assert smoke_report["not_applicable"] == [
            "6_first_turn_has_a_specific_visual_account"]

    def test_the_image_is_inside_the_message_not_a_path(self, smoke_report):
        """A message carrying a path is not a message carrying a picture."""
        check = smoke_report["checks"]["1_image_enters_the_multimodal_message"]
        assert check["builder_image_parts"] == 1
        assert check["builder_payload_types"] == ["Image"]

    def test_the_tools_value_is_quoted_in_a_later_turn(self, smoke_report):
        from scripts.g8_smoke import STUB_SCORE

        check = smoke_report["checks"]["3_result_referenced_later"]
        assert check["ok"] is True
        # 桩的固定分在可接受集合里（顺序按证据链），且确实命中了一个
        assert any(isinstance(n, (int, float)) and abs(n - STUB_SCORE) < 1e-9
                   for n in check["needles"])
        assert check["matched"]

    def test_the_three_records_are_separate(self, smoke_report):
        check = smoke_report["checks"]["5_three_records_kept_separately"]
        assert check["raw_reply"] and check["adopted"] and check["control_reason"]
        # the raw reply is the model's own last turn, not the adopted verdict
        assert check["raw_reply"] != json.dumps(check["adopted"], ensure_ascii=False)

    def test_it_ran_two_tool_calls_and_named_them(self, smoke_report):
        check = smoke_report["checks"]["2_one_legal_call_executed"]
        assert check["executed_sources"] == ["probe_dino_expert", "npr_expert"]


class TestTheReportSaysWhatItWas:
    def test_the_report_records_the_client_and_the_toolbox(self, smoke_report):
        assert smoke_report["client"] == "mock"
        assert smoke_report["experts"] == ["npr_expert", "probe_dino_expert"]
        # CPU runs have no VRAM to report, and say so rather than writing 0
        assert smoke_report["vram_peak_gb"] is None

    def test_the_report_and_the_trace_are_written_where_it_was_told(self, smoke_report):
        report_path = os.path.join(PROJECT_ROOT, smoke_report["report_path"])
        assert os.path.exists(report_path)
        with open(report_path, encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert on_disk["passed"] is True
        # the full session record (evidence chain included) lands beside it
        assert smoke_report["session_record"]
        record = os.path.join(PROJECT_ROOT, smoke_report["session_record"])
        assert os.path.exists(record)
        with open(record, encoding="utf-8") as handle:
            logged = json.load(handle)
        assert logged["evidence_chain"], "会话记录里应当有证据链"

    def test_the_default_destination_is_the_smoke_directory_not_a_batch(self):
        """Products go to `sft_data/g8_smoke/` — no batch identity, no freeze gate."""
        from scripts.g8_smoke import OUTPUT_ROOT

        relative = os.path.relpath(OUTPUT_ROOT, PROJECT_ROOT).replace(os.sep, "/")
        assert relative == "sft_data/g8_smoke"
        assert "train/batches" not in relative


class TestTheGpuGate:
    def test_the_real_client_refuses_without_authorization(self, tmp_path):
        """
        GPU work needs the user's authorization, and the switch is explicit:
        without it the smoke must refuse *before* loading 17.5 GB of weights.
        """
        image = _image(tmp_path)
        result = subprocess.run(
            [sys.executable, "scripts/g8_smoke.py", "--client", "qwen3",
             "--image", str(image), "--out-dir", str(tmp_path / "out")],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=300)
        assert result.returncode != 0
        assert "FORENSIC_ALLOW_GPU" in result.stdout + result.stderr
        assert not (tmp_path / "out" / "smoke_report.json").exists()


class TestTheRealProcessorTurnsTheImageIntoTokens:
    """
    The image path, verified with the actual Qwen3-VL processor — on CPU, before
    any GPU time.  Skips when the pinned snapshot is not on this machine, the
    same way the batch tests skip when a batch is absent.
    """

    def test_the_placeholder_tokens_are_in_the_input(self, tmp_path):
        import pytest as _pytest

        from scripts.g8_smoke import check_the_processor_makes_image_tokens

        snapshot = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct")
        if not os.path.isdir(snapshot):
            _pytest.skip("Qwen3-VL 权重快照不在这台机器上")

        result = check_the_processor_makes_image_tokens(
            str(_image(tmp_path)), local_files_only=True)
        assert result["ok"] is True
        assert result["pil_images"] == 1
        assert result["image_tokens_in_input"] > 0
        # the two sources of the id must agree, or the counting means nothing
        assert result["image_token_id"] == result["image_token_id_from_tokenizer"]
        assert result["pixel_values_shape"] is not None
        # and the template really has no thinking switch (Instruct variant)
        assert result["template_has_thinking_switch"] is False

