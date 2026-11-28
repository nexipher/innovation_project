"""Replay one fixed evidence prefix to base and LoRA Qwen3-VL.

The controller's autonomous loop is intentionally bypassed.  For each case the
script takes the user/assistant turns up to a caller-selected turn (normally the
last state note after both experts have measured), keeps the original image and
the recorded tool images, and appends one neutral instruction to conclude.  No
expert is executed again.  Both arms therefore see the same image paths, the
same tool-return text and the same three panels; only the adapter differs.

This is a diagnostic, not a new validation protocol: it answers whether a
model reads the *same delivered evidence* differently, independently of
whether it decided to call an expert.

Usage:
  FORENSIC_ALLOW_GPU=1 python scripts/replay_same_evidence.py \
      --selection sft_data/lora_small_20261128/replay_cases.json \
      --tag base --out sft_data/lora_small_20261128/replay_fixed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from mllm.message_builder import build_forensic_prompt  # noqa: E402
from mllm.qwen3_client import Qwen3VLClient  # noqa: E402
from utils.parser import Parser  # noqa: E402

DEFAULT_OUT = ROOT / "sft_data/lora_small_20261128/replay_fixed"
CONCLUDE = (
    "[System: 同一组证据已完整送达：原始图、实际工具返回和三张图都未改动。"
    "请只基于这些证据直接输出一个 <verdict>，不要调用任何专家，也不要"
    "改写、补造专家分数。]"
)
NO_CALL = (
    "[System: 本轮是固定证据复验，不再执行工具。请直接输出 <verdict>；"
    "不要输出任何 <call_x>。]"
)


def _load(selection: Path) -> list[dict]:
    payload = json.loads(selection.read_text(encoding="utf-8"))
    return payload["cases"]


def _item(turn: dict) -> dict:
    item = {"from": turn.get("from", "user"), "value": turn.get("value", "")}
    if turn.get("image_paths"):
        item["image_paths"] = list(turn["image_paths"])
    return item


def _history(trace: dict, cut: int | None = None) -> list[dict]:
    """Keep the common evidence prefix, not the source arm's own verdicts.

    With ``cut`` set, take the literal trace prefix (useful for auditing the
    controller's correction turn).  Without it, build a clean common prefix:
    the opening task, each expert's first call turn and evidence turn, and the
    immediately following state note.  Source-arm reasoning/verdicts and
    coverage-correction notes are omitted, so base and LoRA see exactly the
    same tool returns and panels.
    """
    turns = trace.get("conversations") or []
    if cut is not None:
        if cut < 0 or cut >= len(turns):
            raise ValueError(f"cut {cut} outside 0..{len(turns) - 1}")
        return [_item(turn) for turn in turns[:cut + 1]]
    if not turns or turns[0].get("from") != "user":
        raise ValueError("trace does not start with a user turn")
    indices = {0}
    seen = set()
    for index, turn in enumerate(turns):
        if turn.get("from") != "user" or not turn.get("image_paths"):
            continue
        try:
            token = json.loads(turn.get("value") or "{}")
        except ValueError:
            continue
        source = token.get("expert")
        if not source or source in seen:
            continue
        seen.add(source)
        if index > 0 and turns[index - 1].get("from") == "gpt":
            indices.add(index - 1)
        indices.add(index)
        if (index + 1 < len(turns)
                and turns[index + 1].get("from") == "user"
                and str(turns[index + 1].get("value", "")).startswith(
                    "[System: 本会话状态")):
            indices.add(index + 1)
    return [_item(turns[index]) for index in sorted(indices)]


def _images(history: list[dict]) -> list[str]:
    return [path for turn in history for path in (turn.get("image_paths") or [])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--tag", required=True, choices=("base", "lora"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    if not __import__("torch").cuda.is_available():
        raise SystemExit("CUDA unavailable")
    adapter = None
    if args.tag == "lora":
        if not args.adapter:
            raise SystemExit("--tag lora requires --adapter")
        adapter = str(Path(args.adapter).resolve())

    cases = _load(Path(args.selection))
    out = Path(args.out) / args.tag
    if out.exists():
        raise SystemExit(f"{out} exists; move it aside")
    out.mkdir(parents=True)
    client = Qwen3VLClient(
        system_prompt=build_forensic_prompt(("probe", "trufor")),
        adapter_path=adapter, inference_mode="non-thinking",
        do_sample=False, temperature=0.0, max_new_tokens=1024,
        local_files_only=True)
    records = []
    for index, case in enumerate(cases, 1):
        trace = json.loads((ROOT / case["trace"]).read_text(encoding="utf-8"))
        cut = case.get("cut")
        history = _history(trace, int(cut) if cut is not None else None)
        prefix_images = _images(history)
        history.append({"from": "user", "value": CONCLUDE})
        raw = client.generate(trace["image_path"], history)
        attempts = [raw]
        # A fixed replay must not turn into another tool-selection experiment.
        # If the first answer is a call, give the model one identical chance to
        # conclude; record both outputs either way.
        for _ in range(1):
            if "<call_" not in raw:
                break
            history.append({"from": "gpt", "value": raw})
            history.append({"from": "user", "value": NO_CALL})
            raw = client.generate(trace["image_path"], history)
            attempts.append(raw)
        verdict = Parser.parse_verdict(raw) or {}
        calls = [name for name, _ in Parser.executable_calls(raw)]
        record = {
            "sample_id": case["sample_id"],
            "trace": case["trace"],
            "cut": case.get("cut"),
            "image_path": trace["image_path"],
            "prefix_turns": len(history),
            "prefix_tool_images": prefix_images,
            "tool_images": len(prefix_images),
            "attempts": attempts,
            "call_written": calls,
            "verdict": verdict.get("verdict"),
            "whole_image_generation": verdict.get("whole_image_generation"),
            "local_manipulation": verdict.get("local_manipulation"),
            "certainty": verdict.get("certainty"),
            "report": verdict.get("report"),
        }
        records.append(record)
        print(f"[{args.tag} {index}/{len(cases)}] {case['sample_id']} "
              f"whole={record['whole_image_generation']} "
              f"local={record['local_manipulation']} "
              f"overall={record['verdict']} calls={calls}", flush=True)
    (out / "records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tag": args.tag, "cases": len(records),
                      "out": str(out / "records.json")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
