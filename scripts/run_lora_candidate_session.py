"""Load the smoke LoRA into the existing Agent and run one real session.

The session uses the same Qwen3-VL client, system prompt, PROBE/TruFor adapters
and controller as the candidate runs.  The adapter is the one saved by
`train_lora_candidate_smoke.py`; the image is `autosplice__45749__1__q95`,
which is not in the three training samples.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from experts.probe_dino import ProbeDinoExpert  # noqa: E402
from experts.trufor import TruForExpert  # noqa: E402
from mllm.message_builder import build_forensic_prompt  # noqa: E402
from mllm.qwen3_client import Qwen3VLClient  # noqa: E402
from state_machine.controller import ForensicStateMachine  # noqa: E402
from utils.logger import SessionLogger  # noqa: E402

OUT = ROOT / "sft_data/lora_smoke_20261128"
CANDIDATES = ROOT / "sft_data/sft_candidates_20261127"
SAMPLE_ID = "autosplice__45749__1__q95"


def main() -> int:
    if os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    os.environ.setdefault(
        "PROBE_WEIGHTS",
        "/root/autodl-tmp/third_party/probe/DINOv2_best_model_step_34999.pth")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    records = {row["sample_id"]: row for row in json.loads(
        (CANDIDATES / "records.json").read_text(encoding="utf-8"))}
    sample = records[SAMPLE_ID]
    adapter = OUT / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"adapter not found: {adapter}")

    prompt = build_forensic_prompt(("probe", "trufor"))
    client = Qwen3VLClient(
        system_prompt=prompt, adapter_path=str(adapter),
        inference_mode="non-thinking", do_sample=False, temperature=0.0,
        max_new_tokens=1024, local_files_only=True)
    experts = {"probe_dino_expert": ProbeDinoExpert(device="cuda"),
               "trufor_expert": TruForExpert(device="cuda")}
    for name, expert in experts.items():
        reason = expert.unavailable_reason()
        if reason:
            raise SystemExit(f"{name} unavailable: {reason}")

    logger = SessionLogger(sft_dir=str(OUT / "real_session/traces"))
    fsm = ForensicStateMachine(client, experts, logger=logger)
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = fsm.run(str(ROOT / sample["image_path"]),
                     ground_truth=sample["ground_truth"])
    elapsed = time.time() - started
    verdict = result.get("final_verdict") or {}
    trace_path = Path(result["sft_data_path"])
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    summary = {
        "schema": "lora_real_session/1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_id": SAMPLE_ID,
        "image_path": sample["image_path"],
        "ground_truth": sample["ground_truth"],
        "adapter": str(adapter),
        "model": {"id": client._model_id, "revision": client._revision,
                  "inference_mode": client._inference_mode,
                  "do_sample": client._do_sample},
        "elapsed_s": round(elapsed, 2),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "halting_reason": result.get("halting_reason"),
        "model_turns": result.get("model_turn_count"),
        "expert_calls": result.get("expert_call_count"),
        "final_verdict": verdict.get("verdict"),
        "whole_image_generation": verdict.get("whole_image_generation"),
        "local_manipulation": verdict.get("local_manipulation"),
        "suspicious_regions": verdict.get("suspicious_regions"),
        "certainty": verdict.get("certainty"),
        "report": verdict.get("report"),
        "primary_evidence": verdict.get("primary_evidence"),
        "evidence": [{
            "evidence_id": token.get("evidence_id"),
            "source": token.get("source"),
            "status": token.get("status"),
            "strength": token.get("strength"),
            "raw_metric": token.get("raw_metric"),
        } for token in result.get("evidence_chain") or []],
        "trace_path": str(trace_path.relative_to(ROOT)),
        "conversation_turns": len(trace.get("conversations") or []),
        "model_call_count": len(trace.get("metadata", {}).get("model_calls") or []),
    }
    (OUT / "real_session_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
