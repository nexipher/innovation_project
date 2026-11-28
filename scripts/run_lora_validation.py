"""Run the 30-image validation set with the base or LoRA Qwen3-VL.

Both arms use the same autonomous Agent loop, system prompt (`probe+trufor`),
PROBE/TruFor adapters, greedy decoding and non-thinking mode.  The only
difference is the optional LoRA adapter.  Sessions run sequentially; any
session/tool error stops the run and is written to `failures.json`.

Usage:
  FORENSIC_ALLOW_GPU=1 python scripts/run_lora_validation.py --tag base
  FORENSIC_ALLOW_GPU=1 python scripts/run_lora_validation.py --tag lora \
      --adapter sft_data/lora_small_20261128/adapter
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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

SMALL = ROOT / "sft_data/lora_small_20261128"
MANIFEST = SMALL / "validation_manifest.json"
DEFAULT_OUT = SMALL / "validation"


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, choices=("base", "lora"))
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
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
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    samples = manifest["samples"]
    out = Path(args.out)
    tag_dir = out / args.tag
    if tag_dir.exists():
        # Refuse to append to a previous run; the comparison needs one clean arm.
        raise SystemExit(f"{tag_dir} exists; move it aside or use a new --out")
    tag_dir.mkdir(parents=True, exist_ok=True)
    adapter = args.adapter
    if args.tag == "lora":
        if not adapter:
            raise SystemExit("--tag lora requires --adapter")
        adapter = str(Path(adapter).resolve())

    prompt = build_forensic_prompt(("probe", "trufor"))
    client = Qwen3VLClient(
        system_prompt=prompt, adapter_path=adapter,
        inference_mode="non-thinking", do_sample=False, temperature=0.0,
        max_new_tokens=1024, local_files_only=True)
    experts = {"probe_dino_expert": ProbeDinoExpert(device="cuda"),
               "trufor_expert": TruForExpert(device="cuda")}
    for name, expert in experts.items():
        reason = expert.unavailable_reason()
        if reason:
            raise SystemExit(f"{name} unavailable: {reason}")

    torch.cuda.reset_peak_memory_stats()
    records, failures = [], []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        logger = SessionLogger(sft_dir=str(tag_dir / "traces"))
        fsm = ForensicStateMachine(client, experts, logger=logger)
        t0 = time.perf_counter()
        try:
            result = fsm.run(str(ROOT / sample["path"]),
                             ground_truth=sample["ground_truth"])
            elapsed = time.perf_counter() - t0
            verdict = result.get("final_verdict") or {}
            trace_path = Path(result["sft_data_path"]).resolve()
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            record = {
                "index": index,
                "sample_id": sample["sample_id"],
                "source_id": sample["source_id"],
                "group": sample["group"],
                "family": sample["family"],
                "ground_truth": sample["ground_truth"],
                "task_truth": sample["task_truth"],
                "final_verdict": verdict.get("verdict"),
                "whole_image_generation": verdict.get("whole_image_generation"),
                "local_manipulation": verdict.get("local_manipulation"),
                "certainty": verdict.get("certainty"),
                "report": verdict.get("report"),
                "primary_evidence": verdict.get("primary_evidence"),
                "suspicious_regions": verdict.get("suspicious_regions"),
                "coverage_check": verdict.get("coverage_check"),
                "joint_verdict_issues": verdict.get("joint_verdict_issues") or [],
                "halting_reason": result.get("halting_reason"),
                "model_turns": result.get("model_turn_count"),
                "expert_calls": result.get("expert_call_count"),
                "elapsed_s": round(elapsed, 2),
                "evidence": [{
                    "evidence_id": token.get("evidence_id"),
                    "source": token.get("source"),
                    "strength": token.get("strength"),
                    "raw_metric": token.get("raw_metric"),
                    "model_image_layout": token.get("model_image_layout"),
                } for token in result.get("evidence_chain") or []],
                "trace_path": str(trace_path.relative_to(ROOT)),
                "failed_experts": (trace.get("metadata") or {}).get(
                    "failed_experts") or [],
            }
            if record["failed_experts"]:
                raise RuntimeError(f"expert failure: {record['failed_experts']}")
            if not record["final_verdict"]:
                raise RuntimeError("session ended without a final verdict")
            records.append(record)
            write_json(tag_dir / "records" / f"{index:03d}__{sample['sample_id']}.json",
                       record)
            print(f"[{args.tag} {index}/30] {sample['sample_id']} "
                  f"overall={record['final_verdict']} "
                  f"whole={record['whole_image_generation']} "
                  f"local={record['local_manipulation']} "
                  f"calls={record['expert_calls']} {record['elapsed_s']}s",
                  flush=True)
            del fsm, logger
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            failure = {
                "index": index, "sample_id": sample["sample_id"],
                "tag": args.tag, "elapsed_s": round(elapsed, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-8000:],
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            }
            failures.append(failure)
            write_json(tag_dir / "failures.json", failures)
            print(f"[{args.tag} {index}/30] FAILED: {failure['error']}",
                  flush=True)
            break

    write_json(tag_dir / "records.json", records)
    end = {
        "tag": args.tag,
        "adapter": adapter,
        "completed": len(records),
        "expected": len(samples),
        "failed": len(failures),
        "elapsed_s": round(time.time() - started, 1),
        "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
        "failures": failures,
    }
    write_json(tag_dir / "run_end.json", end)
    print(json.dumps(end, ensure_ascii=False, indent=2))
    return 0 if len(records) == len(samples) and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
