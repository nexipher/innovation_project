#!/usr/bin/env python3
"""
G8 minimal smoke (plan.md §3b, proposal §3): does the new toolbox actually work?

Five checks, each of which has failed in this project before:

  1. the image really enters the multimodal message — a message that carries a
     *path* is not a message that carries a picture;
  2. one legal tool request is actually executed by the controller;
  3. the result reaches the right later message and is referenced accurately;
  4. the image context survives the tool call and the conclusion parses;
  5. the model's raw reply, the reply the project adopted, and the external
     control reason are kept **separately** — a run that keeps only the adopted
     one cannot answer "what did the model actually say?".

Toolbox: the new generation's, and nothing else — one mock expert on CPU, or a
real PROBE plus one fixed-value mock expert on GPU.  Products go to
`sft_data/g8_smoke/<timestamp>/`, which is not a batch directory: no batch
identity, no freeze gate, nothing a later stage can mistake for training data.

Usage:
  python scripts/g8_smoke.py --client mock                       # CPU, protocol only
  python scripts/g8_smoke.py --client qwen3 --experts probe,mock # GPU (needs authorization)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from experts.external_base import MockExternalScoreExpert
from mllm.message_builder import collect_images
from state_machine.controller import ForensicStateMachine

OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "sft_data", "g8_smoke")
STUB_SCORE = 0.4242      # the stub's fixed score: recognisable, never rounded away
DEFAULT_IMAGE = os.path.join(PROJECT_ROOT, "dataset", "GenImage_Test",
                             "Midjourney", "0_midjourney_169.png")


# ---------------------------------------------------------------------------
# Toolbox: the new generation's, and only that
# ---------------------------------------------------------------------------


class NoiseStubExpert(MockExternalScoreExpert):
    """
    Fixed-value stand-in for the corroborating expert, under the real tag.

    `source_name` is a class attribute on the external-expert base (the token's
    `source` must be the registered name, not a label invented per call), so a
    stand-in has to be a subclass.  It keeps the real expert's source and
    evidence names — downstream lookups (controller dispatch, ledger checks)
    key on those — while returning a score this script chose, which is what
    lets check 3 compare against a known value.
    """

    source_name = "noise_expert"
    evidence_name = "noise_residual_inconsistency"
    score_reference = "SRM 噪声残差一致性（冒烟固定值桩）"
    weights_env = ""
    weight_source = "smoke stub"
    licence_note = "stub（不涉及任何权重）"

def build_toolbox(names: List[str], client_kind: str) -> Dict[str, object]:
    """
    `source_name -> expert`, restricted to what the smoke asked for.

    `probe` resolves to the real PROBE when its weights are present and to the
    project's mock otherwise; `mock` is a fixed-value expert whose token is
    predictable, so check 3 ("the result is referenced accurately") compares
    against a value this script chose rather than one the model invented.
    """
    toolbox: Dict[str, object] = {}
    for name in names:
        if name == "probe":
            from experts.probe_dino import MockProbeExpert, ProbeDinoExpert

            if client_kind == "mock":
                toolbox["probe_dino_expert"] = MockProbeExpert()
                continue
            expert = ProbeDinoExpert()
            reason = expert.unavailable_reason()
            if reason:
                raise SystemExit(f"PROBE 不可用（{reason}）：GPU 冒烟需要它的权重。")
            toolbox["probe_dino_expert"] = expert
        elif name in ("noise-stub", "mock"):
            # Registered under the `noise` tag on purpose: the prompt's tool
            # table only knows probe/freq/noise/jpeg, so a brand-new tag would
            # be a tool the model is never told about.  The score is fixed by
            # this script, which is what makes check 3 a fact rather than an
            # impression.
            toolbox["noise_expert"] = NoiseStubExpert(score=STUB_SCORE)
        else:
            raise SystemExit(f"未知工具：{name}（冒烟只允许 probe / noise-stub）")
    if not toolbox:
        raise SystemExit("空工具箱：至少注册一个专家")
    return toolbox


# ---------------------------------------------------------------------------
# The five checks
# ---------------------------------------------------------------------------

def check_image_enters_multimodal(image: str, client) -> dict:
    """
    The image must be inside the message the builder produces, and — with a
    real model — inside the tensors the model actually received.

    A message carrying a *path* is not a message carrying a picture: the G4-f
    screen once built text-only calls and the model answered from imagination.
    `build_messages` is the component the real client uses, so checking it is
    checking the real path; `last_input_stats` then confirms it survived the
    processor (`image_tokens` counts the placeholder ids: 0 means the picture
    was dropped between the message and the model).
    """
    from mllm.message_builder import build_messages

    history = [{"from": "user", "value": "<image>\n请分析这张图像。"}]
    messages = build_messages(image, history)
    kinds, count = [], 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                count += 1
                kinds.append(type(part.get("image")).__name__)
    builder_ok = count == 1 and kinds == ["Image"]

    stats = getattr(client, "last_input_stats", None)
    model_ok = None if stats is None else (stats.get("image_tokens") or 0) > 0
    ok = builder_ok and (model_ok is not False)   # None = CPU mock, nothing to check
    return {"ok": ok, "builder_image_parts": count, "builder_payload_types": kinds,
            "model_input_stats": stats, "model_received_image_tokens": model_ok,
            "detail": ("消息里带 PIL 图像" if builder_ok else "消息里没有真正的图像对象")
                      + ("；模型输入含图像 token" if model_ok else
                         ("；模型输入**没有**图像 token" if model_ok is False
                          else "（CPU Mock：模型输入无此概念）"))}


def check_one_call_executed(result: dict) -> dict:
    executed = [entry.get("source") for entry in result.get("evidence_chain") or []
                if entry.get("source")]
    counters = result.get("counters") or {}
    return {"ok": len(executed) >= 1 and counters.get("executions", len(executed)) >= 1,
            "executed_sources": executed,
            "counters": counters,
            "detail": f"执行 {len(executed)} 次工具调用"}


def check_result_is_referenced(trace: dict, expected: Optional[float]) -> dict:
    """
    The tool's number must reappear in a later assistant turn.

    The stub's score is chosen by this script (0.4242), so finding it later is
    a fact about the session.  Several renderings are accepted because a model
    may quote it as `0.4242`, `0.42` or `42.4%`; a match on any of them is
    reported with the form that matched, and the *absence* of all three is the
    failure this check exists for.
    """
    turns = [turn.get("value", "") for turn in (trace.get("conversations") or [])
             if turn.get("from") == "gpt" and isinstance(turn.get("value"), str)]
    # the first assistant turn asked for the tool; later ones must use it
    later = "\n".join(turns[1:]) if len(turns) > 1 else ""
    if expected is None:
        return {"ok": False, "needle": None, "matched_form": None,
                "detail": "工具没有给出可比对的数值"}
    forms = [f"{expected}", f"{expected:.4f}", f"{expected:.2f}", f"{expected * 100:.1f}%"]
    matched = next((form for form in forms if form in later), None)
    return {"ok": matched is not None, "needle": expected, "matched_form": matched,
            "checked_forms": forms,
            "detail": (f"后续轮准确引用了工具数值（{matched}）" if matched
                       else f"后续轮找不到工具数值（试过 {forms}）")}


def check_conclusion_parses(image: str, result: dict, client) -> dict:
    """
    A parseable conclusion, reached with the picture still in context.

    "Still in context" is checked where it can be checked: the message builder
    is asked for the messages of the **final** turn's history (the same
    component the real client calls), and with a real model the last call's
    input statistics are read — `image_tokens > 0` on the closing turn means
    the model could still see the image it was judging.  Looking for image
    parts in `result["conversation"]` finds nothing by construction: that is
    the string transcript, not the multimodal messages.
    """
    from mllm.message_builder import build_messages

    verdict = result.get("final_verdict") or {}
    parsed = bool(verdict) and str(verdict.get("verdict", "")).strip() != ""
    history = result.get("conversation") or []
    messages = build_messages(image, history) if history else []
    image_parts = sum(
        1 for message in messages if isinstance(message.get("content"), list)
        for part in message["content"]
        if isinstance(part, dict) and part.get("type") == "image")
    stats = getattr(client, "last_input_stats", None)     # the closing call
    model_ok = None if stats is None else (stats.get("image_tokens") or 0) > 0
    ok = parsed and image_parts >= 1 and model_ok is not False
    return {"ok": ok, "verdict": verdict, "halting_reason": result.get("halting_reason"),
            "builder_image_parts_on_final_turn": image_parts,
            "closing_call_input_stats": stats,
            "detail": ("结案可解析；收尾轮的构造消息仍带图像"
                       + ("，模型输入含图像 token" if model_ok else
                          ("，但模型输入**没有**图像 token" if model_ok is False
                           else "（CPU Mock：模型输入无此概念）"))
                       if parsed else "结案不可解析")}


def check_three_records_kept(trace: dict, result: dict) -> dict:
    """
    Raw reply / adopted reply / control reason, kept apart.

    A run that keeps only the adopted verdict cannot be audited: this project
    has already had sessions whose adopted label disagreed with the model's own
    closing turn, and the difference is only visible if both survive.
    """
    assistant_turns = [turn.get("value") for turn in (trace.get("conversations") or [])
                       if turn.get("from") == "gpt" and isinstance(turn.get("value"), str)]
    raw_last = assistant_turns[-1] if assistant_turns else ""
    adopted = result.get("final_verdict")
    reason = result.get("halting_reason")
    return {"ok": bool(raw_last) and adopted is not None and bool(reason),
            "raw_reply": raw_last,
            "adopted": adopted,
            "control_reason": reason,
            "detail": "三类记录分别留存" if (raw_last and adopted is not None and reason)
                      else "缺少其中之一"}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def build_client(kind: str, system_prompt: str, inference_mode: str, greedy: bool,
                 local_files_only: bool = False):
    if kind == "mock":
        from mllm.mock_client import MockMLLMClient

        return MockMLLMClient(mode="probe_then_corroborate"), None
    if kind == "qwen3":
        from mllm.qwen3_client import Qwen3VLClient

        client = Qwen3VLClient(system_prompt=system_prompt,
                               inference_mode=inference_mode,
                               do_sample=not greedy,
                               temperature=0.0 if greedy else 0.1,
                               local_files_only=local_files_only)
        return client, client
    raise SystemExit(f"未知客户端：{kind}")


def run(image: str, client_kind: str, experts: List[str], inference_mode: str,
        greedy: bool, out_dir: Optional[str] = None,
        local_files_only: bool = False) -> dict:
    toolbox = build_toolbox(experts, client_kind)
    tags = [ForensicStateMachine.EXPERT_SOURCE_MAP[name] for name in toolbox
            if name in ForensicStateMachine.EXPERT_SOURCE_MAP]
    from mllm.message_builder import build_forensic_prompt

    system_prompt = build_forensic_prompt(tags)
    client, identity_client = build_client(client_kind, system_prompt, inference_mode,
                                           greedy, local_files_only)

    fsm = ForensicStateMachine(client, toolbox)
    started = time.perf_counter()
    result = fsm.run(image)
    elapsed = time.perf_counter() - started
    trace = result.get("_trace") or result.get("trace") or {}
    if not trace:
        # the state machine keeps the session record under its own key
        trace = result.get("session") or {"conversations": result.get("conversation") or [],
                                          "messages": result.get("messages") or []}

    expected = None
    for entry in result.get("evidence_chain") or []:
        if entry.get("source") == "noise_expert" and entry.get("raw_value") is not None:
            expected = entry["raw_value"]
            break

    checks = {
        "1_image_enters_the_multimodal_message": check_image_enters_multimodal(image, client),
        "2_one_legal_call_executed": check_one_call_executed(result),
        "3_result_referenced_later": check_result_is_referenced(trace, expected or STUB_SCORE),
        "4_conclusion_parses_with_image_context": check_conclusion_parses(image, result, client),
        "5_three_records_kept_separately": check_three_records_kept(trace, result),
    }
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "client": client_kind,
        "model": (identity_client.identity() if identity_client else {"name": client.name}),
        "experts": sorted(toolbox),
        "image": os.path.relpath(image, PROJECT_ROOT),
        "elapsed_s": round(elapsed, 2),
        "vram_peak_gb": (round(__import__("torch").cuda.max_memory_allocated() / 1e9, 2)
                         if client_kind == "qwen3" else None),
        "checks": checks,
        "passed": all(check["ok"] for check in checks.values()),
    }

    out_dir = out_dir or os.path.join(
        OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "smoke_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "session_trace.json"), "w", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False, indent=1)
    report["report_path"] = os.path.relpath(os.path.join(out_dir, "smoke_report.json"),
                                            PROJECT_ROOT)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=("mock", "qwen3"), default="mock")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--experts", default="probe,noise-stub",
                        help="冒烟工具箱（probe / noise-stub，逗号分隔）")
    parser.add_argument("--inference-mode", choices=("thinking", "non-thinking"),
                        default="non-thinking")
    parser.add_argument("--greedy", action="store_true", help="贪心解码，便于复现")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true",
                        help="只用已下载的权重（revision 已固定；避免加载时再走网络）")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"图像不存在：{args.image}")
    if args.client == "qwen3" and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        # GPU work needs the user's authorization; this is the switch they flip.
        raise SystemExit(
            "GPU 冒烟需要授权：确认后加 FORENSIC_ALLOW_GPU=1 重跑"
            "（或用 --client mock 先做 CPU 协议验证）。")

    report = run(args.image, args.client, [e.strip() for e in args.experts.split(",") if e.strip()],
                 args.inference_mode, args.greedy, args.out_dir,
                 local_files_only=args.local_files_only)
    for name, check in report["checks"].items():
        print(f"  {'✅' if check['ok'] else '❌'} {name}：{check['detail']}")
    print(f"\n通过：{report['passed']}｜耗时 {report['elapsed_s']}s"
          + (f"｜峰值显存 {report['vram_peak_gb']} GB" if report["vram_peak_gb"] else ""))
    print(f"报告：{report['report_path']}")


if __name__ == "__main__":
    main()
