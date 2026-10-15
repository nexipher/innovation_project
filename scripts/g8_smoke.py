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
  python scripts/g8_smoke.py --client qwen3 --experts probe,mock  # GPU (needs authorization)
  FORENSIC_ALLOW_GPU=1 python scripts/g8_smoke.py --client qwen3 \
      --experts probe,npr,univfd --local-files-only --greedy      # 小批前的真实工具箱
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


class NPRStubExpert(MockExternalScoreExpert):
    """
    Fixed-value stand-in for the corroborating expert, under the real tag.

    `source_name` is a class attribute on the external-expert base (the token's
    `source` must be the registered name, not a label invented per call), so a
    stand-in has to be a subclass.  It keeps the real expert's source and
    evidence names — downstream lookups (controller dispatch, ledger checks)
    key on those — while returning a score this script chose, which is what
    lets check 3 compare against a known value.

    阶段 4.4：注册在 `npr_expert` 下——那是 mock 的候补动作现在会调用的标签。
    """
    source_name = "npr_expert"
    evidence_name = "npr_upsampling_artefact_score"
    score_reference = "NPR（CVPR 2024）冒烟固定值桩"
    weights_env = ""
    weight_source = "smoke stub"
    licence_note = "stub（不涉及任何权重）"

def build_toolbox(names: List[str], client_kind: str) -> Dict[str, object]:
    """
    `source_name -> expert`, restricted to what the smoke asked for.

    `probe` / `npr` / `univfd` resolve to the real experts when their weights
    are present; the mock client gets the project's mocks instead.  `npr-stub`
    is a fixed-value expert whose token is predictable, so check 3 ("the result
    is referenced accurately") compares against a value this script chose rather
    than one the model invented.

    阶段 4.5：真实工具箱就是 probe/npr/univfd，而**后两个的真实路径在小批前从没
    在本机上跑过**（阶段 3 评测走的是评测入口）——冒烟要能同时点它们，否则"专家
    能加载、能出分、能被模型引用"这件事只对 PROBE 成立过。
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
        elif name in ("npr", "univfd"):
            if name == "npr":
                from experts.npr import NPRExpert, MockNPRExpert

                expert = MockNPRExpert() if client_kind == "mock" else NPRExpert()
            else:
                from experts.univfd import MockUnivFDExpert, UnivFDExpert

                expert = (MockUnivFDExpert() if client_kind == "mock"
                          else UnivFDExpert())
            if client_kind != "mock":
                reason = expert.unavailable_reason()
                if reason:
                    raise SystemExit(f"{name} 不可用（{reason}）：GPU 冒烟需要它的权重。")
            toolbox[f"{name}_expert"] = expert
        elif name in ("npr-stub", "mock"):
            # 固定值桩：分数由本脚本给定，check 3 才是在比对事实而不是印象。
            toolbox["npr_expert"] = NPRStubExpert(score=STUB_SCORE)
        else:
            raise SystemExit(f"未知工具：{name}（冒烟允许 probe / npr / univfd / npr-stub）")
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
    # The result carries these as top-level keys, not under "counters" (an
    # empty dict there read as "no counters recorded" when they were right here).
    counters = {key: result[key] for key in
                ("expert_call_count", "unique_evidence_count",
                 "skipped_duplicate_calls", "suppressed_duplicate_count",
                 "weighted_cost") if key in result}
    return {"ok": len(executed) >= 1 and counters.get("expert_call_count", len(executed)) >= 1,
            "executed_sources": executed,
            "counters": counters,
            "detail": f"执行 {len(executed)} 次工具调用（记账 {counters.get('expert_call_count')}）"}


def token_needles(result: dict) -> List:
    """
    What a later turn could legitimately quote from an executed tool: its
    evidence id, its evidence name, or one of its numbers.

    Reading only `raw_value` found nothing (the token stores `raw_metric` and
    `strength`), and reading only numbers is too strict — the real model wrote
    "probe_dino_expert 的测量结果明确指向 AI 生成（P(Fake)=1.0）", naming the
    evidence and quoting a rounded value.  Both are accurate references, so
    both are accepted; the report says which one matched.  The fixed stub value
    goes first when it was used, because that match is unambiguous.
    """
    needles: List = []
    for entry in result.get("evidence_chain") or []:
        # 顺序＝证据链顺序（谁先执行谁在前），不做特例偏好：特例会在换工具后过期。
        value = entry.get("raw_metric", entry.get("strength"))
        for key in ("evidence_id", "evidence_name", "source"):
            if entry.get(key):
                needles.append(str(entry[key]))
        for key in ("raw_metric", "strength", "calibrated_likelihood", "probability"):
            value = entry.get(key)
            if isinstance(value, (int, float)):
                needles.append(float(value))
    return needles


def check_result_is_referenced(trace: dict, needles: List) -> dict:
    """
    The tool's result must come back in a later assistant turn.

    The first assistant turn is where the tool was *requested*; the reference
    has to be in a turn after the tool answer arrived.  A reference counts as
    accurate when the later turn names the evidence (id or name) or quotes one
    of its numbers in a recognisable rendering.
    """
    turns = [turn.get("value", "") for turn in (trace.get("conversations") or [])
             if turn.get("from") == "gpt" and isinstance(turn.get("value"), str)]
    later = "\n".join(turns[1:]) if len(turns) > 1 else ""
    if not needles:
        return {"ok": False, "needles": [], "matched": None,
                "detail": "工具没有给出可比对的证据或数值"}

    tried = []
    for needle in needles:
        if isinstance(needle, str):
            tried.append(needle)
            if needle and needle in later:
                return {"ok": True, "needles": needles, "matched": needle,
                        "kind": "evidence_id_or_name", "tried": tried,
                        "detail": f"后续轮引用了证据 {needle}"}
            continue
        for form in (f"{needle}", f"{needle:.4f}", f"{needle:.2f}", f"{needle * 100:.1f}%"):
            tried.append(form)
            if form in later:
                return {"ok": True, "needles": needles, "matched": form,
                        "kind": "number", "tried": tried,
                        "detail": f"后续轮引用了工具数值 {form}"}
    return {"ok": False, "needles": needles, "matched": None, "tried": tried,
            "detail": f"后续轮找不到该工具的证据或数值（试过 {tried}）"}


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


def check_first_turn_description(trace: dict, client_kind: str) -> dict:
    """
    The first turn must carry a *specific* visual account (proposal §8).

    Mechanical part: a `Visual Observation:` line exists, is not empty, and is
    not one of the stock sentences the g7 batch produced ("图像为合成或经过处理的
    自然图像…").  Specificity itself is a human judgement, so the text is quoted
    into the report verbatim — the proposal asks for a small manual read of raw
    input and reply before scaling generation, and this is where that read
    starts.  The observation and the suspected anomalies are reported apart,
    because the protocol asks for exactly that separation.
    """
    if client_kind != "qwen3":
        # The mock's planning line is a protocol template, not a visual account;
        # judging it here would either fail every CPU run or teach the reader to
        # ignore the check.  A check that cannot be evaluated says so.
        return {"ok": None, "applicable": False, "observation": "", "anomalies": "",
                "chars": 0,
                "detail": "不适用：Mock 不产生视觉观察（该项只在真模型上判）"}
    turns = [turn.get("value", "") for turn in (trace.get("conversations") or [])
             if turn.get("from") == "gpt" and isinstance(turn.get("value"), str)]
    first = turns[0] if turns else ""
    observation = ""
    anomalies = ""
    for line in first.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("visual observation:"):
            observation = stripped.split(":", 1)[1].strip()
        elif stripped.lower().startswith("suspected anomalies:"):
            anomalies = stripped.split(":", 1)[1].strip()
    stock = ("图像为合成或经过处理的自然图像", "无法判断", "无异常")
    ok = (len(observation) >= 15 and not any(phrase in observation for phrase in stock))
    return {"ok": ok, "applicable": True,
            "observation": observation, "anomalies": anomalies,
            "chars": len(observation),
            "detail": (f"首轮视觉观察 {len(observation)} 字，另列疑似异常"
                       if ok else
                       ("首轮没有可读的视觉观察" if not observation
                        else f"视觉观察疑似模板套话：{observation[:40]}"))}


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

def check_the_processor_makes_image_tokens(image: str, local_files_only: bool = True) -> dict:
    """
    Verify the image path with the **real** processor, before any GPU time.

    The processor is small (a few MB) and already on disk, and it is the
    component that turns (text, PIL image) into `input_ids`.  If the placeholder
    tokens are not there, no amount of GPU time would fix check 1 — so this runs
    first, on CPU, and reports the count instead of a claim.

    Mirrors where the id comes from: `config.json`'s `image_token_id`, with the
    tokenizer's `<|image_pad|>` id as a cross-check.
    """
    from huggingface_hub import hf_hub_download

    from mllm.message_builder import build_messages
    from mllm.qwen3_client import MODEL_ID, MODEL_REVISION

    processor = _load_processor(MODEL_ID, MODEL_REVISION, local_files_only)
    messages = build_messages(image, [{"from": "user", "value": "<image>\n请分析这张图像。"}])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = collect_images(messages)
    inputs = processor(text=[text], images=images if images else None, return_tensors="pt")
    ids = inputs["input_ids"]

    # One file, not the whole snapshot: `snapshot_download(local_files_only=True)`
    # refuses while the weight shards are still arriving (it checks completeness),
    # and this check exists precisely to run *before* they land.
    config_path = hf_hub_download(MODEL_ID, "config.json", revision=MODEL_REVISION,
                                  local_files_only=local_files_only)
    with open(config_path, encoding="utf-8") as handle:
        image_token_id = json.load(handle).get("image_token_id")
    cross_check = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    count = int((ids == image_token_id).sum()) if image_token_id is not None else None
    return {"ok": bool(images) and (count or 0) > 0,
            "template_has_thinking_switch": "enable_thinking" in (
                getattr(processor, "chat_template", "") or ""),
            "pil_images": len(images),
            "input_ids_shape": list(ids.shape),
            "image_token_id": image_token_id,
            "image_token_id_from_tokenizer": cross_check,
            "image_tokens_in_input": count,
            "pixel_values_shape": (list(inputs["pixel_values"].shape)
                                   if "pixel_values" in inputs else None)}


def _load_processor(model_id: str, revision: str, local_files_only: bool):
    from transformers import Qwen3VLProcessor

    return Qwen3VLProcessor.from_pretrained(model_id, revision=revision,
                                            local_files_only=local_files_only)


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

    # Resolve the destination first: the session logger writes the *complete*
    # record (evidence chain included) and it must land beside the report, not
    # in `traces/sft_sessions/` where smoke sessions would mix with real ones
    # and every test run would add files to the repository.
    out_dir = out_dir or os.path.join(
        OUTPUT_ROOT, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    from utils.logger import SessionLogger

    fsm = ForensicStateMachine(client, toolbox, SessionLogger(sft_dir=out_dir))
    started = time.perf_counter()
    result = fsm.run(image)
    elapsed = time.perf_counter() - started
    trace = result.get("_trace") or result.get("trace") or {}
    if not trace:
        # the state machine keeps the session record under its own key
        trace = result.get("session") or {"conversations": result.get("conversation") or [],
                                          "messages": result.get("messages") or []}

    # the stub's value first (unambiguous), then every other executed tool's
    needles = token_needles(result)
    if STUB_SCORE not in needles:
        needles.append(STUB_SCORE)      # the stub may never have been called

    checks = {
        "1_image_enters_the_multimodal_message": check_image_enters_multimodal(image, client),
        "2_one_legal_call_executed": check_one_call_executed(result),
        "3_result_referenced_later": check_result_is_referenced(trace, needles),
        "4_conclusion_parses_with_image_context": check_conclusion_parses(image, result, client),
        "5_three_records_kept_separately": check_three_records_kept(trace, result),
        "6_first_turn_has_a_specific_visual_account": check_first_turn_description(trace, client_kind),
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
        # A not-applicable check is neither a pass nor a failure: `passed` asks
        # only about the checks this run could actually make.
        "passed": all(check["ok"] for check in checks.values()
                      if check.get("applicable", True)),
        "not_applicable": [name for name, check in checks.items()
                           if not check.get("applicable", True)],
    }

    # The logger already wrote the full session record next to the report.
    report["session_record"] = result.get("sft_data_path")
    with open(os.path.join(out_dir, "smoke_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    report["report_path"] = os.path.relpath(os.path.join(out_dir, "smoke_report.json"),
                                            PROJECT_ROOT)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", choices=("mock", "qwen3"), default="mock")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--experts", default="probe,npr-stub",
                        help=("冒烟工具箱（probe / npr / univfd / npr-stub，逗号分隔）。"
                              "npr-stub 是固定值桩，用于 check 3 的精确比对；"
                              "真实小批前建议用 probe,npr,univfd 点齐三个。")),
    parser.add_argument("--inference-mode", choices=("thinking", "non-thinking"),
                        default="non-thinking")
    parser.add_argument("--greedy", action="store_true", help="贪心解码，便于复现")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--processor-check", action="store_true",
                        help="只用 processor 验证图像->token 路径（CPU，不需权重）")
    parser.add_argument("--local-files-only", action="store_true",
                        help="只用已下载的权重（revision 已固定；避免加载时再走网络）")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise SystemExit(f"图像不存在：{args.image}")
    if args.processor_check:
        result = check_the_processor_makes_image_tokens(args.image, local_files_only=True)
        for key, value in result.items():
            print(f"  {key}: {value}")
        raise SystemExit(0 if result["ok"] else 1)
    if args.client == "qwen3" and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        # GPU work needs the user's authorization; this is the switch they flip.
        raise SystemExit(
            "GPU 冒烟需要授权：确认后加 FORENSIC_ALLOW_GPU=1 重跑"
            "（或用 --client mock 先做 CPU 协议验证）。")

    report = run(args.image, args.client, [e.strip() for e in args.experts.split(",") if e.strip()],
                 args.inference_mode, args.greedy, args.out_dir,
                 local_files_only=args.local_files_only)
    for name, check in report["checks"].items():
        mark = "—" if not check.get("applicable", True) else ("✅" if check["ok"] else "❌")
        print(f"  {mark} {name}：{check['detail']}")
    print(f"\n通过：{report['passed']}｜耗时 {report['elapsed_s']}s"
          + (f"｜峰值显存 {report['vram_peak_gb']} GB" if report["vram_peak_gb"] else ""))
    print(f"报告：{report['report_path']}")


if __name__ == "__main__":
    main()
