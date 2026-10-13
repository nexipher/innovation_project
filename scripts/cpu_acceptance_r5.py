#!/usr/bin/env python3
"""
Cross-module CPU acceptance before the small GPU batch (§4.17, R5 段).

Five properties the review asked to confirm, each checked end to end —
run → record → conversion → workstation — rather than per module:

  1. v3 semantics agree everywhere: the label is the model's conclusion, the
     confidence is the model's own (0.0 stays 0.0), the posterior is a
     reference, and runtime / record / sample / page say the same thing.
  2. A conclusion that never parsed is **not** an abstention: it carries its
     own status, the abstention counter ignores it, and the gate refuses it.
  3. The single-action protocol is accounted for: a mixed turn executes
     nothing, a bbox-less call measures the whole image, and the identity
     (calls executed == tokens produced) still holds.
  4. A defect anywhere in the supervised dialogue refuses the conversion and
     is recorded — the trajectory is not rewritten to hide it.
  5. The new material is isolated from the existing 145-sample batch: its
     files are byte-identical before and after, and a new batch carries a
     different fingerprint.

The script re-executes itself with FORENSIC_HALTING_POLICY=v3 when the
environment does not say so: the default is still v2, and an acceptance run
that silently used v2 rules would prove nothing.

Usage:
  python scripts/cpu_acceptance_r5.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if os.environ.get("FORENSIC_HALTING_POLICY") != "v3" and \
        not os.environ.get("FORENSIC_ACCEPTANCE_REEXEC"):
    os.environ["FORENSIC_HALTING_POLICY"] = "v3"
    os.environ["FORENSIC_ACCEPTANCE_REEXEC"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from config import HALTING_POLICY, PROJECT_ROOT  # noqa: E402
from state_machine.controller import ForensicStateMachine  # noqa: E402
from utils.batch import batch_fingerprint, batch_paths  # noqa: E402
from utils.final_v2 import conclusion_gate, render_sample  # noqa: E402

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_cpu_acceptance_r5.json")
IMAGE = os.path.join(PROJECT_ROOT, "dataset", "GenImage_Test", "Midjourney",
                     "0_midjourney_169.png")
LEGACY_BATCH = "g7_b1"

CALL = ("<planning>\nSuspected Region: [200, 100, 300, 280]\n</planning>\n"
        "<call_noise>[200, 100, 300, 280]</call_noise>")
BARE_CALL = ("<planning>\nSuspected Region: none\nVisual Anomalies: 未观察到明确异常，需要统计证据\n"
             "Expert Target & Hypothesis: noise\n</planning>\n<call_noise></call_noise>")
GARBAGE = "我认为这张图片看起来很正常，没有发现明显的异常。"
MIXED = CALL + '\n<verdict>{"verdict": "Fake"}</verdict>'


class ScriptedClient:
    """Deterministic client: the acceptance checks the pipeline, not the model."""

    def __init__(self, responses):
        self._responses = list(responses)
        self._index = 0

    def generate(self, image_path, history):          # noqa: D401
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    def reset(self):
        self._index = 0

    @property
    def name(self):
        return "scripted"

    @property
    def mode(self):
        return "scripted"


def _build_experts():
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.jpeg import JPEGExpert
    from experts.noise import NoiseExpert

    return {
        "frequency_expert_v2": FrequencyExpertV2(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }


def run_session(responses: List[str], sft_dir: str, **kwargs) -> dict:
    from utils.logger import SessionLogger

    fsm = ForensicStateMachine(ScriptedClient(responses), _build_experts(),
                               logger=SessionLogger(sft_dir=sft_dir), **kwargs)
    return fsm.run(IMAGE, "Fake")


def _trace(result: dict) -> dict:
    with open(result["sft_data_path"], encoding="utf-8") as handle:
        return json.load(handle)


def _verdict(label: str, confidence, report: str = "基于已获取的测量给出结论。") -> str:
    return (f'<reasoning>噪声残留专家给出测量，综合判断为 {label}。</reasoning>\n'
            f'<verdict>{{"verdict": "{label}", "confidence": {confidence},'
            f' "primary_evidence": [], "report": "{report}"}}</verdict>')


def _trajectory_record(result: dict, trace: dict, **extra) -> dict:
    """The record the generator would write, from the session that just ran."""
    verdict = result["final_verdict"]
    record = {
        "trajectory_id": "acceptance__sample_1",
        "policy": "probe+noise",
        "tools_served": ["noise"],
        "policy_version": trace["metadata"].get("policy_version"),
        "final_verdict": verdict.get("verdict"),
        "verdict_status": verdict.get("verdict_status"),
        "confidence": verdict.get("confidence"),
        "posterior": verdict.get("posterior"),
        "reference": verdict.get("reference"),
        "model_candidate": verdict.get("model_candidate"),
        "evidence_check": verdict.get("evidence_check"),
        "conclusion": verdict.get("conclusion"),
        "report": verdict.get("report"),
        "policy_reasons": verdict.get("policy_reasons"),
        "ground_truth": "Fake",
        "generator": "scripted",
        "source_id": "acceptance/source_1",
        "variant_id": "acceptance_variant_1",
        "split": "train",
        "treatment": "png",
        "container": "png",
        "counters": {"weighted_cost": result["weighted_cost"]},
        "evidence": result["evidence_chain"],
        "trace_path": os.path.relpath(result["sft_data_path"], PROJECT_ROOT)
        .replace(os.sep, "/"),
    }
    record.update(extra)
    return record


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_1_v3_semantics(work: str) -> dict:
    result = run_session([CALL, _verdict("Fake", 0.0)], os.path.join(work, "s1"))
    trace = _trace(result)
    verdict = result["final_verdict"]
    metadata = trace["metadata"]
    record = _trajectory_record(result, trace)
    sample = render_sample(record, "positive", observation_text="观察。",
                           split_entry=None, trace=trace)
    sample_verdict = json.loads(sample["conversations"][-1]["value"]
                                .split("<verdict>")[1].split("</verdict>")[0])

    problems = []
    if verdict["verdict"] != "Fake" or verdict["verdict_status"] != "model_concluded":
        problems.append(f"标签/来源不符：{verdict['verdict']} / {verdict['verdict_status']}")
    if verdict["confidence"] != 0.0:
        problems.append(f"模型自报置信度 0.0 被改成 {verdict['confidence']}")
    if metadata.get("policy_version") != "v3":
        problems.append(f"策略版本 {metadata.get('policy_version')} 不是 v3")
    if not (verdict.get("reference") or {}).get("recommendation"):
        problems.append("参考后验未记录")
    if sample_verdict.get("basis") != "model_judgement":
        problems.append(f"训练样本 basis={sample_verdict.get('basis')}")
    if "posterior" in sample_verdict or "reference_posterior" not in sample_verdict:
        problems.append("参考后验与标签后验未分开")
    if sample["final_verdict"].get("verdict_status") != "model_concluded":
        problems.append("样本 final_verdict 未带 verdict_status")
    if metadata.get("model_abstained"):
        problems.append("合法结案被计成弃权")

    page = _workstation_page(work, [sample], record, trace)
    if "标签来源" not in page or "不是校准概率" not in page:
        problems.append("工作台没有说明标签来源/置信度含义")
    return {"status": "pass" if not problems else "fail", "problems": problems,
            "evidence": {"verdict": verdict["verdict"], "confidence": verdict["confidence"],
                         "verdict_status": verdict["verdict_status"],
                         "basis": sample_verdict.get("basis"),
                         "reference": verdict.get("reference")}}


def check_2_unparsed_is_not_abstention(work: str) -> dict:
    responses = [CALL, GARBAGE, GARBAGE, GARBAGE, GARBAGE, GARBAGE, GARBAGE]
    result = run_session(responses, os.path.join(work, "s2"))
    trace = _trace(result)
    verdict = result["final_verdict"]
    metadata = trace["metadata"]
    record = _trajectory_record(result, trace)
    gate = conclusion_gate(record)

    problems = []
    if verdict["verdict_status"] != "no_parseable_conclusion":
        problems.append(f"状态是 {verdict['verdict_status']}")
    if metadata.get("model_abstained"):
        problems.append("解析失败被计成弃权")
    if gate is None:
        problems.append("门禁没有拒收不可解析的结案")
    return {"status": "pass" if not problems else "fail", "problems": problems,
            "evidence": {"verdict_status": verdict["verdict_status"],
                         "model_abstained": metadata.get("model_abstained"),
                         "gate": gate}}


def check_3_protocol_accounting(work: str) -> dict:
    mixed = run_session([MIXED, _verdict("Uncertain", 0.4)], os.path.join(work, "s3a"))
    bare = run_session([BARE_CALL, _verdict("Uncertain", 0.4)], os.path.join(work, "s3b"))
    bare_metadata = _trace(bare)["metadata"]
    token = (bare["evidence_chain"] or [{}])[0]

    problems = []
    mixed_metadata = _trace(mixed)["metadata"]
    if mixed["expert_call_count"] != 0 or mixed_metadata.get("protocol_violations") != 1:
        problems.append(f"混合轮被部分执行：calls={mixed['expert_call_count']}, "
                        f"violations={mixed_metadata.get('protocol_violations')}")
    if bare["expert_call_count"] != 1 or len(bare["evidence_chain"]) != 1:
        problems.append("无 bbox 调用没有按一次测量记账")
    if token.get("region_normalized_1000") is not None:
        problems.append("无 bbox 调用被塞了一个区域")
    if token.get("measurement_scope") != "global":
        problems.append(f"测量范围 {token.get('measurement_scope')}")
    # the identity the acceptance script relies on: executed calls == tokens
    if bare_metadata.get("expert_call_count") != len(bare["evidence_chain"]):
        problems.append("调用-证据恒等式不成立")
    return {"status": "pass" if not problems else "fail", "problems": problems,
            "evidence": {"mixed_calls": mixed["expert_call_count"],
                         "protocol_violations": mixed_metadata.get("protocol_violations"),
                         "bare_call_tokens": len(bare["evidence_chain"]),
                         "region_normalized_1000": token.get("region_normalized_1000")}}


def check_4_dialogue_defects_are_not_hidden(work: str) -> dict:
    fabricated = ('<reasoning>JPEG 压缩专家显示 blockiness 为 2.5，支持伪造。</reasoning>\n'
                  '<verdict>{"verdict": "Fake", "confidence": 0.9, "report": "JPEG 支持伪造。"}'
                  '</verdict>')
    result = run_session([CALL, fabricated, _verdict("Uncertain", 0.4)],
                         os.path.join(work, "s4"))
    trace = _trace(result)
    record = _trajectory_record(result, trace)

    out = os.path.join(work, "s4out")
    trajectories = os.path.join(work, "s4traj")
    os.makedirs(trajectories, exist_ok=True)
    record_path = os.path.join(trajectories, "t.json")
    with open(record_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
    before = _digest(record_path)
    admission = os.path.join(work, "s4admission.json")
    with open(admission, "w", encoding="utf-8") as handle:
        json.dump({"decisions": [{"trajectory_id": record["trajectory_id"],
                                  "category": "positive"}]}, handle)

    from scripts.build_final_v2 import build
    variants = os.path.join(work, "none.json")
    with open(variants, "w", encoding="utf-8") as handle:
        handle.write("{}")
    metadata = build(trajectories, admission, out, "g4_trial", variants_manifest=variants)
    gated = json.load(open(os.path.join(out, "sft_gated.json"), encoding="utf-8"))
    after = _digest(record_path)

    problems = []
    if metadata["total"] != 0:
        problems.append("含中间轮虚构的轨迹仍被渲染成样本")
    reasons = {entry["reason"] for entry in gated}
    if "supervised_turn_fabrication" not in reasons:
        problems.append(f"拒收原因未记录整对话缺陷：{reasons}")
    if before != after:
        problems.append("原始轨迹被改写")
    return {"status": "pass" if not problems else "fail", "problems": problems,
            "evidence": {"gated": metadata["gated"], "reasons": sorted(reasons),
                         "trajectory_unchanged": before == after}}


def check_5_batch_isolation(work: str) -> dict:
    """The new material must not touch, or be confused with, the 145-sample batch."""
    paths = batch_paths(LEGACY_BATCH)
    batch_dir = paths["samples"]          # sft_data/train/batches/g7_b1
    before = _tree_digest(batch_dir)

    result = run_session([CALL, _verdict("Fake", 0.7)], os.path.join(work, "s5"))
    trace = _trace(result)
    record = _trajectory_record(result, trace)
    record["trajectory_id"] = "acceptance__isolation_probe__png"
    sample = render_sample(record, "positive",
                           observation_text="模型视觉观察：边缘纹理不一致。",
                           split_entry=None, trace=trace)

    after = _tree_digest(batch_dir)
    new_fingerprint = batch_fingerprint([sample])
    manifest = json.load(open(os.path.join(batch_dir, "manifest.json"), encoding="utf-8"))
    legacy = manifest["fingerprint"]

    problems = []
    if before != after:
        problems.append("验收过程改动了既有批次文件")
    if new_fingerprint == legacy:
        problems.append("新样本与旧批次指纹相同")
    if manifest["batch_id"] != LEGACY_BATCH or manifest["samples"] != 145:
        problems.append("旧批次清单本身发生了变化")
    return {"status": "pass" if not problems else "fail", "problems": problems,
            "evidence": {"legacy_fingerprint": legacy,
                         "legacy_samples": manifest["samples"],
                         "new_fingerprint": new_fingerprint,
                         "legacy_tree_unchanged": before == after}}


# ---------------------------------------------------------------------------
# Helpers and report
# ---------------------------------------------------------------------------

def _digest(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()[:16]


def _tree_digest(directory: str) -> str:
    digest = hashlib.sha256()
    for root, _dirs, files in sorted(os.walk(directory)):
        for name in sorted(files):
            path = os.path.join(root, name)
            digest.update(os.path.relpath(path, directory).encode("utf-8"))
            with open(path, "rb") as handle:
                digest.update(handle.read())
    return digest.hexdigest()[:16]


def _workstation_page(work: str, samples: List[dict], record: dict,
                      trace: dict) -> str:
    from scripts.build_review_workstation import build as build_workstation

    samples_dir = os.path.join(work, "ws_samples")
    os.makedirs(samples_dir, exist_ok=True)
    with open(os.path.join(samples_dir, "sft_tool_positive.json"), "w",
              encoding="utf-8") as handle:
        json.dump(samples, handle, ensure_ascii=False)
    trajectories = os.path.join(work, "ws_traj")
    os.makedirs(trajectories, exist_ok=True)
    trace_path = os.path.join(work, "ws_trace.json")
    with open(trace_path, "w", encoding="utf-8") as handle:
        json.dump(trace, handle, ensure_ascii=False)
    record = {**record, "trace_path": trace_path}
    with open(os.path.join(trajectories, "r.json"), "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False)
    variants = os.path.join(work, "none.json")
    if not os.path.exists(variants):
        with open(variants, "w", encoding="utf-8") as handle:
            handle.write("{}")
    out = os.path.join(work, "ws_out")
    build_workstation(samples_dir, out, trajectories_dir=trajectories,
                      variants_manifest=variants, batch_id="acceptance")
    sample_id = samples[0]["id"]
    with open(os.path.join(out, "samples", f"{sample_id}.html"), encoding="utf-8") as handle:
        return handle.read()


def main() -> None:
    from utils.config_fingerprint import calibration_identity, prompt_digest

    work = tempfile.mkdtemp(prefix="g7_cpu_acceptance_")
    checks = []
    try:
        for check in (check_1_v3_semantics, check_2_unparsed_is_not_abstention,
                      check_3_protocol_accounting, check_4_dialogue_defects_are_not_hidden,
                      check_5_batch_isolation):
            outcome = check(work)
            outcome["check"] = check.__name__
            checks.append(outcome)
            print(f"{'PASS' if outcome['status'] == 'pass' else 'FAIL'}  {check.__name__}"
                  + ("" if outcome["status"] == "pass"
                     else "  → " + "；".join(outcome["problems"])))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "versions": {
            "halting_policy": HALTING_POLICY,
            "prompt_digest": prompt_digest(),
            "calibration": calibration_identity(),
        },
        "checks": checks,
        "status": "pass" if all(c["status"] == "pass" for c in checks) else "fail",
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n策略 {report['versions']['halting_policy']}｜提示词 {report['versions']['prompt_digest']}"
          f"｜校准表 {report['versions']['calibration']['digest']}"
          f"（schema {report['versions']['calibration']['schema']}，"
          f"拟合组 {report['versions']['calibration']['fitting_group']}）")
    print(f"总状态：{report['status']} -> {os.path.relpath(OUTPUT_PATH, PROJECT_ROOT)}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
