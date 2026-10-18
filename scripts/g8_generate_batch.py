"""
g8 的批次生成入口（阶段 4.5）。

为什么不是 `generate_trajectories_g4.py`：那是 **G4 多策略矩阵**的入口——策略表是
旧工具箱（noise/jpeg/freq/probe）、客户端是 Qwen2.5、还要过 applicability 门禁。
拿它跑 g8，等于"验收看的管线"和"生成数据的管线"不是同一条。所以小批用自己的入口，
而且**与验收入口同源**：会话由 `main.build_pipeline("qwen")` 构造（CUDA 门禁、专家
不可用即拒、设备自动解析并打印），不是另写一份等价物。

用法：
  # 0) 先建变体（8 源 × 4 格式，类平衡，固定 seed；CPU）
  python scripts/build_variants_g4.py --generation g8 --batch b1 --sources 8

  # 1) CPU 干跑（mock 客户端与专家，验证管道与落盘）
  python scripts/g8_generate_batch.py --generation g8 --batch b1 --dry-run --limit 2

  # 2) 真实会话（需授权；与验收入口同一条管线）
  FORENSIC_ALLOW_GPU=1 PROBE_WEIGHTS=<probe.pth> \\
    python scripts/g8_generate_batch.py --generation g8 --batch b1 --local-files-only --greedy

产物：
  * 记录（每条会话一个）：`sft_data/<gen>/trajectories/<batch>/<variant_id>.json`
  * 会话 trace：`SessionLogger` 默认目录（`traces/sft_sessions/`），记录里写 `trace_path`
  * 批次身份：`sft_data/<gen>/trajectories/<batch>/metadata.json`
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import HALTING_POLICY, PROJECT_ROOT  # noqa: E402
from utils.batch import assert_not_frozen_batch, batch_paths  # noqa: E402

# 与验收一致的默认：贪心、非思考、固定 revision（客户端自己保证）
DEFAULT_MODE = "non-thinking"
# 记录 schema 标着这一批是 g8 主线口径（转换门禁按它决定要不要证据检查字段）
RECORD_SCHEMA = "g8_r1"


def load_manifest(path: str) -> Dict[str, dict]:
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    # 清单键是变体的标识（`build_variants_g4.py` 用相对路径做键）；`path` 才是
    # 要打开的文件，缺省时退回键。
    return {path_key: {**entry, "path": entry.get("path") or path_key}
            for path_key, entry in report["variants"].items()}


def _record(variant: dict, policy_name: str, tools_served: List[str],
            result: dict, trace_path: str, elapsed: float,
            policy_version: str, schema: str) -> dict:
    """
    一条会话 → 一条记录。

    字段照 g7 的记录契约（转换器与审核都按它读），不另立一套：少一个字段，
    下游就会静默少一段（`evidence_check` 缺失会让转换门禁把整条判为旧 schema）。
    """
    verdict = result["final_verdict"]
    metadata = (result.get("trace_metadata") or {})
    return {
        "record_schema": schema,
        "trajectory_id": f"{policy_name}__{variant['variant_id']}",
        "policy": policy_name,
        "policy_version": policy_version,
        "tools_served": list(tools_served),
        "variant_id": variant["variant_id"],
        "source_id": variant["source_id"],
        "split": variant["split"],
        "generator": variant["generator"],
        "treatment": variant["treatment"],
        "container": variant["container"],
        "ground_truth": variant["label"],
        "model_candidate": verdict.get("model_candidate"),
        "final_verdict": verdict.get("verdict"),
        "verdict_status": verdict.get("verdict_status"),
        "confidence": verdict.get("confidence"),
        "report": verdict.get("report"),
        "policy_reasons": verdict.get("policy_reasons"),
        "reference": verdict.get("reference"),
        "posterior": verdict.get("posterior"),
        "conflict_score": verdict.get("conflict_score"),
        "halting_reason": result.get("halting_reason"),
        "conclusion": verdict.get("conclusion"),
        "evidence_check": verdict.get("evidence_check"),
        "evidence": result.get("evidence_chain") or [],
        # 计数器在 trace 的 metadata 里是**平铺**的几个字段（没有嵌套的 counters），
        # 记录里按旧契约合成一份：审核与统计都按这几个名字读。
        "counters": metadata.get("counters") or {
            "model_turns": metadata.get("model_turn_count"),
            "expert_calls": metadata.get("expert_call_count"),
            "unique_evidence": metadata.get("unique_evidence_count"),
            "suppressed_duplicates": metadata.get("suppressed_duplicate_count"),
            "weighted_cost": metadata.get("weighted_cost"),
        },
        "scores": result.get("scores") or {},
        "elapsed_s": round(elapsed, 1),
        "image_path": os.path.relpath(result["image_path"], PROJECT_ROOT)
        if result.get("image_path") else None,
        "trace_path": os.path.relpath(trace_path, PROJECT_ROOT),
    }


def _file_sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_variants_identical(generation: str, batch: str, source_batch: str,
                              manifest_path: str) -> None:
    """
    开跑前逐文件核对：本批的**输入文件**必须与对照批逐字节相同。

    "同一个 seed 重建一遍"只是在假定编码器确定性；名字相同、来源相同、字节不同，
    仍然会让对照不成立，而记录里看不出来。所以这里直接比真实文件的 sha256。
    """
    current = load_manifest(manifest_path)
    source_path = os.path.join(batch_paths(source_batch, generation)["variants"])
    if not os.path.exists(source_path):
        raise SystemExit(f"对照批 {source_batch} 的变体清单不存在：{source_path}")
    source = {entry["variant_id"]: entry for entry in load_manifest(source_path).values()}
    problems = []
    for entry in current.values():
        variant_id = entry["variant_id"]
        other = source.get(variant_id)
        if other is None:
            problems.append(f"{variant_id}: 对照批里没有这个变体")
            continue
        ours = os.path.join(PROJECT_ROOT, entry["path"])
        theirs = os.path.join(PROJECT_ROOT, other["path"])
        if not (os.path.exists(ours) and os.path.exists(theirs)):
            problems.append(f"{variant_id}: 文件缺失")
            continue
        if _file_sha256(ours) != _file_sha256(theirs):
            problems.append(f"{variant_id}: 字节不同")
    extra = set(source) - {entry["variant_id"] for entry in current.values()}
    problems.extend(f"{name}: 对照批有、本批没有" for name in sorted(extra))
    if problems:
        raise SystemExit("与对照批逐文件核对失败（拒绝启动）：\n  " + "\n  ".join(problems))
    print(f"逐文件核对通过：{len(current)} 个输入文件与 {source_batch} 的 sha256 全部相同")


def run_batch(generation: str, batch: str, manifest_path: str,
              tools_policy: str, limit: Optional[int], dry_run: bool,
              allow_frozen: bool, greedy: bool, inference_mode: str,
              local_files_only: bool, out_dir: Optional[str] = None,
              assert_identical_to: Optional[str] = None) -> dict:
    from state_machine.controller import ForensicStateMachine
    from utils.logger import SessionLogger

    assert_not_frozen_batch(batch, "生成新轨迹", generation, allow=allow_frozen)
    paths = batch_paths(batch, generation)
    if out_dir is None:
        if dry_run:
            # 干跑**不写批次目录**：mock 会话与真实会话在记录里长得一样，混进去就是
            # "假数据当真的收"（转换器不会、也无法分辨）。记录**与 trace 都**放到
            # dry_run 目录——只隔离记录、把 mock 的 trace 留在批次目录里，
            # 就是让审阅者在批次材料里翻到一份假会话（实测发生过：6 个干跑 trace）。
            out_dir = os.path.join(PROJECT_ROOT, "traces", "dry_run_records",
                                   generation, batch)
        else:
            out_dir = paths["trajectories"]
    # 真实批次的 trace 落在 `traces/sft_sessions/<代次>/<批次>/`（与旧约定同一棵树）；
    # 干跑与显式 out_dir（测试）都跟着记录走，两样东西永远同进同出。
    default_traces = (not dry_run and out_dir == paths["trajectories"])
    trace_dir = (os.path.join(PROJECT_ROOT, "traces", "sft_sessions", generation, batch)
                 if default_traces else os.path.join(out_dir, "sessions"))
    os.makedirs(out_dir, exist_ok=True)

    if assert_identical_to:
        assert_variants_identical(generation, batch, assert_identical_to, manifest_path)

    variants = load_manifest(manifest_path)
    if limit:
        variants = dict(list(variants.items())[:limit])

    # 客户端与专家由与验收**同一条入口**构造（CUDA 门禁、专家不可用即拒、设备
    # 自动解析都在里面）。客户端只建一次（一次加载 17.5 GB），**每条会话新建
    # 状态机**：一条会话的证据链、计数器与会话文本不该漏到下一条。
    import main as entry

    mllm_type = "mock" if dry_run else "qwen"
    base = entry.build_pipeline(mllm_type, greedy=greedy,
                                inference_mode=inference_mode,
                                local_files_only=local_files_only)
    client, experts = base._mllm, base._experts
    tools_served = sorted(experts)

    records, skipped = [], 0
    started = time.time()
    for key, variant in variants.items():
        record_path = os.path.join(out_dir, f"{variant['variant_id']}__{tools_policy}.json")
        if os.path.exists(record_path):
            skipped += 1
            continue
        image_path = os.path.join(PROJECT_ROOT, variant["path"])
        # 每条会话一个 trace 文件，与记录一一对应（审核页要能点回原始对话）。
        logger = SessionLogger(sft_dir=trace_dir)
        if callable(getattr(client, "reset", None)):
            client.reset()
        fsm = ForensicStateMachine(
            client, experts, logger=logger,
            evidence_injection="text+image",
            # 变体自己知道它属于哪个格式格：校准权重因此取该格的测量。
            condition=variant["treatment"])
        t0 = time.perf_counter()
        result = fsm.run(image_path, ground_truth=variant["label"])
        elapsed = time.perf_counter() - t0
        trace_path = result["sft_data_path"]
        with open(trace_path, encoding="utf-8") as handle:
            trace = json.load(handle)
        result["trace_metadata"] = trace.get("metadata") or {}
        result["image_path"] = image_path
        record = _record(variant, tools_policy, tools_served, result, trace_path,
                         elapsed, HALTING_POLICY, RECORD_SCHEMA)
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
        records.append(record)
        mark = "" if dry_run else f"  VRAM {_vram_gb():.2f} GB"
        print(f"  [{len(records)}] {variant['variant_id']} "
              f"{variant['treatment']:10s} {elapsed:6.1f}s"
              f"  {record['final_verdict']}/{variant['label']}"
              f"  {record['verdict_status']}{mark}", flush=True)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generation": generation,
        "batch": batch,
        "mode": "dry_run" if dry_run else "gpu",
        "policy": tools_policy,
        "policy_version": HALTING_POLICY,
        "tools_served": tools_served,
        "evidence_injection": "text+image",
        "decoding": {"greedy": greedy, "inference_mode": inference_mode},
        "variants_total": len(load_manifest(manifest_path)),
        "variants_this_run": len(variants),
        "sessions_written": len(records),
        "sessions_skipped_existing": skipped,
        "elapsed_s": round(time.time() - started, 1),
        "model": _model_identity(base),
        "experts": _expert_identity(base),
        "prompts": _prompt_identity(),
        "entry": "scripts/g8_generate_batch.py → main.build_pipeline",
        "manifest": os.path.relpath(manifest_path, PROJECT_ROOT),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(f"\n记录 {len(records)} 条（跳过已存在 {skipped} 条）→ {out_dir}")
    return metadata


def _vram_gb() -> float:
    try:
        import torch

        return torch.cuda.max_memory_allocated() / 1024 ** 3
    except Exception:
        return 0.0


def _model_identity(fsm) -> dict:
    client = getattr(fsm, "_mllm", None)
    if client is None or not callable(getattr(client, "identity", None)):
        return {"name": getattr(client, "name", None), "mode": "mock"}
    return client.identity()


def _expert_identity(fsm) -> Dict[str, dict]:
    """
    每个专家：项目已有的身份函数（类/极性/源名）+ 设备与权重出处。

    "同一权重、不同加载器/不同设备"是会让数字悄悄变掉的那类差别，所以设备写进
    批次身份，而不是靠"应该是在 GPU 上跑的"。
    """
    from utils.config_fingerprint import expert_specs

    specs = expert_specs(fsm._experts or {})
    for name, expert in (fsm._experts or {}).items():
        specs.setdefault(name, {})
        specs[name]["device"] = getattr(expert, "device", None)
        for attr in ("weight_source", "licence_note", "score_reference"):
            value = getattr(expert, attr, None)
            if value:
                specs[name][attr] = value
        weights = getattr(expert, "weights_path", None) or getattr(expert, "weight_source", None)
        if isinstance(weights, str) and os.path.exists(weights):
            specs[name]["weights_bytes"] = os.path.getsize(weights)
    return specs


def _prompt_identity() -> dict:
    from utils.config_fingerprint import prompt_digest

    return {"digest": prompt_digest()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--manifest", default=None,
                        help="变体清单（默认 sft_data/<gen>/variants/<batch>/manifest.json）")
    parser.add_argument("--policy", default="g8_default",
                        help="记录里的策略名（g8 只有一条：新工具箱）")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="mock 客户端与 mock 专家（CPU 管道验证，不加载权重）")
    parser.add_argument("--allow-frozen", action="store_true")
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--inference-mode", default=DEFAULT_MODE,
                        choices=("thinking", "non-thinking"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--assert-variants-identical-to", default=None,
                        help="开跑前逐文件核对本批输入与对照批（如 b1）的 sha256 完全相同")
    args = parser.parse_args()

    if not args.dry_run and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        print("真实批次需要授权：确认后加 FORENSIC_ALLOW_GPU=1 重跑"
              "（或先用 --dry-run 在 CPU 上验证管道）", file=sys.stderr)
        return 2
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"

    paths = batch_paths(args.batch, args.generation)
    manifest = args.manifest or paths["variants"]
    if not os.path.exists(manifest):
        print(f"变体清单不存在：{manifest}\n先跑：python scripts/build_variants_g4.py "
              f"--generation {args.generation} --batch {args.batch} --sources 8",
              file=sys.stderr)
        return 2

    metadata = run_batch(args.generation, args.batch, manifest, args.policy,
                         args.limit, args.dry_run, args.allow_frozen, args.greedy,
                         args.inference_mode, args.local_files_only,
                         assert_identical_to=args.assert_variants_identical_to)
    print(f"model: {json.dumps(metadata['model'], ensure_ascii=False)[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
