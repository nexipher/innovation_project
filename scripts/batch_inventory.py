#!/usr/bin/env python3
"""
Where every batch stands, and what it was produced under (§4.17 工程收口).

Three kinds of material must never be confused:

  frozen historical    every batch produced so far: the experts are about to
                       change, so these record a measurement system that will
                       no longer exist — replay material, never a training set
  new-version material anything produced after the expert swap (none yet)

The report is generated from the files themselves (records, manifests,
VERSIONS.json, review dispositions), never from memory, and it says
"未记录" where a version genuinely was not recorded rather than guessing.
It also states the two rules that keep getting lost:

  * re-rendering the review material does not regenerate trajectories;
  * passing an automated check does not admit a sample to training.

Usage:
  python scripts/batch_inventory.py            # writes sft_data/BATCH_INVENTORY.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from utils.batch import LEGACY_BATCH, batch_paths, load_trajectory_records

OUTPUT = os.path.join(PROJECT_ROOT, "sft_data", "BATCH_INVENTORY.md")


def _load(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _batch_paths(batch_id: str) -> Dict[str, object]:
    """
    A batch's paths, following `archive/PATH_MAP.json` when it has been archived.

    Stage 1 (2026-10-15) moved the v1 batches; the historical ids still live in
    `FROZEN.json`, so the inventory must keep reporting them — and saying that
    what it read is the archived copy, rather than degrading to "未记录".
    """
    from utils.batch import resolve_path

    resolved: Dict[str, object] = {}
    archived = False
    for key, value in batch_paths(batch_id).items():
        path, from_archive = resolve_path(value)
        resolved[key] = path
        archived = archived or from_archive
    resolved["archived"] = archived
    return resolved


def _versions(batch_id: str, paths: Optional[dict] = None) -> Dict[str, str]:
    """The four versions, from VERSIONS.json, else the run report, else 未记录."""
    paths = paths or _batch_paths(batch_id)
    found: Dict[str, str] = {}
    versions = _load(os.path.join(paths["trajectories"], "VERSIONS.json"))
    if versions:
        found["来源"] = "VERSIONS.json"
        found["策略"] = versions.get("halting_policy", "未记录")
        found["提示词"] = versions.get("prompt_digest", "未记录")
        calibration = versions.get("calibration") or {}
        found["校准表"] = calibration.get("digest") or "未记录"
        found["拟合组"] = calibration.get("fitting_group") or "未记录"
        fingerprint = versions.get("fingerprint") or {}
        found["代码版本"] = fingerprint.get("git_commit") or "未记录"
        return found

    # Older batches: the per-run report still carries the fingerprint.
    directory = paths["trajectories"]
    reports = sorted(name for name in os.listdir(directory)
                     if name.startswith("report_") and name.endswith(".json")) \
        if os.path.isdir(directory) else []
    for name in reports[-1:]:
        report = _load(os.path.join(directory, name)) or {}
        fingerprint = report.get("config_fingerprint") or {}
        if fingerprint:
            found["来源"] = f"{name}（该批次早于 VERSIONS.json）"
            found["策略"] = fingerprint.get("halting_policy", "未记录")
            found["提示词"] = fingerprint.get("prompts", "未记录")
            found["校准表"] = fingerprint.get("reliability_table", "未记录")
            found["拟合组"] = "未记录（当时无此字段）"
            found["代码版本"] = fingerprint.get("git_commit") or "未记录"
            return found
    return {"来源": "未记录", "策略": "未记录", "提示词": "未记录",
            "校准表": "未记录", "拟合组": "未记录", "代码版本": "未记录"}


def _batch_sample_ids(batch_id: str, paths: Optional[dict] = None) -> set:
    """The sample ids this batch's files actually contain (either direction)."""
    paths = paths or _batch_paths(batch_id)
    ids = set()
    for directory in (paths["samples"], paths.get("reviewed", "")):
        if not directory or not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            if not name.endswith(".json") or name.startswith(("manifest", "review_only",
                                                               "VERSIONS", "metadata",
                                                               "selection", "sft_gated",
                                                               "sft_invalid")):
                continue
            data = _load(os.path.join(directory, name))
            if isinstance(data, list):
                ids |= {entry.get("id") for entry in data if isinstance(entry, dict)}
    return {sample_id for sample_id in ids if sample_id}


def _review_state(batch_id: str, paths: Optional[dict] = None) -> Dict[str, object]:
    """
    What the reviewer has decided about *this batch*.

    The dispositions file is global; counting its lines would report the 237
    decisions of the first review against every batch that came later.
    """
    dispositions = os.path.join(PROJECT_ROOT, "sft_data", "review", "dispositions.jsonl")
    own_ids = _batch_sample_ids(batch_id, paths)
    lines = 0
    decided = set()
    if os.path.exists(dispositions):
        with open(dispositions, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                lines += 1
                sample_id = entry.get("sample_id")
                if sample_id in own_ids:
                    decided.add(sample_id)
    # The export can live in the batch's own review directory or at the
    # review root; looking in one place reported "no export file" for a batch
    # whose 145 decisions were sitting in the other (same false zero as the
    # counter that started this).  Both are checked, and the file's own
    # batch_id/fingerprint are read when present.
    #
    # Each candidate is also followed into the archive (2026-10-15): after the
    # move this reported "无导出文件" for g7_b1 again, whose 145 decisions are
    # exactly where the mapping says they are.
    from utils.batch import resolve_path

    decisions = None
    export_path = None
    exported_from_archive = False
    export_is_own = False
    root_export = None
    for candidate in (
            os.path.join(PROJECT_ROOT, "sft_data", "review",
                         f"review_decisions_{batch_id}.json"),
            os.path.join(PROJECT_ROOT, "sft_data", "review", batch_id,
                         f"review_decisions_{batch_id}.json"),
            os.path.join(PROJECT_ROOT, "sft_data", "review", "review_decisions.json")):
        candidate, from_archive = resolve_path(candidate)
        loaded = _load(candidate)
        if from_archive:
            exported_from_archive = True
        if loaded and loaded.get("decisions"):
            own_name = candidate.endswith(f"review_decisions_{batch_id}.json")
            own_id = loaded.get("batch_id") == batch_id
            if own_name or loaded.get("batch_id") in (None, batch_id):
                decisions, export_path, export_is_own = loaded, candidate, own_name or own_id
                if own_id:
                    break
            elif root_export is None and loaded.get("batch_id"):
                root_export = candidate
    exported = {key for key in ((decisions or {}).get("decisions") or {})
                if key in own_ids}
    # A root-level export belonging to an earlier batch is not this batch's
    # file: saying "0 条（<别人的文件>）" invites the reader to think this batch
    # has an export that is merely empty.
    if decisions is not None and not export_is_own and not exported:
        root_export, decisions, export_path = export_path, None, None
    return {"dispositions_lines": lines,
            "own_samples": len(own_ids),
            "decided_in_batch": len(decided),
            "browser_export": decisions is not None,
            "export_from_archive": exported_from_archive if decisions is not None else False,
            "export_path": (os.path.relpath(export_path, PROJECT_ROOT).replace(os.sep, "/")
                            if export_path else None),
            "export_batch": (decisions or {}).get("batch_id"),
            "other_batch_export": (os.path.relpath(root_export, PROJECT_ROOT).replace(os.sep, "/")
                                   if root_export else None),
            "export_decisions_in_batch": len(exported)}


def _samples(batch_id: str, paths: Optional[dict] = None) -> Dict[str, object]:
    paths = paths or _batch_paths(batch_id)
    manifest = _load(os.path.join(paths["samples"], "manifest.json"))
    review_only = _load(os.path.join(paths["samples"], "review_only.json"))
    counts: Dict[str, int] = {}
    for name in ("sft_tool_positive.json", "sft_no_tool_positive.json",
                 "sft_honest_abstention.json"):
        data = _load(os.path.join(paths["samples"], name))
        if isinstance(data, list) and data:
            counts[name] = len(data)
    return {
        "manifest_fingerprint": (manifest or {}).get("fingerprint"),
        "manifest_samples": (manifest or {}).get("samples"),
        "pending_review_samples": sum(counts.values()) or None,
        "admission_applied": review_only is None,
    }


def collect(batch_ids: List[str]) -> List[dict]:
    rows = []
    for batch_id in batch_ids:
        paths = _batch_paths(batch_id)
        records = load_trajectory_records(paths["trajectories"]) if os.path.isdir(
            paths["trajectories"]) else {}
        policies = sorted({record.get("policy_version") for record in records.values()})
        review = _review_state(batch_id, paths)
        samples = _samples(batch_id, paths)
        # kind: reviewed (historical) / produced-but-unreviewed / qualified
        # "Reviewed" is what the reviewer actually decided about this batch's
        # samples, wherever that lives: the browser export counts, and a
        # write-back is reported separately rather than assumed.
        decided = max(review["decided_in_batch"], review["export_decisions_in_batch"])
        total = review["own_samples"]
        # 2026-10-13 决定：过去的所有批次全部**冻结为历史实验**（换专家后生成新版），
        # 因此不再区分“够不够格训练”——任何一批都不是训练材料。
        if batch_id == LEGACY_BATCH:
            kind = "冻结 · 历史实验（v2 时代，未受 R3–R9 保护）"
        elif decided and decided >= total:
            kind = "冻结 · 历史实验（已审核，仍不用于训练）"
        elif decided:
            kind = f"冻结 · 历史实验（部分审核 {decided}/{total}）"
        else:
            kind = "冻结 · 历史实验（未审核；新材料由新版专家生成）"
        rows.append({
            "batch": batch_id,
            "kind": kind,
            "archived": bool(paths["archived"]),
            "trajectories": len(records),
            "policies": policies or ["未记录"],
            "versions": _versions(batch_id, paths),
            "samples": samples,
            "review": review,
        })
    return rows


def render(rows: List[dict]) -> str:
    lines = [
        "# 批次清单（自动生成，勿手改）",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}｜"
        f"命令：`python scripts/batch_inventory.py`",
        "",
        "**三类材料不得混用**：历史批次 / 待审核批次 / 合格训练数据。"
        "本表由文件本身生成（轨迹、清单、`VERSIONS.json`、审核记录），"
        "缺失的版本一律写“未记录”，不猜。",
        "",
        "**两条一直容易被忘掉的规则**：",
        "",
        "1. **重新渲染工作台 ≠ 重新生成轨迹**——评审样本可以重渲染（例如修好转换器后），"
        "轨迹内容不变；`review_only.json` 标记的批次是评审材料，未过准入。",
        "2. **自动检查通过 ≠ 允许训练**——账本、门禁、筛查器的“通过”只说明没有命中已定义的缺陷模式；"
        "进入 SFT 需要：人工审核写回处置 + `apply_review` 出集 + `build_final_v2` 门禁全过。",
        "",
    ]
    for row in rows:
        versions = row["versions"]
        samples = row["samples"]
        review = row["review"]
        lines += [
            f"## {row['batch']}",
            "",
            f"- 类别：**{row['kind']}**",
            (f"- 位置：**已归档** → 以下数字读自 `archive/legacy_v1_20261015/` 的副本"
             "（`archive/PATH_MAP.json`）" if row.get("archived")
             else "- 位置：仓库原位"),
            f"- 轨迹：{row['trajectories']} 条｜策略版本：{', '.join(row['policies'])}",
            f"- 版本（来自 {versions['来源']}）：策略 {versions['策略']}｜"
            f"提示词 {versions['提示词']}｜校准表 {versions['校准表']}"
            f"（拟合组 {versions['拟合组']}）｜代码 {versions['代码版本']}",
            f"- 样本：清单指纹 {samples['manifest_fingerprint'] or '未记录'}｜"
            f"清单样本 {samples['manifest_samples'] or '未记录'}｜"
            f"评审样本 {samples['pending_review_samples'] or '未记录'}｜"
            f"准入已应用：{'是' if samples['admission_applied'] else '否（评审材料）'}",
            f"- 审核：本批样本 {review['own_samples']} 条｜浏览器导出 "
            f"{review['export_decisions_in_batch']} 条"
            + (f"（{review['export_path']}"
               + ("，读自归档副本" if review.get("export_from_archive") else "")
               + "）" if review["export_path"] else "（无导出文件"
               + (f"；仓库根另有 {review['other_batch_export']}，属于更早的批次"
                  if review.get("other_batch_export") else "")
               + "）")
            + f"｜写回仓库的处置 {review['decided_in_batch']} 条"
            + ("（该导出尚未 `record_review --import`）"
               if review["export_decisions_in_batch"] and not review["decided_in_batch"]
               else "")
            + f"｜全局 dispositions {review['dispositions_lines']} 行（属于更早的 237 集）",
            "",
        ]
    lines += [
        "---",
        "",
        "## 路径约定",
        "",
        "| 路径 | 内容 | 能否训练 |",
        "|------|------|----------|",
        "| `archive/legacy_v1_20261015/` | **2026-10-15 归档的 v1 全部产物** | 历史，不训练 |",
        "| `sft_data/g8/` | 新代次（Qwen3-VL + 新专家组），批次名 `b1`… | 建成后按流程 |",
        "| `sft_data/train/final/` | 最早的冻结基线（v1 时代，已归档） | 只作回归对照 |",
        "| `sft_data/train/final_v2/` | G4 转换产物（v2 时代，含 237 集） | 只作对照 |",
        "| `sft_data/train/batches/<id>/` | 分批产物；`review_only.json` 表示评审材料 | 仅当该批有审核处置且过门禁 |",
        "| `sft_data/trajectories_<id>/` | 批次轨迹（只读，含 `VERSIONS.json`） | 不直接训练 |",
        "| `sft_data/trajectories_<id>_<原因>/` | **留证归档**：因缺陷作废的批次 | 不得用于任何统计或训练 |",
        "| `sft_data/review/<id>/` | 工作台页面（可重渲染） | — |",
        "",
        "进入 SFT 的唯一路径：`record_review --import` → `apply_review` → `build_final_v2`（门禁）。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="*", default=None)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--write-missing-versions", action="store_true",
                        help="为缺 VERSIONS.json 的批次补写一份，并注明是从运行报告回填的")
    args = parser.parse_args()

    batches = args.batches or ["g7_b1", "g7_b2"]
    if args.write_missing_versions:
        for batch_id in batches:
            paths = batch_paths(batch_id)
            target = os.path.join(paths["trajectories"], "VERSIONS.json")
            if os.path.exists(target):
                continue
            recovered = _versions(batch_id)
            if recovered.get("来源", "").startswith("未记录"):
                print(f"{batch_id}: 无可回填的版本信息，跳过")
                continue
            payload = {
                "recorded_at": datetime.now().isoformat(timespec="seconds"),
                "source": f"回填：{recovered['来源']}（原批次未写 VERSIONS.json）",
                "halting_policy": recovered["策略"],
                "prompt_digest": recovered["提示词"],
                "calibration": {"digest": recovered["校准表"],
                                "fitting_group": recovered["拟合组"]},
                "fingerprint": {"git_commit": recovered["代码版本"]},
                "note": "回填值取自该批次自己的运行报告，不是当前环境的值",
            }
            with open(target, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            print(f"{batch_id}: 已回填 {os.path.relpath(target, PROJECT_ROOT)}")
    rows = collect(batches)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(render(rows))
    for row in rows:
        print(f"{row['batch']:8s} {row['kind']:30s} 轨迹 {row['trajectories']:3d} "
              f"策略 {', '.join(row['policies'])} "
              f"审核 {max(row['review']['decided_in_batch'], row['review']['export_decisions_in_batch'])}"
              f"/{row['review']['own_samples']}（写回 {row['review']['decided_in_batch']}）")
    print(f"Report: {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
