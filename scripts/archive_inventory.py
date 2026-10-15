#!/usr/bin/env python3
"""
Pre-flight inventory for the v1 archive (提案 §2.1，只读，不移动任何文件).

The proposal says: list first, verify, then archive.  This is the list.  For
every asset it records where it lives now, how big it is, whether git has it
(so a move is recoverable), a content digest where that is affordable, and
**what role it plays**: runtime dependencies are not history and must not be
moved blindly (proposal §2.6).

It writes two files and touches nothing else:

    ARCHIVE_INVENTORY.md        the readable list, with the caveats stated
    archive_inventory.json      the same data, machine-checkable

Things it deliberately does not do: hash the multi-GB model weights (sizes and
mtimes instead, said plainly), decide what happens to a path, or claim the
archive is complete — completeness is checked against this list, later.

Usage:
  python scripts/archive_inventory.py
  python scripts/archive_inventory.py --verify   # re-check digests against the JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

MD_PATH = os.path.join(PROJECT_ROOT, "ARCHIVE_INVENTORY.md")
JSON_PATH = os.path.join(PROJECT_ROOT, "archive_inventory.json")

# role: docs      source of truth for what was decided, archive wholesale
#       history   produced material (trajectories, calibrations, reviews)
#       runtime   the current code reads it; moving it breaks the pipeline
#       code      retired *code* that is still imported somewhere: data
#                 archiving and code retirement are separate jobs, and moving
#                 a module out from under its importers breaks the package
#       external  outside the repository (weights, datasets), not archived
ASSETS = [
    ("plan.md", "docs", "计划与档案；顶部当前阶段一并封存"),
    ("claude_operation_log.md", "docs", "操作日志（增量追加）"),
    ("CLAUDE_CODE_G7_REPAIR_HANDOFF.md", "docs", "G7 修复交接文档"),
    ("NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md", "docs", "本方案本身（随代次留存，不归档）"),
    ("KNOWN_ISSUES.md", "docs", "问题清单（新版由 `KNOWN_ISSUES.md` 重写）"),
    ("sft_data/trajectories", "history", "237 集轨迹（v1/v2 时代）"),
    ("sft_data/trajectories_g7_b1", "history", "b1 轨迹 + VERSIONS.json"),
    ("sft_data/trajectories_g7_b2", "history", "b2 轨迹 + VERSIONS.json"),
    ("sft_data/trajectories_g7_b2_parser_bug", "history", "缺陷留证归档"),
    ("sft_data/trajectories_g7_b2_reasoning_prose", "history", "缺陷留证归档"),
    ("sft_data/trajectories_g7_b2_statefix", "history", "状态行对照素材"),
    ("sft_data/variants", "history", "237 集变体图"),
    ("sft_data/variants_g7_b1", "history", "b1 变体图 + 清单"),
    ("sft_data/variants_g7_b2", "history", "b2 变体图 + 清单"),
    ("sft_data/train/final", "history", "冻结基线 v1"),
    ("sft_data/train/final_v2", "history", "G4 转换产物（237 集）"),
    ("sft_data/train/batches/g7_b1", "history", "b1 评审样本 + 清单"),
    ("sft_data/train/batches/g7_b2", "history", "b2 评审样本 + 清单"),
    ("sft_data/train/sft_*.json", "history", "早期分桶样本（correct/conflict/format/…）"),
    ("sft_data/review", "history", "工作台页面、人工审核 JSON、dispositions"),
    ("sft_data/stream_a_normal", "history", "早期 SFT 数据流"),
    ("sft_data/stream_b_conflict", "history", "早期 SFT 数据流"),
    ("sft_data/admission_report.json", "history", "237 集准入报告"),
    ("sft_data/admission_report_g7_b1.json", "history", "b1 准入报告"),
    ("sft_data/metadata.json", "history", "早期数据元信息"),
    ("sft_data/trajectories_report.json", "history", "早期运行报告"),
    ("calibration", "runtime", "校准表（`reliability_table.json` 是**运行时默认路径**）"
                              "与历史审计报告混居；须先分离再归档"),
    ("calibration/set", "runtime", "划分与原始分数（构建脚本的输入）"),
    ("calibration/reliability_table_v1.json", "runtime", "旧表，供历史复算；新版不读"),
    ("sft_data/split_v2.json", "runtime", "数据划分（运行与构建都读）"),
    ("sft_data/FROZEN.json", "runtime", "冻结登记：**归档不等于取消门禁**（提案 §2.7）"),
    ("experts/frequency.py", "code", "退役专家 v1 实现，**仍被 experts/__init__ 与多个脚本/测试导入**"),
    ("experts/ela.py", "code", "退役专家实现（同样按代码退役处理，不随数据归档移动）"),
    ("experts/trufor.py", "runtime", "外部对照模型（不进主链，但不属本代次归档对象）"),
    ("scripts/evaluate_experts_g2.py", "runtime", "旧专家的评测脚本；退出后按退役归档"),
    ("scripts/calibrate_experts.py", "runtime", "旧校准流程；新代次沿用或重写前不能移"),
    ("scripts/build_reliability_table.py", "runtime", "校准表构建；新专家接入要改这个"),
    ("utils/reliability.py", "runtime", "运行时读取校准表；旧数值依赖的解除点"),
    ("dataset", "runtime", "数据源，继续使用（提案 §2.3）"),
    ("models", "external", "仓库内模型权重目录（若存在）"),
    ("/root/autodl-tmp/third_party/probe", "external", "PROBE 权重（不随仓库分发）"),
    ("/root/autodl-tmp/third_party/TruFor", "external", "TruFor 权重与代码"),
]

WEIGHT_LIKE = ("models", "third_party", "dataset")


def _digest_file(path: str, limit: int = 64 * 1024 * 1024) -> Optional[str]:
    """sha256 of a file, or None when it is too large to hash on every run."""
    try:
        if os.path.getsize(path) > limit:
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:16]
    except OSError:
        return None


def _digest_tree(directory: str, limit: int = 512 * 1024 * 1024) -> Dict[str, object]:
    """(count, bytes, digest-over-name-and-bytes) for a directory."""
    count = 0
    total = 0
    digest = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(directory)):
        dirs.sort()
        for name in sorted(files):
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            count += 1
            total += size
            digest.update(os.path.relpath(path, directory).encode("utf-8"))
            digest.update(str(size).encode("utf-8"))
            if total <= limit:
                try:
                    with open(path, "rb") as handle:
                        for chunk in iter(lambda: handle.read(1 << 20), b""):
                            digest.update(chunk)
                except OSError:
                    pass
    return {"files": count, "bytes": total,
            "digest": digest.hexdigest()[:16],
            "digest_covers_content": total <= limit}


ARCHIVE_ROLES = ("docs", "history")


def _importers(relative: str) -> List[str]:
    """
    Modules with a real *import statement* for this code path.

    A substring grep counts every mention of the word (`frequency_v2` matched
    `frequency`, and the inventory script matched itself) — the same
    false-positive shape seen elsewhere in this project.  Match the import
    statement, and exclude the longer name.
    """
    import re as _re

    module = (relative[:-3].replace("/", ".") if relative.endswith(".py")
              else relative)
    leaf = module.rsplit(".", 1)[-1]
    pattern = _re.compile(
        rf"^\s*(?:from\s+[\w.]*\b{leaf}\b\s+import|import\s+[\w.]*\b{leaf}\b(?!s|_v2))"
        rf"|^\s*from\s+\.{leaf}\s+import", _re.M)
    found = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "archive")]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, PROJECT_ROOT)
            if rel == relative:
                continue
            try:
                text = open(path, encoding="utf-8").read()
            except OSError:
                continue
            if pattern.search(text):
                found.append(rel)
    return sorted(found)


def _tracked(relative: str) -> str:
    """Whether git has this path (so a move is recoverable from history)."""
    try:
        result = subprocess.run(["git", "ls-files", "--", relative],
                                cwd=PROJECT_ROOT, capture_output=True, text=True,
                                timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return "yes" if result.stdout.strip() else "no"


def collect() -> List[dict]:
    import glob

    rows = []
    expanded = []
    for relative, role, note in ASSETS:
        if "*" in relative:
            matches = sorted(glob.glob(os.path.join(PROJECT_ROOT, relative)))
            for match in matches:
                expanded.append((os.path.relpath(match, PROJECT_ROOT), role, note))
            if not matches:
                expanded.append((relative, role, note))
        else:
            expanded.append((relative, role, note))

    for relative, role, note in expanded:
        path = relative if os.path.isabs(relative) else os.path.join(PROJECT_ROOT, relative)
        row = {"path": relative, "role": role, "note": note,
               "archive": role in ARCHIVE_ROLES,
               "exists": os.path.exists(path), "tracked_by_git": _tracked(relative)}
        if not os.path.exists(path):
            rows.append(row)
            continue
        if os.path.isdir(path):
            # the archive set is small enough to digest in full; the big
            # non-archive trees (dataset, external weights) only get a
            # partial digest, which is *not* proof of a complete backup
            row.update(_digest_tree(path,
                                    limit=(1 << 60) if role in ARCHIVE_ROLES
                                    else 512 * 1024 * 1024))
            row["kind"] = "dir"
        else:
            size = os.path.getsize(path)
            row.update({"kind": "file", "bytes": size,
                        "digest": _digest_file(path),
                        "digest_covers_content": size <= 64 * 1024 * 1024})
        if role == "code":
            importers = _importers(relative)
            row["importers"] = importers
            row["note"] += f"（被 {len(importers)} 处导入：{'、'.join(importers[:4])}"
            row["note"] += "…）" if len(importers) > 4 else "）"
        if any(word in relative for word in WEIGHT_LIKE) and row.get("bytes", 0) > 1 << 30:
            row["digest"] = row.get("digest")
            row["note"] += "（体积大：摘要只覆盖前 512 MB 的名称与字节数，内容未逐块校验）"
        rows.append(row)
    return rows


def _non_overlapping(rows: List[dict]) -> List[dict]:
    """Drop rows nested inside another row, so a total is not double-counted."""
    chosen = []
    for row in sorted(rows, key=lambda item: item["path"]):
        path = row["path"].rstrip("/") + "/"
        if any(path.startswith(parent["path"].rstrip("/") + "/") for parent in chosen):
            continue
        chosen.append(row)
    return chosen


def render(rows: List[dict]) -> str:
    lines = [
        "# v1 归档清单（预检；本文件由 `scripts/archive_inventory.py` 生成）",
        "",
        f"生成时间：{datetime.now().isoformat(timespec='seconds')}｜"
        "**未移动、未删除任何文件**",
        "",
        "**这张表能证明什么**：路径存在、大小、文件数、内容摘要（大目录只覆盖前 512 MB，"
        "已逐行注明）、是否被 git 跟踪。**不能证明什么**：归档目标位置是否足够、"
        "移动后历史复算是否仍可跑（需要路径映射）、以及完整性——完整性要用这张表在移动**之后**复核。",
        "",
        "**`role` 的含义**：`runtime` 表示当前代码会读它，**不能盲目搬走**"
        "（提案 §2.6）；`history` 是产物；`docs` 是决策记录；`external` 在仓库之外。",
        "",
        "| 路径 | 角色 | 类型 | 文件数 | 大小 | 摘要 | git | 说明 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        size = row.get("bytes")
        size_text = "—" if size is None else (
            f"{size / 1024 / 1024:.1f} MB" if size >= 1024 * 1024 else f"{size / 1024:.0f} KB")
        digest = row.get("digest") or "—"
        if row.get("digest") and not row.get("digest_covers_content", True):
            digest += "（部分）"
        lines.append(
            f"| `{row['path']}` | {row['role']} | {row.get('kind', '缺失')} | "
            f"{row.get('files', '—')} | {size_text} | {digest} | "
            f"{row['tracked_by_git']} | {row['note']} |")
    archive_rows = _non_overlapping([row for row in rows
                                     if row["exists"] and row.get("archive")])
    stays = [row for row in rows if row["exists"] and not row.get("archive")]
    lines += [
        "",
        "## 总量（先把口径说清楚）",
        "",
        f"- **待归档（role=docs+history，已去掉父子重叠）**：{len(archive_rows)} 项，"
        f"{sum(row.get('bytes', 0) for row in archive_rows) / 1024 / 1024:.1f} MB",
        f"- **保持原位（runtime/code/external/dataset）**：{len(stays)} 项，"
        f"{sum(row.get('bytes', 0) for row in stays) / 1024 / 1024:.1f} MB",
        "",
        "两个数字都**不是“磁盘总占用”**：清单里既有父子目录重叠（如 `calibration` 与其子目录），"
        "也有仓库之外的外部路径，还有本就不归档的 dataset 与权重。"
        "**大目录的部分摘要不足以证明完整备份**——真正移动之前，对每个待归档目录做一次"
        "全量摘要（本清单已对 docs/history 做全量），移动之后再用 `--verify` 逐行复核。",
        "",
        "## 归档与退役是两件事",
        "",
        "`role=code` 的项是**仍被导入的已退役实现**（如 `experts/frequency.py` 仍被 "
        "`experts/__init__.py`、评测脚本与测试导入）。**数据归档不动代码**；"
        "代码退役在第 4 阶段单独做：先删除/改写导入方，再移动实现。",
        "",
        "## 归档前必须处理的运行时依赖（提案 §2.6）",
        "",
        "| 依赖 | 谁在读 | 处理 |",
        "|---|---|---|",
        "| `calibration/reliability_table.json` | `utils/reliability.py`（**默认路径**）、"
        "控制器每会话 | 先切新版配置或临时兼容表，再归档旧表；`reliability_table_v1.json` 只给复算 |",
        "| `calibration/set/*`、`sft_data/split_v2.json` | 构建脚本、划分断言 | 新代次另立划分与原始分数目录，旧目录随后归档 |",
        "| `sft_data/FROZEN.json` | 三个写入入口的门禁 | 跟随代码保留；代次化命名后仍以它为准 |",
        "| `experts/trufor.py` | 外部对照（不进主链） | 保留到新专家评测完成 |",
        "| `dataset/` | 数据源 | 继续使用，不归档（提案 §2.3） |",
        "",
        "## 归档时**必须**同时给出的东西",
        "",
        "1. `PATH_MAP.json`：原路径 → 归档路径，供历史复算解析（提案 §2.5）；",
        "2. 移动**之后**用本清单复跑一次，逐行核对摘要（本文件的 `--verify`）；",
        "3. 代次命名与冻结登记升级（见 `NEW_GENERATION_NAMING.md`），"
        "避免冻结身份永久占用新版批次名。",
    ]
    return "\n".join(lines) + "\n"


def verify(rows: List[dict]) -> int:
    """Re-check every recorded digest; used after the move."""
    problems = []
    for row in rows:
        if not row.get("exists"):
            continue
        path = row["path"] if os.path.isabs(row["path"]) else os.path.join(
            PROJECT_ROOT, row["path"])
        if not os.path.exists(path):
            problems.append(f"缺失：{row['path']}")
            continue
        if row.get("kind") == "dir":
            fresh = _digest_tree(path)
        else:
            fresh = {"digest": _digest_file(path),
                     "bytes": os.path.getsize(path)}
        if row.get("digest") and fresh.get("digest") != row["digest"]:
            problems.append(f"摘要不一致：{row['path']}（记录 {row['digest']}，现在 {fresh.get('digest')}）")
        elif row.get("bytes") and fresh.get("bytes") != row["bytes"]:
            problems.append(f"大小不一致：{row['path']}（记录 {row['bytes']}，现在 {fresh.get('bytes')}）")
    if problems:
        print("核对失败：")
        for problem in problems:
            print("  -", problem)
        return 1
    print(f"核对通过：{len(rows)} 项与清单一致")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="用已记录的清单核对当前文件（归档之后跑）")
    args = parser.parse_args()

    if args.verify:
        with open(JSON_PATH, encoding="utf-8") as handle:
            rows = json.load(handle)["assets"]
        raise SystemExit(verify(rows))

    rows = collect()
    with open(JSON_PATH, "w", encoding="utf-8") as handle:
        json.dump({"generated_at": datetime.now().isoformat(timespec="seconds"),
                   "note": "预检清单；未移动任何文件",
                   "assets": rows}, handle, ensure_ascii=False, indent=2)
    with open(MD_PATH, "w", encoding="utf-8") as handle:
        handle.write(render(rows))

    present = [row for row in rows if row["exists"]]
    grand = sum(row.get("bytes", 0) for row in present)
    archive_rows = _non_overlapping([row for row in present if row.get("archive")])
    archive_total = sum(row.get("bytes", 0) for row in archive_rows)
    stays = [row for row in present if not row.get("archive")]
    print(f"清单完成：{len(present)}/{len(rows)} 项存在")
    print(f"  待归档（docs+history，已去父子重叠）：{len(archive_rows)} 项，"
          f"{archive_total / 1024 / 1024:.1f} MB")
    print(f"  保持原位（runtime/code/external/dataset）：{len(stays)} 项，"
          f"{sum(row.get('bytes', 0) for row in stays) / 1024 / 1024:.1f} MB")
    print("  注意：两项相加也不等于'磁盘总占用'（清单本身含父子重叠与外部路径）")
    print(f"  {os.path.relpath(MD_PATH, PROJECT_ROOT)}")
    print(f"  {os.path.relpath(JSON_PATH, PROJECT_ROOT)}")
    print("未移动、未删除任何文件；归档后用 --verify 复核。")


if __name__ == "__main__":
    main()
