#!/usr/bin/env python3
"""
Stage 1: move the v1 historical material into the archive (提案 §2).

Only `role=docs` and `role=history` items move, and only ones that are not
still read by live code:

  * the **dispositions ledger** (`sft_data/review/dispositions.jsonl`) stays:
    `record_review` reads it at runtime and it is append-only;
  * **code** never moves (retired implementations are still imported);
  * **runtime** assets (calibration, splits, FROZEN/GENERATIONS) never move.

What it writes, beside the moved files:

    archive/legacy_v1_20261015/MANIFEST.json   what moved, from where, digests
    archive/PATH_MAP.json                      old path -> archive path
    archive/legacy_v1_20261015/NOTE.md         what this archive is, and its limits

The archived copies of the review workstations keep working: their relative
image links are rewritten to the mapped location, and the rewrite is recorded
(`links_rewritten`) rather than left for someone to discover.

Usage:
  python scripts/archive_v1.py --dry-run     # print the plan, move nothing
  python scripts/archive_v1.py               # do it
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

ARCHIVE_ROOT = os.path.join(PROJECT_ROOT, "archive", "legacy_v1_20261015")
PATH_MAP = os.path.join(PROJECT_ROOT, "archive", "PATH_MAP.json")
INVENTORY = os.path.join(PROJECT_ROOT, "docs", "archive", "archive_inventory.json")

# old path -> (section, new name).  Sections follow the proposal's layout.
MOVES = [
    ("plan.md", "docs", "plan_v1.md"),
    ("claude_operation_log.md", "docs", "claude_operation_log_v1.md"),
    ("CLAUDE_CODE_G7_REPAIR_HANDOFF.md", "docs", "CLAUDE_CODE_G7_REPAIR_HANDOFF.md"),
    ("sft_data/trajectories", "experiments", "trajectories"),
    ("sft_data/trajectories_g7_b1", "experiments", "trajectories_g7_b1"),
    ("sft_data/trajectories_g7_b2", "experiments", "trajectories_g7_b2"),
    ("sft_data/trajectories_g7_b2_parser_bug", "experiments", "trajectories_g7_b2_parser_bug"),
    ("sft_data/trajectories_g7_b2_reasoning_prose", "experiments", "trajectories_g7_b2_reasoning_prose"),
    ("sft_data/trajectories_g7_b2_statefix", "experiments", "trajectories_g7_b2_statefix"),
    ("sft_data/trajectories_report.json", "experiments", "trajectories_report.json"),
    ("sft_data/variants", "experiments", "variants"),
    ("sft_data/variants_g7_b1", "experiments", "variants_g7_b1"),
    ("sft_data/variants_g7_b2", "experiments", "variants_g7_b2"),
    ("sft_data/train/final", "experiments", "train_final"),
    ("sft_data/train/final_v2", "experiments", "train_final_v2"),
    ("sft_data/train/batches/g7_b1", "experiments", "train_batches_g7_b1"),
    ("sft_data/train/batches/g7_b2", "experiments", "train_batches_g7_b2"),
    ("sft_data/stream_a_normal", "experiments", "stream_a_normal"),
    ("sft_data/stream_b_conflict", "experiments", "stream_b_conflict"),
    ("sft_data/admission_report.json", "experiments", "admission_report.json"),
    ("sft_data/admission_report_g7_b1.json", "experiments", "admission_report_g7_b1.json"),
    ("sft_data/metadata.json", "experiments", "metadata.json"),
    ("sft_data/review/index.html", "review", "index.html"),
    ("sft_data/review/samples", "review", "samples"),
    ("sft_data/review/review_sets.json", "review", "review_sets.json"),
    ("sft_data/review/worklist.csv", "review", "worklist.csv"),
    ("sft_data/review/review_decisions.json", "review", "review_decisions.json"),
    ("sft_data/review/g7_b1", "review", "g7_b1"),
    ("sft_data/review/g7_b2", "review", "g7_b2"),
]
EARLY_SAMPLES = "sft_data/train/sft_*.json"

STAYS = {
    "sft_data/review/dispositions.jsonl":
        "运行时记录（record_review 读写、增量追加），归档会中断工具，故保留原位",
    "sft_data/FROZEN.json": "冻结门禁的登记表，跟随代码",
    "sft_data/GENERATIONS.json": "代次索引，跟随代码",
    "sft_data/split_v2.json": "数据划分，运行与构建都读",
    "sft_data/g8": "新代次工作区（空骨架）",
    "calibration": "校准与历史审计混居；校准表是运行时默认路径，须先解除依赖再归档（阶段 4）",
    "dataset": "数据源，继续使用",
}


def _digest_dir(path: str) -> Dict[str, object]:
    import hashlib

    count = 0
    total = 0
    digest = hashlib.sha256()
    for root, dirs, files in sorted(os.walk(path)):
        dirs.sort()
        for name in sorted(files):
            full = os.path.join(root, name)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            count += 1
            total += size
            digest.update(os.path.relpath(full, path).encode("utf-8"))
            digest.update(str(size).encode("utf-8"))
            try:
                with open(full, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(chunk)
            except OSError:
                pass
    return {"files": count, "bytes": total, "digest": digest.hexdigest()[:16]}


def _digest(path: str) -> Dict[str, object]:
    import hashlib

    if os.path.isdir(path):
        return _digest_dir(path)
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return {"files": 1, "bytes": os.path.getsize(path),
                "digest": digest.hexdigest()[:16]}
    except OSError:
        return {}


def plan() -> List[dict]:
    import glob

    entries = []
    for old, section, name in MOVES:
        if not os.path.exists(os.path.join(PROJECT_ROOT, old)):
            entries.append({"from": old, "to": None, "missing": True})
            continue
        entries.append({"from": old, "to": f"archive/legacy_v1_20261015/{section}/{name}",
                        "section": section})
    for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, EARLY_SAMPLES))):
        relative = os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")
        entries.append({"from": relative,
                        "to": f"archive/legacy_v1_20261015/experiments/{os.path.basename(path)}",
                        "section": "experiments"})
    return entries


_LINK = re.compile(r'(?P<attr>\b(?:src|href))\s*=\s*"(?P<url>[^"]*)"')


def _mapped(project_relative: str, prefix_map: List[tuple]) -> Optional[str]:
    """Project-relative path -> archive-relative path, if it moved."""
    for old, new in prefix_map:
        if project_relative == old:
            return new
        if project_relative.startswith(old + "/"):
            return new + project_relative[len(old):]
    return None


def _transform_page(text: str, old_page: str, new_page: str,
                    prefix_map: List[tuple]) -> str:
    """
    Keep a moved page pointing at what it pointed at before the move.

    Each relative link is resolved against the page's **original** location,
    mapped through the same table that moved the files, and re-expressed
    relative to the page's **new** location.  Links to things that did not move
    (`dataset/`, `traces/`) come out with a different `../` depth but the same
    target — which is the point: an archived page that looks fine and shows a
    broken image is worse than one that plainly says where it points.
    """
    old_dir = os.path.dirname(old_page)
    new_dir = os.path.dirname(new_page)

    def replace(match: "re.Match") -> str:
        url = match.group("url")
        if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", url) or url.startswith(("#", "/")):
            return match.group(0)  # scheme-qualified, root-relative or fragment
        target, _, suffix = url.partition("#")
        if not target:
            return match.group(0)
        resolved = os.path.normpath(os.path.join(old_dir, target)).replace(os.sep, "/")
        if resolved.startswith(".."):  # escaped the project: leave it
            return match.group(0)
        new_target = _mapped(resolved, prefix_map) or resolved
        link = os.path.relpath(os.path.join(PROJECT_ROOT, new_target),
                               os.path.join(PROJECT_ROOT, new_dir)).replace(os.sep, "/")
        if suffix:
            link = f"{link}#{suffix}"
        return f'{match.group("attr")}="{link}"'

    return _LINK.sub(replace, text)


def _page_maps(moved: List[dict]):
    prefix_map = sorted(
        ((entry["from"].rstrip("/"), entry["to"].rstrip("/")) for entry in moved),
        key=lambda pair: -len(pair[0]))
    reverse_map = sorted(((to, old) for old, to in prefix_map), key=lambda p: -len(p[0]))
    return prefix_map, reverse_map


def rewrite_links(archive_path: str, moved: List[dict]) -> int:
    """Apply the page transform to every moved `.html` in the archive."""
    prefix_map, reverse_map = _page_maps(moved)
    rewritten = 0
    for root, _dirs, files in os.walk(archive_path):
        for name in sorted(files):
            if not name.endswith(".html"):
                continue
            page = os.path.join(root, name)
            page_relative = os.path.relpath(page, PROJECT_ROOT).replace(os.sep, "/")
            old_page = _mapped(page_relative, reverse_map)
            if old_page is None:  # not a moved page: leave it alone
                continue
            try:
                text = open(page, encoding="utf-8").read()
            except OSError:
                continue
            new_text = _transform_page(text, old_page, page_relative, prefix_map)
            if new_text != text:
                with open(page, "w", encoding="utf-8") as handle:
                    handle.write(new_text)
                rewritten += 1
    return rewritten


def _blob_revision(old_path: str, new_path: Optional[str] = None) -> Optional[str]:
    """
    The revision whose tree still holds `old_path` as it was **before the move**.

    Reading `HEAD:<old_path>` is only right while the move is uncommitted —
    once it lands, `HEAD:plan.md` is the *new* plan and the check compares the
    archived v1 document against a different file (two false "content changed"
    failures, seen when this first ran after the commit).

    Nor is "the commit that deleted the old path" correct in general: the old
    path may still exist (`plan.md` does — it is the new plan).  What is always
    true is that the move **added** the new path; the commit that added it is
    the move, and its parent holds the pre-move state.
    """
    if new_path:
        added = subprocess.run(
            ["git", "log", "--diff-filter=A", "--format=%H", "-1", "--", new_path],
            cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
        if added:
            parent = subprocess.run(["git", "rev-parse", "--verify", f"{added[0]}^"],
                                    cwd=PROJECT_ROOT, capture_output=True, text=True)
            if parent.returncode == 0:
                return f"{parent.stdout.strip()}:{old_path}"
    # No recorded addition (or a root commit): fall back to the last state
    # before the path disappeared, which is the deletion commit's parent.
    log = subprocess.run(["git", "log", "--format=%H", "-1", "--", old_path],
                         cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.split()
    if not log:
        return None
    parent = subprocess.run(["git", "rev-parse", "--verify", f"{log[0]}^"],
                            cwd=PROJECT_ROOT, capture_output=True, text=True)
    if parent.returncode != 0:
        return None
    return f"{parent.stdout.strip()}:{old_path}"


def _git_blob(old_path: str, new_path: Optional[str] = None) -> Optional[bytes]:
    """The content of `old_path` at the revision where it last had that content."""
    revision = _blob_revision(old_path, new_path)
    if revision is None:
        return None
    result = subprocess.run(["git", "show", revision],
                            cwd=PROJECT_ROOT, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def record_links(moved: List[dict]) -> List[dict]:
    """
    Write down exactly which pages the link rewrite changed, and prove it.

    The rewrite is the one thing in this move that changes bytes, so it is the
    one thing that could hide a mistake.  For every archived page whose content
    differs from its committed blob, the transform is re-applied to that blob:
    if the result is not byte-for-byte what is on disk, the page was changed by
    something other than the recorded rewrite, and the move stops here.

    The reference is the revision *before the move* (`_blob_revision`), so this
    keeps working after the move is committed — the earlier version read `HEAD`
    and began comparing the archived plan against the new one.
    """
    import hashlib

    prefix_map, reverse_map = _page_maps(moved)
    records = []
    for root, _dirs, files in os.walk(ARCHIVE_ROOT):
        for name in sorted(files):
            if not name.endswith(".html"):
                continue
            page = os.path.join(root, name)
            page_relative = os.path.relpath(page, PROJECT_ROOT).replace(os.sep, "/")
            old_page = _mapped(page_relative, reverse_map)
            if old_page is None:
                continue
            blob = _git_blob(old_page, page_relative)
            if blob is None:
                continue
            disk = open(page, "rb").read()
            if disk == blob:
                continue
            expected = _transform_page(blob.decode("utf-8"), old_page,
                                       page_relative, prefix_map).encode("utf-8")
            if expected != disk:
                raise SystemExit(
                    f"归档页面被登记的改写之外的东西改动了：{page_relative}\n"
                    "（这不是链接深度问题，先查清楚再继续。）")
            records.append({"page": page_relative, "was": old_page,
                            "pre_digest": hashlib.sha256(blob).hexdigest()[:16],
                            "pre_bytes": len(blob), "post_bytes": len(disk)})
    return records


def verify(moved: List[dict]) -> int:
    """
    Check the archive file by file — this is the "移动后逐行核对" pass.

    Two sources of truth, used where each one is actually strong:

      * committed content (`HEAD`) for tracked files: the moved file must be
        byte-identical to the blob it came from, except for pages the link
        rewrite changed (each recorded with its pre-move digest);
      * the pre-move inventory (`archive_inventory.json`) for the ignore-listed
        binaries git never held — file counts and byte totals must match.
    """
    import hashlib

    table = {entry["page"]: entry for entry in record_links(moved)}
    problems = []
    checked = 0
    untracked_bytes = 0
    untracked_files = 0

    for entry in moved:
        target = os.path.join(PROJECT_ROOT, entry["to"])
        if not os.path.exists(target):
            problems.append(f"缺失：{entry['to']}")
            continue
        files = [target] if os.path.isfile(target) else [
            os.path.join(root, name)
            for root, _dirs, names in os.walk(target) for name in sorted(names)]
        for full in files:
            relative = os.path.relpath(full, PROJECT_ROOT).replace(os.sep, "/")
            suffix = relative[len(entry["to"].rstrip("/")):].lstrip("/")
            old_relative = (entry["from"].rstrip("/") + ("/" + suffix if suffix else ""))
            blob = _git_blob(old_relative, relative)
            digest = hashlib.sha256(open(full, "rb").read()).hexdigest()
            if blob is None:  # never in git: covered by the count/bytes check
                untracked_files += 1
                untracked_bytes += os.path.getsize(full)
            elif digest != hashlib.sha256(blob).hexdigest():
                if relative not in table:
                    problems.append(f"内容变更（非登记的链接改写）：{relative}")
            checked += 1

    # The binary side: same files, same bytes (the inventory recorded both
    # before the move).  A file list that shrank is exactly what this catches.
    try:
        inventory = json.load(open(INVENTORY, encoding="utf-8"))["assets"]
    except (OSError, ValueError):
        inventory = []
    recorded = {row["path"]: row for row in inventory if row.get("role") == "history"}
    for entry in moved:
        row = recorded.get(entry["from"])
        if not row or not row.get("bytes"):
            continue
        # The inventory recorded a file count only for directories; single
        # files carry bytes alone, and comparing a missing count to 1 would
        # report nine failures that are not failures.
        if entry.get("bytes") != row.get("bytes") or (
                row.get("files") is not None and entry.get("files") != row.get("files")):
            problems.append(
                f"体量与移动前清单不符：{entry['from']}"
                f"（归档记录 {entry.get('files')} 个文件 / {entry.get('bytes')} 字节，"
                f"清单 {row.get('files')} 个 / {row.get('bytes')} 字节）")

    if problems:
        print("核对失败：")
        for problem in problems:
            print("  -", problem)
        return 1
    print(f"核对通过：{len(moved)} 项、{checked} 个文件"
          f"（其中 {len(table)} 页为登记的链接改写，"
          f"{untracked_files} 个 git 未收录的二进制按 体量/计数 核对，共 {untracked_bytes} 字节）")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="逐文件核对已完成的归档（须在提交这次移动之前跑）")
    parser.add_argument("--record-links", action="store_true",
                        help="把（并校验）链接改写登记进 MANIFEST.json")
    args = parser.parse_args()

    if args.verify:
        with open(PATH_MAP, encoding="utf-8") as handle:
            moved = json.load(handle)["moved"]
        raise SystemExit(verify(moved))

    if args.record_links:
        with open(PATH_MAP, encoding="utf-8") as handle:
            moved = json.load(handle)["moved"]
        manifest_path = os.path.join(ARCHIVE_ROOT, "MANIFEST.json")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        records = record_links(moved)
        manifest["links_rewritten"] = {"pages": len(records), "files": records}
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        print(f"已登记 {len(records)} 页链接改写（每页含改写前的摘要与字节数）")
        return

    entries = plan()
    movable = [entry for entry in entries if entry.get("to")]
    missing = [entry for entry in entries if entry.get("missing")]
    print(f"计划移动 {len(movable)} 项，缺失 {len(missing)} 项")
    for entry in movable:
        print(f"  {entry['from']}  ->  {entry['to']}")
    for entry in missing:
        print(f"  （缺失，跳过）{entry['from']}")
    if args.dry_run:
        print("\n（dry-run：未移动任何文件）")
        return

    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    moved: List[dict] = []
    for entry in movable:
        source = os.path.join(PROJECT_ROOT, entry["from"])
        target = os.path.join(PROJECT_ROOT, entry["to"])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        digest = _digest(source)
        # Plain move, not `git mv`: these trees mix tracked files with
        # gitignored images (the variant PNGs never enter the repository), and
        # `git mv` refuses a directory it does not fully track.  Git's rename
        # detection still pairs the old and new paths at commit time.
        shutil.move(source, target)
        moved.append({"from": entry["from"], "to": entry["to"],
                      "section": entry.get("section"), **digest})
        print(f"  移动 {entry['from']} -> {entry['to']}")

    # the archived workstations keep their images
    rewritten = rewrite_links(ARCHIVE_ROOT, moved)
    link_records = record_links(moved)

    generated_at = datetime.now().isoformat(timespec="seconds")
    with open(PATH_MAP, "w", encoding="utf-8") as handle:
        json.dump({"archived_at": generated_at,
                   "archive_root": "archive/legacy_v1_20261015",
                   "note": "旧路径 -> 归档路径；历史复算用 utils.batch.resolve_path 解析，"
                           "命中归档会在报告里记为 resolved_from_archive",
                   "moved": [{"from": entry["from"], "to": entry["to"],
                              "kind": "dir" if entry.get("files", 0) > 1 else "file",
                              "files": entry.get("files"), "bytes": entry.get("bytes"),
                              "digest": entry.get("digest")} for entry in moved]},
                  handle, ensure_ascii=False, indent=2)

    manifest = {
        "archived_at": generated_at,
        "generation": "g7（含更早的 g4_trial/237 集）",
        "why": "新版专家组与 Qwen3-VL 迁移（docs/design/NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md §2）："
               "旧实验全部冻结，不删除、不覆盖；新代次在新目录生成",
        "docs": [entry for entry in moved if entry["section"] == "docs"],
        "experiments": [entry for entry in moved if entry["section"] == "experiments"],
        "review": [entry for entry in moved if entry["section"] == "review"],
        "stays_in_place": STAYS,
        # The rewrite is the only thing here that changes bytes; the table is
        # what makes that a recorded fact instead of an assumption.  Each
        # entry was proved by re-applying the transform to the committed blob.
        "links_rewritten": {"pages": len(link_records), "files": link_records},
        "limits": [
            "冻结登记（sft_data/FROZEN.json）与代次索引（sft_data/GENERATIONS.json）留在原位，"
            "归档不解除任何门禁",
            "归档副本的路径由 archive/PATH_MAP.json 描述；旧命令不会直接可跑",
            "自本文件起，任何统计都必须说明读的是 live 还是 archive",
        ],
    }
    with open(os.path.join(ARCHIVE_ROOT, "MANIFEST.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    note = f"""# v1 归档说明（legacy_v1_20261015）

{generated_at} 归档。**这份归档是历史实验，不是训练材料**：全部批次已冻结
（`sft_data/FROZEN.json`），训练入口遇到即拒绝。

- 清单：[`MANIFEST.json`](MANIFEST.json)（每一项的原路径、大小、摘要）
- 路径映射：[`../PATH_MAP.json`](../PATH_MAP.json)（旧路径 → 归档路径）
- 留在原位的运行时资产见 MANIFEST 的 `stays_in_place`

**使用归档时的三条规矩**

1. 复算请走 `utils.batch.resolve_path()`：命中归档副本时，报告里必须写
   `resolved_from_archive=true`——"读的是副本"是结果的一部分。
2. 归档页面的图片链接已按映射改写（{len(link_records)} 个 HTML 文件，逐页记录在
   `MANIFEST.json` 的 `links_rewritten`，含改写前的摘要）；**原页面未改**。
3. 归档不解除门禁：`apply_review` / `build_final_v2` / 轨迹生成对冻结批次仍然拒绝。
"""
    with open(os.path.join(ARCHIVE_ROOT, "NOTE.md"), "w", encoding="utf-8") as handle:
        handle.write(note)

    print(f"\n归档完成：{len(moved)} 项 → {os.path.relpath(ARCHIVE_ROOT, PROJECT_ROOT)}/")
    print(f"  路径映射：{os.path.relpath(PATH_MAP, PROJECT_ROOT)}")
    print(f"  清单：{os.path.relpath(os.path.join(ARCHIVE_ROOT, 'MANIFEST.json'), PROJECT_ROOT)}")
    print(f"  改写链接的页面：{rewritten}（逐页记录在 MANIFEST.json）")
    print("  下一步：python scripts/archive_v1.py --verify（须在提交前跑）")


if __name__ == "__main__":
    main()
