"""
登记 b2 的**验收协议身份**（2026-10-18 用户指定，plan §4.12b）。

外部 Agent 的启动门禁不比较"用户传入的声明"，而是比较**实际构建出来的管线**与
这里登记的验收身份。本脚本只做三件事：

  1. 用与门禁同一套逐样本核对（`utils/acceptance.py`）确认 b2 已全部通过；
  2. 从 b2 自己的 `metadata.json` 采集身份，并核对每个协议文件的**当前字节**
     与 **b2 运行提交**中的字节一致——不一致说明 b2 验收的不是当前协议；
  3. 写 `sft_data/<gen>/review/<batch>/accepted_protocol.json`（最新）并追加
     `accepted_protocol_history.jsonl`（历史版本）。

**不写 `FROZEN.json`**：冻结仍是用户的显式动作。本脚本也不会修改任何审核决定。

用法：
  python scripts/g8_accept_protocol.py --generation g8 --batch b2 --reviewer hj
  python scripts/g8_accept_protocol.py --generation g8 --batch b2 --reviewer hj --dry-run
  python scripts/g8_accept_protocol.py --generation g8 --batch b2 --reviewer hj \\
      --reason "验收通过后首次登记"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from utils.acceptance import format_problems, verify_review_coverage  # noqa: E402
from utils.batch import batch_paths  # noqa: E402
from utils.protocol_identity import (  # noqa: E402
    SCHEMA, file_sha256, identity_from_metadata, identity_sha256,
    protocol_file_paths)

HISTORY_NAME = "accepted_protocol_history.jsonl"


def _git(args: list) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                          text=True, timeout=60)


def _commit_bytes(commit: str, relative: str):
    """b2 提交里该文件的字节；文件不在提交里就返回 (None, 原因)。"""
    result = subprocess.run(["git", "show", f"{commit}:{relative}"],
                            cwd=PROJECT_ROOT, capture_output=True, timeout=60)
    if result.returncode != 0:
        return None, result.stderr.decode("utf-8", "replace").strip()
    return result.stdout, ""


def _verify_protocol_files_unchanged_since(commit: str, paths: list) -> list:
    """逐个文件比对：当前字节 vs b2 提交中的字节。返回问题清单。"""
    problems = []
    for relative in paths:
        absolute = os.path.join(PROJECT_ROOT, relative)
        if not os.path.isfile(absolute):
            problems.append(f"{relative}: 当前文件不存在")
            continue
        old, error = _commit_bytes(commit, relative)
        if old is None:
            problems.append(f"{relative}: 在 b2 提交 {commit[:12]} 里读不到（{error}）")
            continue
        current = file_sha256(absolute)
        at_b2 = hashlib.sha256(old).hexdigest()
        if current != at_b2:
            problems.append(
                f"{relative}: 自 b2 后已变（b2 {at_b2[:12]}…，当前 {current[:12]}…）")
    return problems


def _non_output_dirty(code: dict) -> list:
    """b2 代码身份里，非输出目录的 dirty 路径。输出目录不算代码改动。"""
    return [path for path in (code.get("dirty_paths") or [])
            if not path.startswith(("sft_data/", "traces/"))]


def register(generation: str, batch: str, reviewer: str, *,
             reason: str = "", dry_run: bool = False,
             records_dir: str | None = None, ledger_path: str | None = None,
             manifest_path: str | None = None, metadata_path: str | None = None,
             accepted_path: str | None = None) -> dict:
    if not reviewer.strip():
        raise SystemExit("审核人不能为空：验收协议身份必须知道是谁确认的。")
    paths = batch_paths(batch, generation)
    records_dir = records_dir or paths["trajectories"]
    metadata_path = metadata_path or os.path.join(records_dir, "metadata.json")
    accepted_path = accepted_path or os.path.join(
        PROJECT_ROOT, "sft_data", generation, "review", batch,
        "accepted_protocol.json")

    # 1) 逐样本核对：与门禁同一实现，失败不登记。
    coverage = verify_review_coverage(
        generation, batch, records_dir=records_dir, ledger_path=ledger_path,
        manifest_path=manifest_path, metadata_path=metadata_path)
    if not coverage["ok"]:
        raise SystemExit(
            f"验收协议身份未登记：{generation}/{batch} 的逐样本验收未通过。\n  "
            + "\n  ".join(format_problems(coverage))
            + "\n先完成人工验收并写回台账，再登记协议身份。")

    # 2) 从 b2 自己的批次元数据取身份。
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"批次元数据无法读取：{metadata_path}（{exc}）")
    if metadata.get("generation") != generation or metadata.get("batch") != batch:
        raise SystemExit(
            f"批次元数据属于 {metadata.get('generation')}/{metadata.get('batch')}，"
            f"不是 {generation}/{batch}：{metadata_path}")

    code = metadata.get("code") or {}
    commit = (code.get("commit") or "").strip()
    if len(commit) < 12 or any(ch not in "0123456789abcdef" for ch in commit.lower()):
        raise SystemExit(
            f"批次元数据没有可用的代码提交（code.commit={commit!r}），"
            "无法核对协议文件自 b2 后未变。")
    dirty_non_output = _non_output_dirty(code)
    if dirty_non_output:
        raise SystemExit(
            "b2 运行时已跟踪代码文件是 dirty 的，且不全是输出目录：\n  "
            + "\n  ".join(dirty_non_output)
            + "\n无法证明协议文件就是提交里的那份；不能登记。")

    experts = metadata.get("experts") or {}
    classes = {spec.get("class") for spec in experts.values() if spec.get("class")}
    files = protocol_file_paths(classes)
    file_problems = _verify_protocol_files_unchanged_since(commit, files)
    if file_problems:
        raise SystemExit(
            "验收协议身份未登记：这些协议文件自 b2 运行后已变化，"
            "b2 的人工验收不再覆盖当前协议。\n  " + "\n  ".join(file_problems)
            + "\n若变更是有意的，先重新跑 b2 并重新验收；不能用旧验收登记新协议。")

    # 3) 归一化身份并检查必需字段。
    identity = identity_from_metadata(metadata)
    problems = []
    model = identity["model"]
    for field in ("checkpoint", "revision", "dtype", "inference_mode"):
        if not model.get(field):
            problems.append(f"模型身份缺 {field}")
    if not identity["prompts"].get("forensic_sha256"):
        problems.append("提示词身份缺 forensic_sha256")
    if not identity["tools"].get("tools_served"):
        problems.append("工具身份缺 tools_served")
    if not identity["tools"].get("policy"):
        problems.append("工具身份缺 policy")
    for name, spec in sorted(identity["experts"].items()):
        if not spec.get("weights_sha256"):
            problems.append(f"专家 {name} 缺 weights_sha256")
        if not spec.get("preprocess_sha256"):
            problems.append(f"专家 {name} 缺登记表预处理（preprocess）")
    if problems:
        raise SystemExit("验收协议身份不完整，不能登记：\n  " + "\n  ".join(problems))

    warnings = {}
    if code.get("dirty"):
        warnings["b2_code_dirty"] = True
        warnings["b2_dirty_paths_non_output"] = dirty_non_output
        warnings["b2_dirty_note"] = ("b2 运行时记录的 dirty 路径全部是输出目录"
                                     "（sft_data/traces）；协议文件已逐个与提交核对。")

    payload = {
        "schema": SCHEMA,
        "generation": generation,
        "batch": batch,
        "accepted_by": reviewer,
        "accepted_at": datetime.now().isoformat(timespec="seconds"),
        "reason": reason.strip(),
        "source": {
            "metadata": os.path.relpath(metadata_path, PROJECT_ROOT),
            "metadata_sha256": file_sha256(metadata_path),
            "code_commit": commit,
        },
        "identity": identity,
        "identity_sha256": identity_sha256(identity),
        "coverage": {
            "records": coverage["current"],
            "expected": coverage["expected"],
            "passed": coverage["current"],
            "extra_decisions": coverage["extra_decisions"],
        },
        "warnings": warnings,
    }

    if dry_run:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print("\n（dry-run）未写任何文件")
        return payload

    if os.path.exists(accepted_path) and not reason.strip():
        raise SystemExit(
            f"{accepted_path} 已存在；重新登记必须用 --reason 说明这次变更，"
            "历史记录会保留在 " + HISTORY_NAME + "。")

    os.makedirs(os.path.dirname(accepted_path), exist_ok=True)
    temporary = accepted_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, accepted_path)
    with open(os.path.join(os.path.dirname(accepted_path), HISTORY_NAME),
              "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--batch", required=True)
    parser.add_argument("--reviewer", required=True,
                        help="确认本次验收的人（写进身份文件）")
    parser.add_argument("--reason", default="",
                        help="重新登记时必须说明变更原因")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--records-dir", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--accepted", default=None)
    args = parser.parse_args()

    payload = register(
        args.generation, args.batch, args.reviewer,
        reason=args.reason, dry_run=args.dry_run,
        records_dir=args.records_dir, ledger_path=args.ledger,
        manifest_path=args.manifest, metadata_path=args.metadata,
        accepted_path=args.accepted)
    target = args.accepted or os.path.join(
        PROJECT_ROOT, "sft_data", args.generation, "review", args.batch,
        "accepted_protocol.json")
    print(f"验收协议身份：{os.path.relpath(target, PROJECT_ROOT)}")
    print(f"identity_sha256: {payload['identity_sha256']}")
    print(f"模型：{payload['identity']['model']['checkpoint']}@"
          f"{payload['identity']['model']['revision'][:12]}")
    print(f"专家：{', '.join(payload['identity']['tools']['tools_served'])}")
    print(f"证据包：{payload['identity']['evidence_pack']['version']}")
    for name, warning in (payload.get("warnings") or {}).items():
        print(f"警告 {name}: {warning}")
    if args.dry_run:
        print("（dry-run）未写任何文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
