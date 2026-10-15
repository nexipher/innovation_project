"""
把审核工作台导出的决定**写回仓库**（阶段 4.6）。

页面里保存的只是浏览器草稿（`localStorage`）——刷新不丢，但**不等于写进仓库**。
这条命令是唯一的写回路径，它经 `utils.review_store.check_import` 校验：

* 代次/批次/schema 不匹配 → **整份拒收**（跨代次、跨批次导入一律不接）；
* 本批没有的样本 → 拒收该条；
* 待修/拒绝没写原因、审核人空、分类不认识 → 拒收该条；
* 内容指纹与当前不一致 → **接受但标"已过期"**（不继承、也不静默沿用旧决定）。

用法：
  python scripts/g8_review_import.py review_g8_b1.json --generation g8 --batch b1
  python scripts/g8_review_import.py review_g8_b1.json --dry-run    # 只看会写什么

台账是**追加式** jsonl（`sft_data/<gen>/review/<batch>/dispositions.jsonl`）：
改主意、换审核人都留痕，`latest()` 取最后一条。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from utils.batch import batch_paths  # noqa: E402
from utils.review_store import (append_ledger, check_import,  # noqa: E402
                                load_ledger, progress, sample_fingerprint)


def current_fingerprints(generation: str, batch: str,
                         records_dir: str | None = None) -> dict:
    """当前每条样本的内容指纹（与工作台嵌进页面的那套算法**同一个函数**）。"""
    records_dir = records_dir or batch_paths(batch, generation)["trajectories"]
    fingerprints = {}
    for name in sorted(os.listdir(records_dir)):
        if not name.endswith(".json") or name == "metadata.json":
            continue
        with open(os.path.join(records_dir, name), encoding="utf-8") as handle:
            record = json.load(handle)
        trace_path = record.get("trace_path")
        trace = None
        if trace_path and os.path.exists(os.path.join(PROJECT_ROOT, trace_path)):
            with open(os.path.join(PROJECT_ROOT, trace_path), encoding="utf-8") as handle:
                trace = json.load(handle)
        fingerprints[record["trajectory_id"]] = sample_fingerprint(record, trace)
    return fingerprints


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("file", help="工作台导出的 JSON")
    parser.add_argument("--generation", default=None)
    parser.add_argument("--batch", default=None)
    parser.add_argument("--records-dir", default=None)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.file, encoding="utf-8") as handle:
        payload = json.load(handle)
    generation = args.generation or payload.get("generation") or ""
    batch = args.batch or payload.get("batch") or ""
    if not generation or not batch:
        print("缺代次/批次：文件里没有，命令行也没给。", file=sys.stderr)
        return 2

    fingerprints = current_fingerprints(generation, batch, args.records_dir)
    accepted, rejected = check_import(payload, generation, batch, fingerprints)
    if not accepted and rejected and rejected[0]["trajectory_id"] == "*":
        print(f"整份拒收：{rejected[0]['reason']}", file=sys.stderr)
        return 3

    for item in rejected:
        print(f"拒收 {item['trajectory_id']}：{item['reason']}")
    stale = [d for d in accepted if d.stale]
    for decision in stale:
        print(f"过期 {decision.trajectory_id}：{decision.stale_reason}")

    if args.dry_run:
        print(f"（dry-run）将写入 {len(accepted)} 条（其中过期 {len(stale)} 条），"
              f"拒收 {len(rejected)} 条")
        return 0

    path = append_ledger(generation, batch, accepted, path=args.ledger)
    state = progress(load_ledger(generation, batch, path=args.ledger),
                     total=len(fingerprints))
    print(f"已写入 {len(accepted)} 条 → {os.path.relpath(path, PROJECT_ROOT)}")
    print(f"进度：{json.dumps(state, ensure_ascii=False)}")
    print("提醒：这一步只写验收台账，**不改任何训练产物**（验收 ≠ 训练）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
