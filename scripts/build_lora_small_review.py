"""Regenerate the 24-sample LoRA list and dry-run summary from records.

Labels are read from `sft_candidates_20261127/records.json`; no hand-copied
label is trusted.  The expected truth/target distributions are asserted, so a
changed record cannot silently leave the list stale.

Outputs:
  sft_data/lora_small_20261128/TRAINING_CANDIDATES.md
  sft_data/lora_small_20261128/DRY_RUN.md
  sft_data/lora_small_20261128/selection_pending.json (labels/statistics added)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
CANDIDATES = ROOT / "sft_data/sft_candidates_20261127"

TRUTH = {"Fake": 16, "Real": 8, "Uncertain": 0}
TARGET = {"Fake": 16, "Real": 5, "Uncertain": 3}


def _dist(rows, key):
    counts = Counter(row[key] for row in rows)
    return {name: counts.get(name, 0) for name in
            ("Fake", "Real", "Uncertain")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(SMALL))
    args = parser.parse_args()
    out = Path(args.out)
    selection = json.loads((out / "selection_pending.json").read_text(
        encoding="utf-8"))
    records = {row["sample_id"]: row for row in json.loads(
        (CANDIDATES / "records.json").read_text(encoding="utf-8"))}
    manifest = {row["sample_id"]: row for row in json.loads(
        (CANDIDATES / "manifest.json").read_text(encoding="utf-8"))["samples"]}
    dry = json.loads((out / "dry_run.json").read_text(encoding="utf-8"))
    dry_by_id = {row["sample_id"]: row for row in dry["samples"]}
    ids = selection["sample_ids"]
    if len(ids) != 24 or len(set(ids)) != 24:
        raise SystemExit(f"selection must contain 24 unique ids, got {len(ids)}")
    missing = [sid for sid in ids if sid not in records or sid not in dry_by_id]
    if missing:
        raise SystemExit(f"missing records/dry-run rows: {missing}")

    rows = []
    for sid in ids:
        record = records[sid]
        source = manifest[sid]
        dry_row = dry_by_id[sid]
        rows.append({
            "sample_id": sid,
            "kind": source.get("kind"),
            "role": source.get("selection_role"),
            "truth": record["ground_truth"],
            "whole": record["whole_image_generation"],
            "local": record["local_manipulation"],
            "overall": record["final_verdict"],
            "calls": record["counters"]["expert_calls"],
            "tokens": dry_row["input_tokens"],
            "images": dry_row["total_pil_images"],
            "images_per_tool_turn": dry_row["images_per_tool_turn"],
            "approved": sid in selection["approved_reuse"],
        })
    truth = _dist(rows, "truth")
    target = _dist(rows, "overall")
    if truth != TRUTH:
        raise SystemExit(f"truth distribution {truth} != {TRUTH}")
    if target != TARGET:
        raise SystemExit(f"target distribution {target} != {TARGET}")

    selection["records_source"] = "sft_data/sft_candidates_20261127/records.json"
    selection["labels"] = {row["sample_id"]: {
        "truth": row["truth"], "whole": row["whole"],
        "local": row["local"], "overall": row["overall"],
        "tokens": row["tokens"], "images": row["images"],
    } for row in rows}
    selection["truth_distribution"] = truth
    selection["target_distribution"] = target
    selection["dry_run"] = {
        "path": "sft_data/lora_small_20261128/dry_run.json",
        "longest": dry["longest"]["sample_id"],
        "most_images": dry["most_images"]["sample_id"],
        "top3_tokens": dry["top3_tokens"],
        "top3_images": dry["top3_images"],
        "profiling_set": dry["profiling_set"],
        "over_max_length": dry["over_max_length"],
    }
    (out / "selection_pending.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    lines = [
        "# 24 条 LoRA 小试训练样板 — 待确认清单",
        "",
        "状态：**pending confirmation**。尚未启动 GPU 训练。标签以",
        "`sft_data/sft_candidates_20261127/records.json` 为准；统一按",
        "**整图 / 局部 / 总体** 展示。",
        "",
        "## 分布",
        "",
        "| 分布 | Fake | Real | Uncertain |",
        "|---|---:|---:|---:|",
        f"| 真值 | {truth['Fake']} | {truth['Real']} | {truth['Uncertain']} |",
        f"| 训练目标（总体） | {target['Fake']} | {target['Real']} | {target['Uncertain']} |",
        "",
        "分项目标分布：整图 `Real 9 / Uncertain 8 / AI-generated 7`；",
        "局部 `Tampered 9 / Not tampered 5 / Uncertain 3 / Not applicable 7`。",
        "",
        "## 已通过、直接复用（3）",
        "",
        "| sample | 整图 / 局部 / 总体 | 真值 |",
        "|---|---|---|",
    ]
    for row in rows:
        if row["approved"]:
            lines.append(f"| `{row['sample_id']}` | {row['whole']} / {row['local']} / "
                         f"{row['overall']} | {row['truth']} |")
    lines += [
        "",
        "## 待确认（21）",
        "",
        "| # | sample | 类型 | 真值 | 整图 / 局部 / 总体 | calls | tokens | images |",
        "|---:|---|---|---|---|---:|---:|---:|",
    ]
    pending = [row for row in rows if not row["approved"]]
    for index, row in enumerate(pending, 1):
        lines.append(
            f"| {index} | `{row['sample_id']}` | {row['role']} | {row['truth']} | "
            f"{row['whole']} / {row['local']} / {row['overall']} | "
            f"{row['calls']} | {row['tokens']} | {row['images']} |")
    lines += [
        "",
        "## 建议排除的 2 条",
        "",
        "| sample | 原因 |",
        "|---|---|",
    ]
    for sid, reason in selection["excluded"].items():
        lines.append(f"| `{sid}` | {reason} |")
    lines += [
        "",
        "## Dry-run 摘要",
        "",
        f"- 24/24 通过；token 范围 {min(row['tokens'] for row in rows)}–"
        f"{max(row['tokens'] for row in rows)}；图片数范围 "
        f"{min(row['images'] for row in rows)}–{max(row['images'] for row in rows)}。",
        f"- **最长且图最多**：`{dry['longest']['sample_id']}`，"
        f"{dry['longest']['input_tokens']} tokens，{dry['longest']['total_pil_images']} 图。",
        f"- token 前 3：{', '.join(f'`{x}`' for x in dry['top3_tokens'])}。",
        f"- 图片数前 3（5 图并列，按 token 打破平局）："
        f"{', '.join(f'`{x}`' for x in dry['top3_images'])}。",
        f"- 10 步 profiling 改用："
        f"{', '.join(f'`{x}`' for x in dry['profiling_set'])}。",
        f"- **max_length 警告**：{len(dry['over_max_length'])} 条超过当前 10000："
        f"{', '.join(f'`{x}`' for x in dry['over_max_length'])}。"
        "正式 profiling/训练前需把 max_length 提到至少 12000，或先决定截断策略；"
        "本次未改训练参数。",
        "",
        "完整逐条 dry-run 见 `DRY_RUN.md`；原始 JSON 见 `dry_run.json`。",
        "",
        "## 确认后执行顺序",
        "",
        "1. 用上述 profiling 3 条跑 10 步，记录单步耗时、峰值显存；若 Midjourney 的",
        "   11677 token 在 max_length 12000 下 OOM，先报告。",
        "2. 从原版 Qwen3-VL 重新开始，24 条 × 2 epoch（48 步）正式训练并保存 adapter。",
        "3. 约 30 张与 22 个训练源不同源的验证图，同一自主调用流程对比原版 vs LoRA；",
        "   汇报局部检出、真图误报、两分项同时判对、弃权、调用次数和最大变化案例。",
        "4. 不扩 250 条；OOM 先报告。",
    ]
    (out / "TRAINING_CANDIDATES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    dry_lines = [
        "# 24 条 LoRA 小试 — Qwen3-VL dry-run",
        "",
        "使用 `scripts/train_lora_candidate_smoke.py --selection ... --dry-run`，",
        "与训练完全相同的数据组装、processor、assistant span 和 loss 掩码。",
        "",
        "| sample | tokens | images | images/tool turn | calls | supervised turns | supervised tokens |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (-item["tokens"], item["sample_id"])):
        d = dry_by_id[row["sample_id"]]
        dry_lines.append(
            f"| `{row['sample_id']}` | {d['input_tokens']} | {d['total_pil_images']} | "
            f"{d['images_per_tool_turn']} | {' → '.join(d['calls'])} | "
            f"{d['supervised_turns']} | {d['supervised_tokens']} |")
    dry_lines += [
        "",
        "## 极值",
        "",
        f"- 最长：`{dry['longest']['sample_id']}` "
        f"{dry['longest']['input_tokens']} tokens。",
        f"- 图最多：`{dry['most_images']['sample_id']}` "
        f"{dry['most_images']['total_pil_images']} 图（22 条双调用样本均为 5 图）。",
        f"- 10 步 profiling 集：{', '.join(dry['profiling_set'])}。",
        f"- 超过 max_length=10000：{dry['over_max_length'] or '无'}。",
        "",
        "## 边界",
        "",
        "- dry-run 只加载 processor，不加载模型、不使用 GPU。",
        "- 每条的原图在首 user turn；TruFor 三图在 TruFor 调用后的 tool turn；",
        "  PROBE 裁剪在 PROBE 调用后的 turn；工具结果/系统状态 labels=-100。",
    ]
    (out / "DRY_RUN.md").write_text("\n".join(dry_lines) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "truth": truth, "target": target,
                      "longest": dry["longest"]["sample_id"],
                      "most_images": dry["most_images"]["sample_id"],
                      "profiling_set": dry["profiling_set"],
                      "over_max_length": dry["over_max_length"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
