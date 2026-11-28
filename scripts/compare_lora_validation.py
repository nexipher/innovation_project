"""Compare base Qwen3-VL vs LoRA on the 30-image validation set."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
DEFAULT_OUT = SMALL / "validation"


def _load(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"]: row for row in rows}


def _truth(row: dict, key: str):
    return (row.get("task_truth") or {}).get(key)


def _both_correct(row: dict) -> bool:
    return (row.get("whole_image_generation") == _truth(row, "whole_image_generation")
            and row.get("local_manipulation") == _truth(row, "local_manipulation"))


def _rate(n: int, d: int):
    return round(n / d, 4) if d else None


def _group(row: dict) -> str:
    local = _truth(row, "local_manipulation")
    whole = _truth(row, "whole_image_generation")
    if local == "Tampered":
        return "local_tampered"
    if whole == "AI-generated":
        return "whole_generated"
    return "real"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    base = _load(out / "base/records.json")
    lora = _load(out / "lora/records.json")
    base_end = json.loads((out / "base/run_end.json").read_text(encoding="utf-8"))
    lora_end = json.loads((out / "lora/run_end.json").read_text(encoding="utf-8"))
    ids = sorted(set(base) | set(lora))
    missing = {"base": [sid for sid in ids if sid not in base],
               "lora": [sid for sid in ids if sid not in lora]}
    if missing["base"] or missing["lora"]:
        raise SystemExit(f"incomplete arms: {missing}")
    rows = []
    for sid in ids:
        b, r = base[sid], lora[sid]
        row = {
            "sample_id": sid,
            "group": _group(b),
            "ground_truth": b["ground_truth"],
            "truth_whole": _truth(b, "whole_image_generation"),
            "truth_local": _truth(b, "local_manipulation"),
            "base_whole": b.get("whole_image_generation"),
            "base_local": b.get("local_manipulation"),
            "base_overall": b.get("final_verdict"),
            "base_certainty": b.get("certainty"),
            "base_calls": b.get("expert_calls"),
            "lora_whole": r.get("whole_image_generation"),
            "lora_local": r.get("local_manipulation"),
            "lora_overall": r.get("final_verdict"),
            "lora_certainty": r.get("certainty"),
            "lora_calls": r.get("expert_calls"),
            "base_both": _both_correct(b),
            "lora_both": _both_correct(r),
            "changed_overall": b.get("final_verdict") != r.get("final_verdict"),
            "changed_local": b.get("local_manipulation") != r.get("local_manipulation"),
            "changed_whole": b.get("whole_image_generation") != r.get("whole_image_generation"),
            "base_elapsed_s": b.get("elapsed_s"),
            "lora_elapsed_s": r.get("elapsed_s"),
        }
        rows.append(row)

    def summary(arm: str):
        def get(row, key):
            return row[f"{arm}_{key}"]
        tampered = [row for row in rows if row["group"] == "local_tampered"]
        real = [row for row in rows if row["group"] == "real"]
        generated = [row for row in rows if row["group"] == "whole_generated"]
        return {
            "local_tampered_detected": sum(1 for row in tampered
                                           if get(row, "local") == "Tampered"),
            "local_tampered_n": len(tampered),
            "real_local_false_positive": sum(1 for row in real
                                              if get(row, "local") == "Tampered"),
            "real_local_false_positive_n": len(real),
            "real_whole_false_positive": sum(1 for row in real
                                              if get(row, "whole") == "AI-generated"),
            "real_overall_false_positive": sum(1 for row in real
                                                if get(row, "overall") == "Fake"),
            "both_correct": sum(1 for row in rows if get(row, "both")),
            "both_correct_n": len(rows),
            "both_correct_tampered": sum(1 for row in tampered if get(row, "both")),
            "both_correct_generated": sum(1 for row in generated if get(row, "both")),
            "both_correct_real": sum(1 for row in real if get(row, "both")),
            "abstained": sum(1 for row in rows if get(row, "overall") == "Uncertain"),
            "abstained_n": len(rows),
            "calls_total": sum(int(get(row, "calls") or 0) for row in rows),
            "calls_mean": round(sum(int(get(row, "calls") or 0) for row in rows)
                                / len(rows), 3),
            "call_distribution": dict(Counter(int(get(row, "calls") or 0)
                                              for row in rows)),
            "local_distribution": dict(Counter(get(row, "local") for row in rows)),
            "whole_distribution": dict(Counter(get(row, "whole") for row in rows)),
            "overall_distribution": dict(Counter(get(row, "overall") for row in rows)),
            "elapsed_sum_s": round(sum(float(row[f"{arm}_elapsed_s"] or 0)
                                       for row in rows), 1),
        }

    base_stats, lora_stats = summary("base"), summary("lora")
    changed = [row for row in rows if (row["changed_local"] or row["changed_whole"]
                                       or row["changed_overall"]
                                       or row["base_calls"] != row["lora_calls"])]
    changed.sort(key=lambda row: (
        not (row["base_both"] and not row["lora_both"]),
        row["lora_both"] == row["base_both"],
        row["sample_id"]))
    report = {
        "schema": "lora_validation_comparison/1",
        "samples": len(rows),
        "base": base_stats,
        "lora": lora_stats,
        "base_run": base_end,
        "lora_run": lora_end,
        "changed": len(changed),
        "changed_cases": changed,
        "per_sample": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = list(rows[0])
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# 30 图验证：原版 Qwen3-VL vs LoRA",
        "",
        f"- 样本：{len(rows)}；local_tampered {base_stats['local_tampered_n']}、"
        f"真图 {base_stats['real_local_false_positive_n']}、整图生成 "
        f"{base_stats['both_correct_generated']} 条 both-correct（共 "
        f"{sum(1 for row in rows if row['group'] == 'whole_generated')}）。",
        f"- 原版：局部检出 {base_stats['local_tampered_detected']}/"
        f"{base_stats['local_tampered_n']}；真图局部误报 "
        f"{base_stats['real_local_false_positive']}/"
        f"{base_stats['real_local_false_positive_n']}；两分项同时判对 "
        f"{base_stats['both_correct']}/{base_stats['both_correct_n']}；弃权 "
        f"{base_stats['abstained']}/{base_stats['abstained_n']}；调用 "
        f"{base_stats['calls_total']}（均 {base_stats['calls_mean']}）。",
        f"- LoRA：局部检出 {lora_stats['local_tampered_detected']}/"
        f"{lora_stats['local_tampered_n']}；真图局部误报 "
        f"{lora_stats['real_local_false_positive']}/"
        f"{lora_stats['real_local_false_positive_n']}；两分项同时判对 "
        f"{lora_stats['both_correct']}/{lora_stats['both_correct_n']}；弃权 "
        f"{lora_stats['abstained']}/{lora_stats['abstained_n']}；调用 "
        f"{lora_stats['calls_total']}（均 {lora_stats['calls_mean']}）。",
        f"- 运行：base {base_end.get('elapsed_s')} s / peak "
        f"{base_end.get('peak_allocated_gb')} GB；LoRA {lora_end.get('elapsed_s')} s / peak "
        f"{lora_end.get('peak_allocated_gb')} GB。",
        "",
        "## 分项",
        "",
        "| 指标 | 原版 | LoRA |",
        "|---|---:|---:|",
        f"| 局部篡改检出 | {base_stats['local_tampered_detected']}/{base_stats['local_tampered_n']} | "
        f"{lora_stats['local_tampered_detected']}/{lora_stats['local_tampered_n']} |",
        f"| 真图局部误报 | {base_stats['real_local_false_positive']}/{base_stats['real_local_false_positive_n']} | "
        f"{lora_stats['real_local_false_positive']}/{lora_stats['real_local_false_positive_n']} |",
        f"| 真图整图误报（AI-generated） | {base_stats['real_whole_false_positive']} | "
        f"{lora_stats['real_whole_false_positive']} |",
        f"| 真图总体误报（Fake） | {base_stats['real_overall_false_positive']} | "
        f"{lora_stats['real_overall_false_positive']} |",
        f"| 两分项同时判对（30） | {base_stats['both_correct']} | {lora_stats['both_correct']} |",
        f"| 其中篡改图（10） | {base_stats['both_correct_tampered']} | {lora_stats['both_correct_tampered']} |",
        f"| 其中生成图（10） | {base_stats['both_correct_generated']} | {lora_stats['both_correct_generated']} |",
        f"| 其中真图（10） | {base_stats['both_correct_real']} | {lora_stats['both_correct_real']} |",
        f"| 弃权 | {base_stats['abstained']} | {lora_stats['abstained']} |",
        f"| 专家调用总数 | {base_stats['calls_total']} | {lora_stats['calls_total']} |",
        f"| 平均每会话调用 | {base_stats['calls_mean']} | {lora_stats['calls_mean']} |",
        "",
        "## 调用分布",
        "",
        f"- 原版：`{base_stats['call_distribution']}`",
        f"- LoRA：`{lora_stats['call_distribution']}`",
        "",
        "## 变化最大的案例",
        "",
        "| sample | group | 原版 whole/local/overall | LoRA whole/local/overall | 原版/ LoRA both | calls |",
        "|---|---|---|---|---|---|",
    ]
    for row in changed[:12]:
        lines.append(
            f"| `{row['sample_id']}` | {row['group']} | "
            f"{row['base_whole']} / {row['base_local']} / {row['base_overall']} | "
            f"{row['lora_whole']} / {row['lora_local']} / {row['lora_overall']} | "
            f"{row['base_both']} / {row['lora_both']} | "
            f"{row['base_calls']} / {row['lora_calls']} |")
    lines += [
        "",
        "## 边界",
        "",
        "- 同一 30 图、同一自主流程、同一 PROBE/TruFor；仅 Qwen 是否加载 LoRA 不同。",
        "- 两分项同时判对 = 整图与局部都与任务真值一致；不是总体准确率。",
        "- 真图误报按 truth 为 Real 的 10 条（5 AutoSplice 真图 + 5 Real）。",
        "- 生成图的局部真值为 Not applicable；本表不把局部响应算作误报。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "base": base_stats, "lora": lora_stats,
                      "changed": len(changed),
                      "report": str(out / "report.md")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
