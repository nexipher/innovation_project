"""Compare the 30-image validation before and after the 2026-11-28 fixes."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"


def _records(path: Path) -> dict:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {row["sample_id"]: row for row in rows}


def _trace_verdict(record: dict) -> dict:
    path = ROOT / record["trace_path"]
    if not path.exists():
        return {}
    trace = json.loads(path.read_text(encoding="utf-8"))
    return trace.get("final_verdict") or {}


def _truth(record: dict, key: str):
    return (record.get("task_truth") or {}).get(key)


def _group(record: dict) -> str:
    if _truth(record, "local_manipulation") == "Tampered":
        return "local_tampered"
    if _truth(record, "whole_image_generation") == "AI-generated":
        return "whole_generated"
    return "real"


def _consistent(overall, whole, local) -> bool:
    fake_dimension = whole == "AI-generated" or local == "Tampered"
    if overall == "Uncertain" and fake_dimension:
        return False
    if overall == "Real" and fake_dimension:
        return False
    if overall == "Fake" and not fake_dimension:
        return False
    return True


def _panel_source(record: dict):
    path = ROOT / record["trace_path"]
    if not path.exists():
        return None
    trace = json.loads(path.read_text(encoding="utf-8"))
    layouts = []
    for token in trace.get("evidence_chain") or []:
        if token.get("source") != "trufor_expert":
            continue
        layout = token.get("model_image_layout") or {}
        sources = {
            value.get("region_source") or value.get("inspection_region_source")
            for value in layout.values() if isinstance(value, dict)}
        layouts.extend(sorted(source for source in sources if source))
    return ",".join(sorted(set(layouts))) or None


def _stats(records: dict, arm: str) -> dict:
    rows = []
    for record in records.values():
        trace = _trace_verdict(record)
        coverage = trace.get("coverage_check") or {}
        whole = record.get("whole_image_generation")
        local = record.get("local_manipulation")
        overall = record.get("final_verdict")
        rows.append({
            **record,
            "group": _group(record),
            "consistent": _consistent(overall, whole, local),
            "generated_local_na": local == "Not applicable",
            "coverage_status": coverage.get("status"),
            "remaining_coverage": [
                issue.get("code")
                for issue in coverage.get("remaining_issues") or []
                if isinstance(issue, dict) and issue.get("code")],
            "panel_source": _panel_source(record),
        })
    tampered = [row for row in rows if row["group"] == "local_tampered"]
    real = [row for row in rows if row["group"] == "real"]
    generated = [row for row in rows if row["group"] == "whole_generated"]
    return {
        "samples": len(rows),
        "local_tampered_detected": sum(row["local_manipulation"] == "Tampered"
                                       for row in tampered),
        "local_tampered_n": len(tampered),
        "real_local_false_positive": sum(row["local_manipulation"] == "Tampered"
                                          for row in real),
        "real_local_false_positive_n": len(real),
        "real_whole_false_positive": sum(row["whole_image_generation"] == "AI-generated"
                                          for row in real),
        "real_overall_false_positive": sum(row["final_verdict"] == "Fake"
                                            for row in real),
        "both_correct": sum(
            row["whole_image_generation"] == _truth(row, "whole_image_generation")
            and row["local_manipulation"] == _truth(row, "local_manipulation")
            for row in rows),
        "both_correct_n": len(rows),
        "both_correct_tampered": sum(
            row["whole_image_generation"] == _truth(row, "whole_image_generation")
            and row["local_manipulation"] == _truth(row, "local_manipulation")
            for row in tampered),
        "both_correct_generated": sum(
            row["whole_image_generation"] == _truth(row, "whole_image_generation")
            and row["local_manipulation"] == _truth(row, "local_manipulation")
            for row in generated),
        "both_correct_real": sum(
            row["whole_image_generation"] == _truth(row, "whole_image_generation")
            and row["local_manipulation"] == _truth(row, "local_manipulation")
            for row in real),
        "abstained": sum(row["final_verdict"] == "Uncertain" for row in rows),
        "calls_total": sum(int(row.get("expert_calls") or 0) for row in rows),
        "calls_mean": round(sum(int(row.get("expert_calls") or 0)
                                for row in rows) / len(rows), 3),
        "consistent": sum(row["consistent"] for row in rows),
        "generated_local_na": sum(row["generated_local_na"] for row in generated),
        "coverage_failed": sum(row["coverage_status"] == "failed" for row in rows),
        "coverage_pass": sum(row["coverage_status"] in ("pass", "pass_after_correction")
                             for row in rows),
        "panel_sources": dict(Counter(row["panel_source"] for row in rows)),
        "local_distribution": dict(Counter(row["local_manipulation"] for row in rows)),
        "whole_distribution": dict(Counter(row["whole_image_generation"] for row in rows)),
        "overall_distribution": dict(Counter(row["final_verdict"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default=str(SMALL / "validation"))
    parser.add_argument("--after", default=str(SMALL / "validation_fixed"))
    parser.add_argument("--out", default=str(SMALL / "analysis/fix_comparison"))
    args = parser.parse_args()
    before_dir, after_dir = Path(args.before), Path(args.after)
    before = {arm: _records(before_dir / arm / "records.json")
              for arm in ("base", "lora")}
    after = {arm: _records(after_dir / arm / "records.json")
             for arm in ("base", "lora")}
    ids = sorted(set(before["base"]) | set(after["base"]))
    before_missing = [sid for sid in ids if sid not in before["base"]
                      or sid not in before["lora"]]
    after_missing = [sid for sid in ids if sid not in after["base"]
                     or sid not in after["lora"]]
    if before_missing or after_missing:
        raise SystemExit(f"incomplete arms: before={before_missing} "
                         f"after={after_missing}")
    stats = {}
    rows = []
    for arm in ("base", "lora"):
        stats[arm] = {
            "before": _stats(before[arm], arm),
            "after": _stats(after[arm], arm),
        }
    for sid in ids:
        b = before["base"][sid]
        row = {
            "sample_id": sid,
            "group": _group(b),
            "truth_whole": _truth(b, "whole_image_generation"),
            "truth_local": _truth(b, "local_manipulation"),
        }
        for arm in ("base", "lora"):
            for side, source in (("before", before), ("after", after)):
                record = source[arm][sid]
                trace = _trace_verdict(record)
                coverage = trace.get("coverage_check") or {}
                prefix = f"{arm}_{side}"
                row[f"{prefix}_whole"] = record.get("whole_image_generation")
                row[f"{prefix}_local"] = record.get("local_manipulation")
                row[f"{prefix}_overall"] = record.get("final_verdict")
                row[f"{prefix}_calls"] = record.get("expert_calls")
                row[f"{prefix}_coverage"] = coverage.get("status")
                row[f"{prefix}_consistent"] = _consistent(
                    record.get("final_verdict"),
                    record.get("whole_image_generation"),
                    record.get("local_manipulation"))
                row[f"{prefix}_both"] = (
                    record.get("whole_image_generation") == row["truth_whole"]
                    and record.get("local_manipulation") == row["truth_local"])
                row[f"{prefix}_panel"] = _panel_source(record)
        rows.append(row)

    focus = [row for row in rows if row["group"] == "local_tampered"
             or row["sample_id"].endswith("d3429f29-5d3a-4da3-829a-0fb387554a98__q95")
             or any(key in row["sample_id"] for key in
                    ("BigGAN__739", "VQDM__VQDM_1000_200_01_143"))]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "lora_fix_comparison/1", "before": str(before_dir),
               "after": str(after_dir), "stats": stats, "per_sample": rows,
               "focus": focus}
    (out / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    def line(arm, key, label):
        b = stats[arm]["before"][key]
        a = stats[arm]["after"][key]
        return f"| {label} | {b} | {a} | {a - b:+d} |" if isinstance(a, int) else \
               f"| {label} | {b} | {a} | {a - b:+.3f} |"
    lines = [
        "# 30 图修复前后对照",
        "",
        "修复：无 bbox 的 TruFor 三图改为 high-response 包围盒；覆盖纠正要求未测 TruFor/PROBE "
        "先补测，禁止两位专家已测时重复调用；局部仅候选时明确整图 AI-generated "
        "仍保持总体 Fake；整图 AI-generated 且 TruFor 低分时局部改 Not applicable。",
        "",
        "## 总表",
        "",
        "| 指标 | 原版 before | 原版 after | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("local_tampered_detected", "局部篡改检出（/10）"),
        ("real_local_false_positive", "真图局部误报（/10）"),
        ("real_whole_false_positive", "真图整图误报（/10）"),
        ("real_overall_false_positive", "真图总体误报（/10）"),
        ("both_correct", "两分项同时判对（/30）"),
        ("both_correct_tampered", "其中篡改图（/10）"),
        ("both_correct_generated", "其中生成图（/10）"),
        ("both_correct_real", "其中真图（/10）"),
        ("abstained", "弃权（/30）"),
        ("calls_total", "专家调用总数"),
        ("consistent", "分项/总体一致（/30）"),
        ("generated_local_na", "生成图 local Not applicable（/10）"),
        ("coverage_failed", "覆盖检查 failed（/30）"),
    ):
        lines.append(line("base", key, label))
    lines += [
        "",
        "| 指标 | LoRA before | LoRA after | Δ |",
        "|---|---:|---:|---:|",
    ]
    for key, label in (
        ("local_tampered_detected", "局部篡改检出（/10）"),
        ("real_local_false_positive", "真图局部误报（/10）"),
        ("real_whole_false_positive", "真图整图误报（/10）"),
        ("real_overall_false_positive", "真图总体误报（/10）"),
        ("both_correct", "两分项同时判对（/30）"),
        ("both_correct_tampered", "其中篡改图（/10）"),
        ("both_correct_generated", "其中生成图（/10）"),
        ("both_correct_real", "其中真图（/10）"),
        ("abstained", "弃权（/30）"),
        ("calls_total", "专家调用总数"),
        ("consistent", "分项/总体一致（/30）"),
        ("generated_local_na", "生成图 local Not applicable（/10）"),
        ("coverage_failed", "覆盖检查 failed（/30）"),
    ):
        lines.append(line("lora", key, label))
    lines += [
        "",
        "## 重点案例",
        "",
        "| sample | truth whole/local | base before | base after | LoRA before | LoRA after |",
        "|---|---|---|---|---|---|",
    ]
    for row in focus:
        lines.append(
            f"| `{row['sample_id']}` | {row['truth_whole']}/{row['truth_local']} | "
            f"{row['base_before_whole']}/{row['base_before_local']}/{row['base_before_overall']} | "
            f"{row['base_after_whole']}/{row['base_after_local']}/{row['base_after_overall']} | "
            f"{row['lora_before_whole']}/{row['lora_before_local']}/{row['lora_before_overall']} | "
            f"{row['lora_after_whole']}/{row['lora_after_local']}/{row['lora_after_overall']} |")
    lines += [
        "",
        "## 边界",
        "",
        "- 同一 30 图、同一 greedy/non-thinking 设置；修复只改证据呈现和覆盖纠正，"
        "未重训、未改专家权重/阈值。",
        "- `consistent` 只检查总体与分项的标签关系（Fake 分项是否存在），不等同于真值正确。",
        "- 覆盖检查失败是记录，不会静默改写模型标签。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stats": stats, "focus": len(focus),
                      "report": str(out / "report.md")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
