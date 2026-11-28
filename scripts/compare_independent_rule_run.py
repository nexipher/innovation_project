"""Compare the 2026-11-28 independent-local rule run with validation_fixed base."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
DEFAULT_OLD = SMALL / "validation_fixed/base/records.json"
DEFAULT_NEW = SMALL / "validation_independent_20261128/base/records.json"
DEFAULT_MANIFEST = SMALL / "validation_manifest_independent_20261128.json"
DEFAULT_OUT = SMALL / "analysis/rule_update_20261128"


def _records(path: Path) -> dict:
    return {row["sample_id"]: row for row in json.loads(
        path.read_text(encoding="utf-8"))}


def _truth(row: dict, key: str):
    return (row.get("task_truth") or {}).get(key)


def _consistency(overall, whole, local, allow_na: bool = False) -> bool:
    if local not in ("Tampered", "Not tampered", "Uncertain"):
        if not (allow_na and local == "Not applicable"):
            return False
    fake_dimension = whole == "AI-generated" or local == "Tampered"
    if overall == "Real" and fake_dimension:
        return False
    if overall == "Fake" and not fake_dimension:
        return False
    if overall == "Uncertain" and fake_dimension:
        return False
    return True


def _group(row: dict) -> str:
    return row.get("local_truth_group") or (
        "forged_positive" if _truth(row, "local_manipulation") == "Tampered"
        else "real_negative")


def _local_correct(row: dict, label: str) -> bool:
    if _group(row) == "forged_positive":
        return label == "Tampered"
    return label == "Not tampered"


def _local_fp(row: dict, label: str) -> bool:
    return _group(row) != "forged_positive" and label == "Tampered"


def _overall_truth(row: dict) -> str:
    return "Fake" if row.get("ground_truth") == "Fake" else "Real"


def _stats(rows: list[dict], arm: str) -> dict:
    forged = [row for row in rows if _group(row) == "forged_positive"]
    real = [row for row in rows if _group(row) == "real_negative"]
    generated = [row for row in rows if _group(row) == "generated_negative"]
    def label(row):
        return row[arm]["local_manipulation"]
    def overall(row):
        return row[arm]["final_verdict"]
    return {
        "local_tampered_detected": sum(label(row) == "Tampered" for row in forged),
        "local_tampered_n": len(forged),
        "real_local_false_positive": sum(_local_fp(row, label(row)) for row in real),
        "real_local_false_positive_n": len(real),
        "generated_local_false_positive": sum(
            _local_fp(row, label(row)) for row in generated),
        "generated_local_false_positive_n": len(generated),
        "real_local_uncertain": sum(label(row) == "Uncertain" for row in real),
        "generated_local_uncertain": sum(label(row) == "Uncertain"
                                          for row in generated),
        "local_abstained": sum(label(row) == "Uncertain" for row in rows),
        "local_abstained_n": len(rows),
        "overall_abstained": sum(overall(row) == "Uncertain" for row in rows),
        "overall_abstained_n": len(rows),
        "consistent": sum(
            _consistency(overall(row), row[arm]["whole_image_generation"],
                         label(row), allow_na=(arm == "old")) for row in rows),
        "consistent_n": len(rows),
        "overall_correct": sum(overall(row) == _overall_truth(row) for row in rows),
        "overall_correct_n": len(rows),
        "calls_total": sum(int(row[arm].get("expert_calls") or 0) for row in rows),
        "calls_mean": round(sum(int(row[arm].get("expert_calls") or 0)
                                for row in rows) / len(rows), 3),
        "local_distribution": dict(Counter(label(row) for row in rows)),
        "overall_distribution": dict(Counter(overall(row) for row in rows)),
        "coverage_failed": sum((row[arm].get("coverage_check") or {}).get("status")
                               == "failed" for row in rows),
        "coverage_pass": sum((row[arm].get("coverage_check") or {}).get("status")
                             in ("pass", "pass_after_correction") for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default=str(DEFAULT_OLD))
    parser.add_argument("--new", default=str(DEFAULT_NEW))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    old = _records(Path(args.old))
    new = _records(Path(args.new))
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    truth = {row["sample_id"]: row for row in manifest["samples"]}
    ids = sorted(truth)
    rows = []
    for sid in ids:
        base = {**truth[sid], **old[sid],
                "task_truth": truth[sid]["task_truth"],
                "local_truth_group": truth[sid].get("local_truth_group"),
                "local_truth_old": truth[sid].get("local_truth_old")}
        old_row = old[sid]
        new_row = new[sid]
        row = {
            "sample_id": sid,
            "group": _group(base),
            "ground_truth": base["ground_truth"],
            "truth_whole": _truth(base, "whole_image_generation"),
            "truth_local": _truth(base, "local_manipulation"),
            "truth_local_old": base["local_truth_old"],
            "old_whole": old_row.get("whole_image_generation"),
            "old_local": old_row.get("local_manipulation"),
            "old_overall": old_row.get("final_verdict"),
            "old_calls": old_row.get("expert_calls"),
            "old_coverage": (old_row.get("coverage_check") or {}).get("status"),
            "new_whole": new_row.get("whole_image_generation"),
            "new_local": new_row.get("local_manipulation"),
            "new_overall": new_row.get("final_verdict"),
            "new_calls": new_row.get("expert_calls"),
            "new_coverage": (new_row.get("coverage_check") or {}).get("status"),
            "new_local_correct": _local_correct(base, new_row.get("local_manipulation") or ""),
            "new_local_fp": _local_fp(base, new_row.get("local_manipulation") or ""),
            "new_consistent": _consistency(
                new_row.get("final_verdict"), new_row.get("whole_image_generation"),
                new_row.get("local_manipulation")),
            "new_overall_correct": new_row.get("final_verdict") == _overall_truth(base),
        }
        rows.append(row)
    old_stats = _stats([{**truth[sid], "old": old[sid],
                         "task_truth": truth[sid]["task_truth"],
                         "local_truth_group": truth[sid].get("local_truth_group"),
                         "local_truth_old": truth[sid].get("local_truth_old")}
                        for sid in ids], "old")
    new_stats = _stats([{**truth[sid], "new": new[sid],
                         "task_truth": truth[sid]["task_truth"],
                         "local_truth_group": truth[sid].get("local_truth_group"),
                         "local_truth_old": truth[sid].get("local_truth_old")}
                        for sid in ids], "new")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "independent_rule_comparison/1",
               "old": str(args.old), "new": str(args.new),
               "old_stats": old_stats, "new_stats": new_stats,
               "per_sample": rows}
    (out / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    focus_ids = ["val__autosplice__40489__2__q95",
                 "val__autosplice__60788__0__q95",
                 "val__autosplice__57143__0__q95",
                 "val__autosplice__41993__0__q95",
                 "val__autosplice__67065__0__q95",
                 "val__autosplice__42958__1__q95"]
    focus = [row for row in rows if row["sample_id"] in focus_ids]
    lines = [
        "# 联合取证规则更新复验：历史口径 vs 独立局部口径",
        "",
        "新规则：整图生成与局部编辑独立；局部只输出 Tampered / Not tampered / "
        "Uncertain，取消 Not applicable；whole=AI-generated 时总体必须 Fake。"
        "历史 `validation_fixed/base` 标签未改，仅在对照中作为旧口径。",
        "",
        "## 总表",
        "",
        "| 指标 | 旧 base | 新 base | Δ |",
        "|---|---:|---:|---:|",
    ]
    def line(key, label, fmt="{}"):
        a, b = old_stats[key], new_stats[key]
        delta = b - a
        return f"| {label} | {fmt.format(a)} | {fmt.format(b)} | {delta:+} |"
    for key, label in (
        ("local_tampered_detected", "局部篡改检出（10 伪造）"),
        ("real_local_false_positive", "真图局部误报（10 真图）"),
        ("generated_local_false_positive", "生成图局部误报（10 构造确认负例）"),
        ("real_local_uncertain", "真图局部 Uncertain"),
        ("generated_local_uncertain", "生成图局部 Uncertain"),
        ("local_abstained", "局部 Uncertain 总数（30）"),
        ("overall_abstained", "总体弃权（30）"),
        ("consistent", "分项/总体结案一致（30）"),
        ("overall_correct", "总体与真值一致（30）"),
        ("calls_total", "专家调用总数"),
    ):
        lines.append(line(key, label))
    lines += [
        "",
        "## 重点案例",
        "",
        "| sample | truth | 旧 whole/local/overall | 新 whole/local/overall | 新一致性 |",
        "|---|---|---|---|---|",
    ]
    for row in focus:
        lines.append(
            f"| `{row['sample_id']}` | {row['truth_whole']}/{row['truth_local']} | "
            f"{row['old_whole']}/{row['old_local']}/{row['old_overall']} | "
            f"{row['new_whole']}/{row['new_local']}/{row['new_overall']} | "
            f"{row['new_consistent']} |")
    lines += [
        "",
        "## 读法",
        "",
        "- 新口径把生成图也视为可能被后续局部篡改；旧 `Not applicable` 仅作为历史记录保留，"
        "不计入新规则的局部负例。",
        "- 生成图局部负例的构造依据见 `truth_provenance.csv`：GenImage manifest 记录 "
        "generator-only q95 派生，没有局部编辑步骤或 mask。",
        "- 旧 base 的 `Not applicable` 在对照表中原样展示，不重写历史文件。",
        "",
        "## 边界",
        "",
        "本轮只改提示/结案检查和复验；裁剪、专家、Qwen 配置固定；不重训、不扩样、不碰 test。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"old": old_stats, "new": new_stats,
                      "report": str(out / "report.md")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
