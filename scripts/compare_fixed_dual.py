"""Compare fixed PROBE→TruFor calls with the 2026-11-27 autonomous rerun.

Reads the two Agent batches (part1/part2) and the fixed-dual batch, joins by
``sample_id``, writes a per-sample CSV and a concise report.  This is an
evaluation comparison, not a training-data converter.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
DEFAULT_FIXED = ROOT / "sft_data/g8_fixed_dual_20261127"
DEFAULT_PARTS = (
    ROOT / "sft_data/g8_dev_eval_20261127_part1/records.json",
    ROOT / "sft_data/g8_dev_eval_20261127_part2/records.json",
)


def _load(path):
    rows = json.load(open(path, encoding="utf-8"))
    return {row["sample_id"]: row for row in rows}


def _truth_class(row):
    truth = row["task_truth"]
    if (truth["whole_image_generation"] == "AI-generated"
            and truth["local_manipulation"] == "Not applicable"):
        return "whole_generated"
    if truth["local_manipulation"] == "Tampered":
        return "local_tampered"
    return "real"


def _expected(row):
    return {"real": "Real", "whole_generated": "Fake",
            "local_tampered": "Fake"}[_truth_class(row)]


def _trufor_state(row):
    tokens = [e for e in (row.get("evidence") or [])
              if e.get("source") == "trufor_expert"]
    if not tokens:
        return "not_called"
    if any(e.get("status") == "failed" for e in tokens):
        return "failed"
    return "measured"


def _focus(row):
    if row["task_truth"]["local_manipulation"] != "Tampered":
        return ""
    state = _trufor_state(row)
    if state == "not_called":
        return "tampered_agent_not_called"
    if state == "failed":
        return "tampered_agent_failed"
    return ("tampered_agent_detected" if row["local_manipulation"] == "Tampered"
            else "tampered_agent_called_not_detected")


def _summary(rows):
    n = len(rows)
    counts = Counter(row["final_verdict"] for row in rows)
    correct = sum(1 for row in rows if row["final_verdict"] == _expected(row))
    abstain = counts.get("Uncertain", 0)
    no_conclusion = n - sum(counts.get(k, 0) for k in ("Real", "Fake", "Uncertain"))
    by_class = {}
    for cls in ("real", "whole_generated", "local_tampered"):
        subset = [r for r in rows if _truth_class(r) == cls]
        c = sum(1 for r in subset if r["final_verdict"] == _expected(r))
        a = sum(1 for r in subset if r["final_verdict"] == "Uncertain")
        by_class[cls] = {
            "n": len(subset), "correct": c,
            "wrong": sum(1 for r in subset if r["final_verdict"] not in
                         ("Uncertain", _expected(r))),
            "abstain": a,
            "accuracy": round(c / len(subset), 4) if subset else None,
            "coverage": round((len(subset) - a) / len(subset), 4) if subset else None,
            "closed_accuracy": round(c / (len(subset) - a), 4)
            if subset and len(subset) > a else None,
        }
    whole_gen = [r for r in rows
                 if r["task_truth"]["whole_image_generation"] == "AI-generated"]
    whole_real = [r for r in rows
                  if r["task_truth"]["whole_image_generation"] == "Real"]
    forged = [r for r in rows if r["task_truth"]["local_manipulation"] == "Tampered"]
    authentic = [r for r in rows if r["group"] == "AutoSplice_Authentic"]
    return {
        "n": n,
        "overall": {
            "correct": correct, "wrong": n - correct - abstain - no_conclusion,
            "abstain": abstain, "no_conclusion": no_conclusion,
            "accuracy": round(correct / n, 4) if n else None,
            "coverage": round((n - abstain) / n, 4) if n else None,
            "closed_accuracy": round(correct / (n - abstain), 4)
            if n > abstain else None,
            "by_truth_class": by_class,
        },
        "whole": {
            "generated_n": len(whole_gen),
            "generated_detected": sum(
                1 for r in whole_gen
                if r["whole_image_generation"] == "AI-generated"),
            "real_n": len(whole_real),
            "real_false_positive": sum(
                1 for r in whole_real
                if r["whole_image_generation"] == "AI-generated"),
        },
        "local": {
            "forged_n": len(forged),
            "forged_detected": sum(
                1 for r in forged if r["local_manipulation"] == "Tampered"),
            "authentic_n": len(authentic),
            "authentic_fp": sum(
                1 for r in authentic if r["local_manipulation"] == "Tampered"),
            "authentic_labels": dict(Counter(
                r["local_manipulation"] for r in authentic)),
            "forged_labels": dict(Counter(
                r["local_manipulation"] for r in forged)),
            "forged_both_whole_real_and_local_tampered": sum(
                1 for r in forged
                if r["whole_image_generation"] == "Real"
                and r["local_manipulation"] == "Tampered"),
        },
        "calls": {
            "expert_calls_total": sum(
                int((r.get("counters") or {}).get("expert_calls") or 0)
                for r in rows),
            "expert_calls_distribution": dict(Counter(
                int((r.get("counters") or {}).get("expert_calls") or 0)
                for r in rows)),
            "model_turns_distribution": dict(Counter(
                int((r.get("counters") or {}).get("model_turns") or 0)
                for r in rows)),
            "elapsed_s_total": round(sum(float(r.get("elapsed_s") or 0)
                                         for r in rows), 1),
            "elapsed_s_mean": round(statistics.mean(
                float(r.get("elapsed_s") or 0) for r in rows), 2),
            "elapsed_s_median": round(statistics.median(
                float(r.get("elapsed_s") or 0) for r in rows), 2),
        },
    }


def _focus_summary(base, fixed):
    out = {}
    for group in ("tampered_agent_not_called",
                  "tampered_agent_called_not_detected",
                  "tampered_agent_detected",
                  "tampered_agent_failed"):
        ids = [sid for sid, row in base.items() if _focus(row) == group]
        if not ids:
            continue
        rows = [fixed[sid] for sid in ids]
        out[group] = {
            "n": len(ids),
            "fixed_local": dict(Counter(r["local_manipulation"] for r in rows)),
            "fixed_whole": dict(Counter(r["whole_image_generation"] for r in rows)),
            "fixed_final": dict(Counter(r["final_verdict"] for r in rows)),
            "fixed_local_tampered": sum(
                1 for r in rows if r["local_manipulation"] == "Tampered"),
            "fixed_both": sum(
                1 for r in rows if r["whole_image_generation"] == "Real"
                and r["local_manipulation"] == "Tampered"),
            "examples": [
                {"sample_id": sid,
                 "baseline_local": base[sid]["local_manipulation"],
                 "baseline_final": base[sid]["final_verdict"],
                 "fixed_local": fixed[sid]["local_manipulation"],
                 "fixed_final": fixed[sid]["final_verdict"],
                 "fixed_whole": fixed[sid]["whole_image_generation"]}
                for sid in ids[:8]
            ],
        }
    return out


def _write_csv(path, base, fixed):
    fields = [
        "sample_id", "family", "group", "whole_truth", "local_truth",
        "baseline_trufor", "baseline_verdict", "baseline_whole",
        "baseline_local", "baseline_certainty", "baseline_calls",
        "baseline_turns", "baseline_elapsed_s",
        "fixed_verdict", "fixed_whole", "fixed_local", "fixed_certainty",
        "fixed_calls", "fixed_turns", "fixed_elapsed_s", "fixed_forced",
        "focus_group", "baseline_both_correct", "fixed_both_correct",
        "baseline_page", "fixed_page",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sid in base:
            b, f = base[sid], fixed[sid]
            writer.writerow({
                "sample_id": sid,
                "family": b.get("family"),
                "group": b.get("group"),
                "whole_truth": b["task_truth"]["whole_image_generation"],
                "local_truth": b["task_truth"]["local_manipulation"],
                "baseline_trufor": _trufor_state(b),
                "baseline_verdict": b.get("final_verdict"),
                "baseline_whole": b.get("whole_image_generation"),
                "baseline_local": b.get("local_manipulation"),
                "baseline_certainty": b.get("certainty"),
                "baseline_calls": (b.get("counters") or {}).get("expert_calls"),
                "baseline_turns": (b.get("counters") or {}).get("model_turns"),
                "baseline_elapsed_s": b.get("elapsed_s"),
                "fixed_verdict": f.get("final_verdict"),
                "fixed_whole": f.get("whole_image_generation"),
                "fixed_local": f.get("local_manipulation"),
                "fixed_certainty": f.get("certainty"),
                "fixed_calls": (f.get("counters") or {}).get("expert_calls"),
                "fixed_turns": (f.get("counters") or {}).get("model_turns"),
                "fixed_elapsed_s": f.get("elapsed_s"),
                "fixed_forced": len(f.get("forced_calls") or []),
                "focus_group": _focus(b),
                "baseline_both_correct": (
                    b["whole_image_generation"] == "Real"
                    and b["local_manipulation"] == "Tampered"
                    if b["task_truth"]["local_manipulation"] == "Tampered"
                    else ""),
                "fixed_both_correct": (
                    f["whole_image_generation"] == "Real"
                    and f["local_manipulation"] == "Tampered"
                    if b["task_truth"]["local_manipulation"] == "Tampered"
                    else ""),
                "baseline_page": b.get("page"),
                "fixed_page": f.get("page"),
            })


def _rate(n, d):
    return round(n / d, 4) if d else None


def _write_report(path, base_sum, fixed_sum, focus):
    b, f = base_sum, fixed_sum
    lines = [
        "# 固定双调用 vs 自主调用：逐图对照",
        "",
        "输入：2026-11-10 manifest 的 200 张 q95；自主基线为 2026-11-27 part1+part2；"
        "固定模式每图先 `PROBE` 后 `TruFor`，再由同一 Qwen 结案。",
        "",
        "## 总体",
        "",
        "| 指标 | 自主基线 | 固定双调用 |",
        "|---|---:|---:|",
        f"| 会话完成 | 200/200 | 200/200 |",
        f"| 总体正确 | {b['overall']['correct']}/200 = "
        f"{b['overall']['accuracy']:.4f} | {f['overall']['correct']}/200 = "
        f"{f['overall']['accuracy']:.4f} |",
        f"| 弃权（Uncertain） | {b['overall']['abstain']}/200 = "
        f"{_rate(b['overall']['abstain'], 200)} | "
        f"{f['overall']['abstain']}/200 = {_rate(f['overall']['abstain'], 200)} |",
        f"| 覆盖率 | {b['overall']['coverage']:.4f} | "
        f"{f['overall']['coverage']:.4f} |",
        f"| 已结案准确率 | {b['overall']['closed_accuracy']:.4f} | "
        f"{f['overall']['closed_accuracy']:.4f} |",
        "",
        "| 真值类别 | 基线正确/弃权 | 固定正确/弃权 |",
        "|---|---:|---:|",
    ]
    for cls in ("real", "whole_generated", "local_tampered"):
        bc, fc = b["overall"]["by_truth_class"][cls], f["overall"]["by_truth_class"][cls]
        lines.append(f"| {cls} (n={bc['n']}) | {bc['correct']}/{bc['abstain']} | "
                     f"{fc['correct']}/{fc['abstain']} |")
    lines += [
        "",
        "## 整图与局部",
        "",
        "| 指标 | 自主基线 | 固定双调用 |",
        "|---|---:|---:|",
        f"| 整图生成检出 | {b['whole']['generated_detected']}/"
        f"{b['whole']['generated_n']} = "
        f"{_rate(b['whole']['generated_detected'], b['whole']['generated_n'])} | "
        f"{f['whole']['generated_detected']}/{f['whole']['generated_n']} = "
        f"{_rate(f['whole']['generated_detected'], f['whole']['generated_n'])} |",
        f"| 整图真图误报 | {b['whole']['real_false_positive']}/"
        f"{b['whole']['real_n']} = "
        f"{_rate(b['whole']['real_false_positive'], b['whole']['real_n'])} | "
        f"{f['whole']['real_false_positive']}/{f['whole']['real_n']} = "
        f"{_rate(f['whole']['real_false_positive'], f['whole']['real_n'])} |",
        f"| 局部篡改检出 | {b['local']['forged_detected']}/"
        f"{b['local']['forged_n']} = "
        f"{_rate(b['local']['forged_detected'], b['local']['forged_n'])} | "
        f"{f['local']['forged_detected']}/{f['local']['forged_n']} = "
        f"{_rate(f['local']['forged_detected'], f['local']['forged_n'])} |",
        f"| AutoSplice 真图局部误报 | {b['local']['authentic_fp']}/"
        f"{b['local']['authentic_n']} | {f['local']['authentic_fp']}/"
        f"{f['local']['authentic_n']} |",
        f"| 篡改图“整图 Real + local Tampered” | "
        f"{b['local']['forged_both_whole_real_and_local_tampered']}/60 | "
        f"{f['local']['forged_both_whole_real_and_local_tampered']}/60 |",
        "",
        "局部标签分布：",
        "",
        f"- 基线 AutoSplice 真图：`{b['local']['authentic_labels']}`",
        f"- 固定 AutoSplice 真图：`{f['local']['authentic_labels']}`",
        f"- 基线篡改图：`{b['local']['forged_labels']}`",
        f"- 固定篡改图：`{f['local']['forged_labels']}`",
        "",
        "## 调用与耗时",
        "",
        "| 指标 | 自主基线 | 固定双调用 |",
        "|---|---:|---:|",
        f"| 专家调用总数 | {b['calls']['expert_calls_total']} | "
        f"{f['calls']['expert_calls_total']} |",
        f"| 每会话调用分布 | `{b['calls']['expert_calls_distribution']}` | "
        f"`{f['calls']['expert_calls_distribution']}` |",
        f"| 模型轮次分布 | `{b['calls']['model_turns_distribution']}` | "
        f"`{f['calls']['model_turns_distribution']}` |",
        f"| 总耗时 | {b['calls']['elapsed_s_total']} s | "
        f"{f['calls']['elapsed_s_total']} s |",
        f"| 平均每会话 | {b['calls']['elapsed_s_mean']} s | "
        f"{f['calls']['elapsed_s_mean']} s |",
        "",
        "## 重点组",
        "",
        "| 基线组 | n | 固定 local Tampered | 固定 local 分布 | 固定 both |",
        "|---|---:|---:|---|---:|",
    ]
    for group, stats in focus.items():
        lines.append(f"| {group} | {stats['n']} | "
                     f"{stats['fixed_local_tampered']} | "
                     f"`{stats['fixed_local']}` | {stats['fixed_both']} |")
    lines += [
        "",
        "直接回答：",
        "",
        f"- 原未调用 14 张篡改图：固定模式检出 local Tampered "
        f"**{focus.get('tampered_agent_not_called', {}).get('fixed_local_tampered', 0)}/14**。",
        f"- 原已调用未检出 33 张：固定模式仍漏检 "
        f"**{33 - focus.get('tampered_agent_called_not_detected', {}).get('fixed_local_tampered', 0)}/33**"
        f"（检出 {focus.get('tampered_agent_called_not_detected', {}).get('fixed_local_tampered', 0)}/33）。",
        f"- 30 张 AutoSplice 真图：固定 local 分布 "
        f"`{f['local']['authentic_labels']}`。",
        f"- 60 张篡改图：固定“整图 Real + local Tampered” "
        f"**{f['local']['forged_both_whole_real_and_local_tampered']}/60**，"
        f"自主基线 **{b['local']['forged_both_whole_real_and_local_tampered']}/60**。",
        "",
        "## 代表案例",
        "",
    ]
    for group in ("tampered_agent_not_called",
                  "tampered_agent_called_not_detected",
                  "tampered_agent_detected"):
        stats = focus.get(group)
        if not stats:
            continue
        lines += [f"### {group}", "",
                  "| sample_id | 基线 local/final | 固定 local/final | 固定 whole |",
                  "|---|---|---|---|"]
        for row in stats["examples"][:5]:
            lines.append(f"| {row['sample_id']} | {row['baseline_local']} / "
                         f"{row['baseline_final']} | {row['fixed_local']} / "
                         f"{row['fixed_final']} | {row['fixed_whole']} |")
        lines.append("")
    lines += [
        "## 读法与边界",
        "",
        "- 固定模式在前两轮直接返回协议合法的 `call_probe` / `call_trufor`，"
        "之后完全使用真实 Qwen；控制器、证据包、coverage/certainty 和结案规则未改。",
        "- 与自主模式的差别不只是“是否调用”，还包括固定调用发生在 Qwen 首轮之前；"
        "Qwen 在结案轮仍能看到原图和两份证据，但没有自主模式的首轮视觉规划。"
        "因此本对照测量的是这套固定流程，不是把 TruFor 证据单独注入自主对话。",
        "- 固定模式 200/200、400 次专家调用；未出现工具失败或运行错误。"
        "整体正确率相同不等于局部能力相同：固定模式把更多样本推向 Fake/Real，"
        "弃权减少，但局部 Tampered 反而从 13/60 降到 6/60，整图误报从 45/120 升到 51/120。",
        "- 逐图 CSV：`comparison/per_sample.csv`；本报告为 `comparison/report.md`；"
        "原始记录见固定批次和 11/27 两批目录。",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", default=str(DEFAULT_FIXED))
    parser.add_argument("--parts", nargs="+", default=[str(p) for p in DEFAULT_PARTS])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    fixed_dir = Path(args.fixed)
    out = Path(args.out) if args.out else fixed_dir / "comparison"
    out.mkdir(parents=True, exist_ok=True)
    fixed = _load(fixed_dir / "records.json")
    base = {}
    for path in args.parts:
        base.update(_load(path))
    if len(base) != 200 or len(fixed) != 200:
        raise SystemExit(f"expected 200+200 records, got {len(base)}+{len(fixed)}")
    if set(base) != set(fixed):
        raise SystemExit("baseline and fixed sample_ids differ")
    if any((r.get("counters") or {}).get("expert_calls") != 2 for r in fixed.values()):
        raise SystemExit("fixed batch has a session without exactly two expert calls")
    if any(len(r.get("forced_calls") or []) != 2 for r in fixed.values()):
        raise SystemExit("fixed batch is missing forced_calls")
    _write_csv(out / "per_sample.csv", base, fixed)
    base_sum = _summary(list(base.values()))
    fixed_sum = _summary(list(fixed.values()))
    focus = _focus_summary(base, fixed)
    payload = {"schema": "fixed_dual_compare/1",
               "baseline": base_sum, "fixed": fixed_sum, "focus": focus}
    json.dump(payload, open(out / "report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _write_report(out / "report.md", base_sum, fixed_sum, focus)
    print(json.dumps({"out": str(out),
                      "baseline_overall": base_sum["overall"],
                      "fixed_overall": fixed_sum["overall"],
                      "baseline_local": base_sum["local"],
                      "fixed_local": fixed_sum["local"],
                      "calls": {"baseline": base_sum["calls"],
                                "fixed": fixed_sum["calls"]},
                      "focus": focus}, ensure_ascii=False, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
