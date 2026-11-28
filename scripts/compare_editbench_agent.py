"""Compare the EditBench base-Agent run with the existing independent TruFor."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
BATCH = ROOT / "sft_data/editbench_verification_20261128"
TF_IMAGE = BATCH / "analysis/trufor/per_image.csv"
TF_PAIR = BATCH / "analysis/trufor/per_pair.csv"
AGENT = BATCH / "analysis/agent/agent/records.json"
DEFAULT_OUT = BATCH / "analysis/compare"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence(record: dict, source: str):
    return next((item for item in record.get("evidence") or []
                 if item.get("source") == source), None)


def _view(record: dict) -> dict:
    trufor = _evidence(record, "trufor_expert")
    probe = _evidence(record, "probe_dino_expert")
    return {
        "whole": record.get("whole_image_generation"),
        "local": record.get("local_manipulation"),
        "overall": record.get("final_verdict"),
        "certainty": record.get("certainty"),
        "calls": record.get("expert_calls"),
        "trufor_called": trufor is not None,
        "trufor_strength": trufor.get("strength") if trufor else None,
        "trufor_metric": trufor.get("raw_metric") if trufor else None,
        "probe_called": probe is not None,
        "probe_metric": probe.get("raw_metric") if probe else None,
        "report": record.get("report"),
        "suspicious_regions": record.get("suspicious_regions"),
        "coverage_check": record.get("coverage_check"),
        "trace_path": record.get("trace_path"),
        "halting_reason": record.get("halting_reason"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf-image", default=str(TF_IMAGE))
    parser.add_argument("--tf-pair", default=str(TF_PAIR))
    parser.add_argument("--agent", default=str(AGENT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    tf_rows = list(csv.DictReader(Path(args.tf_image).open(encoding="utf-8")))
    tf = {(row["pair_id"], row["variant"]): row for row in tf_rows}
    records = _load(Path(args.agent))
    agents = {row["sample_id"]: row for row in records}

    rows = []
    for (pair_id, variant), tf_row in sorted(tf.items()):
        record = agents.get(f"{pair_id}__{variant}")
        if not record:
            raise SystemExit(f"missing Agent record: {pair_id}__{variant}")
        view = _view(record)
        local_truth = ("Not tampered" if variant == "original" else "Tampered")
        whole_truth = "AI-generated"
        tf_score = float(
            tf_row["original_score"] if variant == "original"
            else tf_row["edited_score"])
        tf_iou = float(tf_row["iou"])
        tf_high = float(tf_row["mask_high_fraction"])
        row = {
            "pair_id": pair_id,
            "variant": variant,
            "local_truth": local_truth,
            "whole_truth": whole_truth,
            "tf_score": tf_score,
            "tf_mask_mean": float(tf_row["mask_mean"]),
            "tf_mask_high_fraction": tf_high,
            "tf_iou": tf_iou,
            "tf_localized": tf_iou >= 0.1,
            "tf_low_score_heatmap": (
                variant == "edited" and tf_score < 0.5
                and (tf_iou >= 0.1 or tf_high >= 0.1)),
            "agent": view,
            "whole_correct": view["whole"] == whole_truth,
            "local_correct": view["local"] == local_truth,
            "local_tampered": view["local"] == "Tampered",
            "local_uncertain": view["local"] == "Uncertain",
            "overall_uncertain": view["overall"] == "Uncertain",
            "both_correct": (variant == "edited"
                             and view["whole"] == whole_truth
                             and view["local"] == "Tampered"),
        }
        rows.append(row)

    edited = [r for r in rows if r["variant"] == "edited"]
    original = [r for r in rows if r["variant"] == "original"]
    agg = {
        "images": len(rows),
        "whole_ai_correct": sum(r["whole_correct"] for r in rows),
        "edited_whole_ai": sum(r["whole_correct"] for r in edited),
        "original_whole_ai": sum(r["whole_correct"] for r in original),
        "edited_local_tampered": sum(r["local_tampered"] for r in edited),
        "edited_both": sum(r["both_correct"] for r in edited),
        "edited_local_uncertain": sum(r["local_uncertain"] for r in edited),
        "original_local_fp": sum(r["local_tampered"] for r in original),
        "original_local_uncertain": sum(r["local_uncertain"] for r in original),
        "overall_uncertain": sum(r["overall_uncertain"] for r in rows),
        "edited_overall_uncertain": sum(r["overall_uncertain"] for r in edited),
        "original_overall_uncertain": sum(r["overall_uncertain"] for r in original),
        "trufor_called": sum(r["agent"]["trufor_called"] for r in rows),
        "calls_total": sum(int(r["agent"]["calls"] or 0) for r in rows),
        "calls_mean": round(sum(int(r["agent"]["calls"] or 0) for r in rows) / len(rows), 2),
        "edited_tf_low_score_heatmap": [r["pair_id"] for r in edited
                                        if r["tf_low_score_heatmap"]],
    }
    payload = {"schema": "editbench_agent_comparison/1",
               "trufor": str(Path(args.tf_image).relative_to(ROOT)),
               "agent": str(Path(args.agent).relative_to(ROOT)),
               "aggregate": agg, "rows": rows}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["pair_id", "variant", "local_truth", "tf_score", "tf_mask_mean",
              "tf_mask_high_fraction", "tf_iou", "tf_localized",
              "tf_low_score_heatmap", "agent_whole", "agent_local", "agent_overall", "agent_calls",
              "agent_trufor_called", "agent_trufor_strength",
              "agent_trufor_metric", "agent_probe_metric", "agent_report",
              "agent_trace_path"]
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            a = row["agent"]
            writer.writerow({
                "pair_id": row["pair_id"], "variant": row["variant"],
                "local_truth": row["local_truth"], "tf_score": row["tf_score"],
                "tf_mask_mean": row["tf_mask_mean"],
                "tf_mask_high_fraction": row["tf_mask_high_fraction"],
                "tf_iou": row["tf_iou"], "tf_localized": row["tf_localized"],
                "tf_low_score_heatmap": row["tf_low_score_heatmap"],
                "agent_whole": a["whole"], "agent_local": a["local"],
                "agent_overall": a["overall"], "agent_calls": a["calls"],
                "agent_trufor_called": a["trufor_called"],
                "agent_trufor_strength": a["trufor_strength"],
                "agent_trufor_metric": a["trufor_metric"],
                "agent_probe_metric": a["probe_metric"],
                "agent_report": a["report"], "agent_trace_path": a["trace_path"],
            })

    lines = [
        "# EditBench 5 对 — base Agent vs 独立 TruFor",
        "",
        "同一批 10 张 JPEG-q95；Agent 沿用当前 PROBE+TruFor、裁剪和自主调用，"
        "未改规则；mask 仅用于离线评估。",
        "",
        "## 聚合",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 整图 AI-generated 判对 | {agg['whole_ai_correct']}/10 |",
        f"| 编辑图整图 AI-generated | {agg['edited_whole_ai']}/5 |",
        f"| 编辑图局部 Tampered | {agg['edited_local_tampered']}/5 |",
        f"| 编辑图 both（整图 AI + 局部 Tampered） | {agg['edited_both']}/5 |",
        f"| 未编辑原图局部误报（Tampered） | {agg['original_local_fp']}/5 |",
        f"| 未编辑原图局部 Uncertain | {agg['original_local_uncertain']}/5 |",
        f"| 总体弃权（Uncertain） | {agg['overall_uncertain']}/10 |",
        f"| 编辑图总体弃权 | {agg['edited_overall_uncertain']}/5 |",
        f"| 原图总体弃权 | {agg['original_overall_uncertain']}/5 |",
        f"| TruFor 调用 | {agg['trufor_called']}/10 |",
        f"| 专家调用总数 / 平均每图 | {agg['calls_total']} / {agg['calls_mean']} |",
        "",
        "整图错误 3/10：`eb_gen_015_sawfish__edited` whole=Uncertain（只调用了 TruFor，未调用 PROBE）；"
        "`eb_gen_114_three__original` 与 `eb_gen_114_three__edited` whole=Real"
        "（PROBE 0.0908 / 0.1482）。",
        "",
        "## 逐图对照",
        "",
        "| pair | variant | TF score | TF mask mean | TF IoU | Agent whole | Agent local | Agent overall | TF called |",
        "|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        a = row["agent"]
        lines.append(
            f"| `{row['pair_id']}` | {row['variant']} | {row['tf_score']:.3f} | "
            f"{row['tf_mask_mean']:.3f} | {row['tf_iou']:.3f} | "
            f"{a['whole']} | {a['local']} | {a['overall']} | "
            f"{'Y' if a['trufor_called'] else 'N'} |")
    lines += [
        "",
        "## 重点案例",
        "",
    ]
    focus = ["eb_gen_029_pig", "eb_gen_064_triangle", "eb_gen_015_sawfish"]
    for pair_id in focus:
        row = next(r for r in rows if r["pair_id"] == pair_id
                   and (r["variant"] == ("edited" if pair_id != "eb_gen_015_sawfish"
                                         else "original")))
        a = row["agent"]
        lines += [
            f"### {pair_id} / {row['variant']}",
            "",
            f"- 真值：whole AI-generated，local {row['local_truth']}。",
            f"- TruFor：score {row['tf_score']:.3f}，mask mean {row['tf_mask_mean']:.3f}，"
            f"mask high {row['tf_mask_high_fraction']:.3f}，IoU {row['tf_iou']:.3f}；"
            f"低分热图覆盖={row['tf_low_score_heatmap']}。",
            f"- Agent：whole **{a['whole']}** / local **{a['local']}** / "
            f"overall **{a['overall']}**；calls {a['calls']}；"
            f"TruFor called {a['trufor_called']} "
            f"(strength {a['trufor_strength']}, metric {a['trufor_metric']})。",
            f"- 结案依据：{a['report']}",
            f"- 热图：`analysis/trufor/previews/{pair_id}.jpg`",
            f"- trace：`{a['trace_path']}`",
            "",
        ]
    lines += [
        "## 产物",
        "",
        "- `per_image.csv`：10 行，逐图 TruFor 与 Agent 对照。",
        "- `comparison.json`：完整聚合和 Agent 记录。",
        "- 判定依据中的 Agent 报告直接来自 `analysis/agent/agent/records.json`；"
        "热图见 `analysis/trufor/previews/`。",
        "",
        "## 边界",
        "",
        "本轮只跑 base Agent；未改规则、未训练、未安装 PIM。"
        "TruFor 的 0.947/0.601 等分数来自上一轮独立 JPEG-q95 运行。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": agg, "report": str(out / "report.md")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
