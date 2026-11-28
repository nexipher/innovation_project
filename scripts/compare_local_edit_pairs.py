"""Join independent TruFor and Agent results for the 10 GenImage edit pairs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
BATCH = ROOT / "sft_data/genimage_local_edits_20261128"
DEFAULT_TRUFOR = BATCH / "analysis/trufor/trufor_analysis.json"
DEFAULT_AGENT = BATCH / "analysis/agent/agent/records.json"
DEFAULT_OUT = BATCH / "analysis/compare"


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _evidence(record: dict, source: str):
    return next((item for item in record.get("evidence") or []
                 if item.get("source") == source), None)


def _agent_view(record: dict) -> dict:
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
        "report": record.get("report"),
        "suspicious_regions": record.get("suspicious_regions"),
        "coverage_check": record.get("coverage_check"),
        "halting_reason": record.get("halting_reason"),
        "trace_path": record.get("trace_path"),
    }


def _agg(rows: list[dict]) -> dict:
    return {
        "pairs": len(rows),
        "edited_whole_ai": sum(row["agent_edited"]["whole"] == "AI-generated" for row in rows),
        "edited_local_tampered": sum(row["agent_edited"]["local"] == "Tampered" for row in rows),
        "edited_both": sum(row["edited_both"] for row in rows),
        "edited_local_uncertain": sum(row["agent_edited"]["local"] == "Uncertain" for row in rows),
        "edited_overall_uncertain": sum(row["agent_edited"]["overall"] == "Uncertain" for row in rows),
        "original_whole_ai": sum(row["agent_original"]["whole"] == "AI-generated" for row in rows),
        "original_local_fp": sum(row["agent_original"]["local"] == "Tampered" for row in rows),
        "original_local_uncertain": sum(row["agent_original"]["local"] == "Uncertain" for row in rows),
        "original_overall_uncertain": sum(row["agent_original"]["overall"] == "Uncertain" for row in rows),
        "all_overall_uncertain": sum(
            row["agent_original"]["overall"] == "Uncertain"
            or row["agent_edited"]["overall"] == "Uncertain" for row in rows),
        "edited_trufor_called": sum(row["agent_edited"]["trufor_called"] for row in rows),
        "original_trufor_called": sum(row["agent_original"]["trufor_called"] for row in rows),
        "calls_total": sum(int(row["agent_original"]["calls"] or 0)
                           + int(row["agent_edited"]["calls"] or 0) for row in rows),
        "calls_mean": round(sum(int(row["agent_original"]["calls"] or 0)
                                + int(row["agent_edited"]["calls"] or 0)
                                for row in rows) / max(1, len(rows) * 2), 2),
        "classes": dict(Counter(row["agent_class"] for row in rows)),
    }


def _write_reps(rows: list[dict], out_dir: Path) -> list[str]:
    by_id = {row["pair_id"]: row for row in rows}
    picks = ["p02_wukong_tag_add", "p01_vqdm_box_label_add",
             "p05_biggan_stick_delete", "p07_glide_830_caption_delete"]
    for cls in ("agent_not_called", "usable_but_missed",
                "detected_without_trufor_evidence", "detected",
                "trufor_not_localized"):
        for row in rows:
            if row["agent_class"] == cls and row["pair_id"] not in picks:
                picks.append(row["pair_id"])
                break
    for row in rows:
        if row["agent_original"]["local"] == "Tampered" and row["pair_id"] not in picks:
            picks.append(row["pair_id"])
            break
    picks = picks[:5]
    reps_dir = out_dir / "representatives"
    reps_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for pair_id in picks:
        row = by_id[pair_id]
        pair = row["pair"]
        agent_o = row["agent_original"]
        agent_e = row["agent_edited"]
        tf = row["trufor"]
        metrics = tf["edited"]
        orig = tf["original"]
        def img(src, caption):
            return (f"<figure><img src=\"{html.escape(src)}\" loading=\"lazy\">"
                    f"<figcaption>{html.escape(caption)}</figcaption></figure>")
        trace_name = Path(agent_e.get("trace_path") or "").name
        trace_link = f"../../agent/agent/traces/{trace_name}" if trace_name else ""
        cards.append(f"""
<article id="{html.escape(pair_id)}">
<h2>{html.escape(pair_id)}</h2>
<p class="meta">{html.escape(pair['generator'])} · {html.escape(pair['edit_type'])} ·
{html.escape(pair['edit_description'])}</p>
<p><b>TruFor</b>：原图 score {orig['score']:.3f} → 编辑版 {metrics['score']:.3f}
（Δ {row['score_delta']:+.3f}）；mask mean {orig['mask_mean']:.3f} →
{metrics['mask_mean']:.3f}；编辑版 IoU {metrics['iou']:.3f}，
mask 高响应 {metrics['mask_high_fraction']:.3f}。
<b>分类</b>：{html.escape(row['agent_class'])}。</p>
<p><b>Agent 原图</b>：whole {html.escape(str(agent_o['whole']))} / local
{html.escape(str(agent_o['local']))} / overall {html.escape(str(agent_o['overall']))}，
calls {agent_o['calls']}，TruFor called {agent_o['trufor_called']}。</p>
<p><b>Agent 编辑版</b>：whole {html.escape(str(agent_e['whole']))} / local
{html.escape(str(agent_e['local']))} / overall {html.escape(str(agent_e['overall']))}，
calls {agent_e['calls']}，TruFor called {agent_e['trufor_called']}
({html.escape(str(agent_e['trufor_strength']))}, metric
{html.escape(str(agent_e['trufor_metric']))})。</p>
<p><b>结案报告</b>：{html.escape(str(agent_e['report'] or ''))}</p>
<p class="meta">trace: <code>{html.escape(agent_e.get('trace_path') or '')}</code></p>
<div class="grid">
{img('../../../pairs/' + pair['pair_id'] + '/original.jpg', '原图')}
{img('../../../pairs/' + pair['pair_id'] + '/edited.jpg', '编辑版')}
{img('../../../pairs/' + pair['pair_id'] + '/mask.png', 'mask（白色为编辑区域）')}
{img('../../trufor/trufor_maps/' + pair['pair_id'] + '__edited__anomaly_map.png', '编辑版异常图')}
{img('../../trufor/trufor_maps/' + pair['pair_id'] + '__edited__confidence_map.png', '编辑版可靠性图')}
</div>
</article>""")
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenImage 配对代表案例 — TruFor / Agent</title>
<style>
body {{ font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 1500px; margin: 24px auto; padding: 0 20px; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #666; }}
article {{ border-top: 1px solid #bbb; padding: 16px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
figure {{ margin: 0; }}
figure img {{ width: 100%; max-height: 60vh; object-fit: contain;
             background: #111; border-radius: 4px; }}
figcaption {{ color: #666; font-size: 13px; margin-top: 4px; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>GenImage 配对代表案例 — TruFor / Agent</h1>
<p class="meta">同一批 10 对、20 张图；TruFor 独立结果与 Agent 结案记录对照。
异常图/可靠性图为编辑版整图；mask 单独展示。</p>
{''.join(cards)}
</body></html>
"""
    (reps_dir / "index.html").write_text(html_text, encoding="utf-8")
    return picks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trufor", default=str(DEFAULT_TRUFOR))
    parser.add_argument("--agent", default=str(DEFAULT_AGENT))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    trufor_payload = _load(Path(args.trufor))
    records = _load(Path(args.agent))
    record_map = {row["sample_id"]: row for row in records}
    trufor_map = {row["pair_id"]: row for row in trufor_payload["pairs"]}

    rows = []
    for pair_id, tf in trufor_map.items():
        orig_record = record_map.get(f"{pair_id}__original")
        edit_record = record_map.get(f"{pair_id}__edited")
        if not orig_record or not edit_record:
            raise SystemExit(f"missing Agent record for {pair_id}")
        agent_original = _agent_view(orig_record)
        agent_edited = _agent_view(edit_record)
        metrics = tf["edited"]
        tf_found = (
            metrics["score"] >= 0.5
            or (metrics["mask_high_fraction"] or 0) >= 0.1
            or (metrics["iou"] or 0) >= 0.1)
        if agent_edited["local"] == "Tampered":
            agent_class = "detected"
            if not tf_found:
                agent_class = "detected_without_trufor_evidence"
        elif not tf_found:
            agent_class = "trufor_not_localized"
        elif not agent_edited["trufor_called"]:
            agent_class = "agent_not_called"
        else:
            agent_class = "usable_but_missed"
        rows.append({
            "pair_id": pair_id,
            "generator": tf["generator"],
            "edit_type": tf["edit_type"],
            "mask_fraction": metrics["mask_fraction"],
            "large_edit": tf["large_edit"],
            "trufor": tf,
            "score_delta": tf["score_delta"],
            "mask_mean_delta": tf["mask_mean_delta"],
            "tf_found": tf_found,
            "agent_original": agent_original,
            "agent_edited": agent_edited,
            "edited_both": (
                agent_edited["whole"] == "AI-generated"
                and agent_edited["local"] == "Tampered"),
            "agent_class": agent_class,
            "pair": next(pair for pair in _load(BATCH / "manifest.json")["pairs"]
                         if pair["pair_id"] == pair_id),
        })

    aggregate_all = _agg(rows)
    aggregate_no_p05 = _agg([row for row in rows if not row["large_edit"]])
    payload = {
        "schema": "local_edit_pair_comparison/1",
        "trufor": str(Path(args.trufor).relative_to(ROOT)),
        "agent": str(Path(args.agent).relative_to(ROOT)),
        "agent_aggregate_all": aggregate_all,
        "agent_aggregate_excluding_p05": aggregate_no_p05,
        "trufor_aggregate": trufor_payload["aggregate"],
        "pairs": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    fields = [
        "pair_id", "generator", "edit_type", "large_edit", "mask_fraction",
        "tf_original_score", "tf_edited_score", "tf_score_delta",
        "tf_original_mask_mean", "tf_edited_mask_mean", "tf_mask_mean_delta",
        "tf_edited_iou", "tf_edited_mask_high_fraction", "tf_found",
        "tf_class", "agent_class", "agent_edited_both",
        "agent_original_whole", "agent_original_local", "agent_original_overall",
        "agent_original_calls", "agent_original_trufor_called",
        "agent_edited_whole", "agent_edited_local", "agent_edited_overall",
        "agent_edited_calls", "agent_edited_trufor_called",
        "agent_edited_trufor_strength", "agent_edited_trufor_metric",
        "agent_edited_report",
    ]
    with (out / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            o, e, tf = row["agent_original"], row["agent_edited"], row["trufor"]
            writer.writerow({
                "pair_id": row["pair_id"], "generator": row["generator"],
                "edit_type": row["edit_type"], "large_edit": row["large_edit"],
                "mask_fraction": row["mask_fraction"],
                "tf_original_score": tf["original"]["score"],
                "tf_edited_score": tf["edited"]["score"],
                "tf_score_delta": row["score_delta"],
                "tf_original_mask_mean": tf["original"]["mask_mean"],
                "tf_edited_mask_mean": tf["edited"]["mask_mean"],
                "tf_mask_mean_delta": row["mask_mean_delta"],
                "tf_edited_iou": tf["edited"]["iou"],
                "tf_edited_mask_high_fraction": tf["edited"]["mask_high_fraction"],
                "tf_found": row["tf_found"], "tf_class": tf["trufor_class"],
                "agent_class": row["agent_class"],
                "agent_edited_both": row["edited_both"],
                "agent_original_whole": o["whole"], "agent_original_local": o["local"],
                "agent_original_overall": o["overall"],
                "agent_original_calls": o["calls"],
                "agent_original_trufor_called": o["trufor_called"],
                "agent_edited_whole": e["whole"], "agent_edited_local": e["local"],
                "agent_edited_overall": e["overall"], "agent_edited_calls": e["calls"],
                "agent_edited_trufor_called": e["trufor_called"],
                "agent_edited_trufor_strength": e["trufor_strength"],
                "agent_edited_trufor_metric": e["trufor_metric"],
                "agent_edited_report": e["report"],
            })

    image_fields = [
        "pair_id", "variant", "generator", "edit_type", "whole_truth",
        "local_truth", "tf_score", "tf_mask_mean", "tf_mask_p95",
        "tf_mask_high_fraction", "tf_iou", "tf_peak_in_mask",
        "agent_whole", "agent_local", "agent_overall", "agent_calls",
        "agent_trufor_called", "agent_trufor_strength",
        "agent_trufor_metric", "agent_report", "agent_trace_path",
    ]
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            tf = row["trufor"]
            for variant, metrics, agent in (
                    ("original", tf["original"], row["agent_original"]),
                    ("edited", tf["edited"], row["agent_edited"])):
                writer.writerow({
                    "pair_id": row["pair_id"], "variant": variant,
                    "generator": row["generator"], "edit_type": row["edit_type"],
                    "whole_truth": "AI-generated",
                    "local_truth": ("Not tampered" if variant == "original"
                                    else "Tampered"),
                    "tf_score": metrics["score"],
                    "tf_mask_mean": metrics["mask_mean"],
                    "tf_mask_p95": metrics["mask_p95"],
                    "tf_mask_high_fraction": metrics["mask_high_fraction"],
                    "tf_iou": metrics["iou"],
                    "tf_peak_in_mask": metrics["peak_in_mask"],
                    "agent_whole": agent["whole"], "agent_local": agent["local"],
                    "agent_overall": agent["overall"],
                    "agent_calls": agent["calls"],
                    "agent_trufor_called": agent["trufor_called"],
                    "agent_trufor_strength": agent["trufor_strength"],
                    "agent_trufor_metric": agent["trufor_metric"],
                    "agent_report": agent["report"],
                    "agent_trace_path": agent["trace_path"],
                })

    reps = _write_reps(rows, out)
    lines = [
        "# 10 对 GenImage 整图生成/局部编辑 — TruFor + Agent",
        "",
        "固定当前 PROBE + TruFor、high-response 裁剪和自主调用流程；未改规则、未训练。",
        "TruFor 先独立跑 20 张，再用同一批图跑 Agent（原版 Qwen3-VL）。",
        "",
        "## 独立 TruFor",
        "",
        "| 指标 | 全部 10 对 | 不含 p05（9 对） |",
        "|---|---:|---:|",
        f"| 编辑版图像级检出（score>=0.5） | {trufor_payload['aggregate']['edited_image_detected']}/10 | {trufor_payload['aggregate']['excluding_p05']['edited_image_detected']}/9 |",
        f"| 编辑版定位命中（IoU>=0.1） | {trufor_payload['aggregate']['edited_localized']}/10 | {trufor_payload['aggregate']['excluding_p05']['edited_localized']}/9 |",
        f"| 编辑版 mask 高响应占比>=0.1 | {trufor_payload['aggregate']['edited_mask_high_fraction_ge_0.1']}/10 | {trufor_payload['aggregate']['excluding_p05']['edited_mask_high_fraction_ge_0.1']}/9 |",
        f"| 编辑版检出且定位 | {trufor_payload['aggregate']['edited_detected_and_localized']}/10 | {trufor_payload['aggregate']['excluding_p05']['edited_detected_and_localized']}/9 |",
        f"| 原图 score 误报 | {trufor_payload['aggregate']['original_score_fp']}/10 | {trufor_payload['aggregate']['excluding_p05']['original_score_fp']}/9 |",
        f"| 原图同区域定位误报 | {trufor_payload['aggregate']['original_localized_fp']}/10 | {trufor_payload['aggregate']['excluding_p05']['original_localized_fp']}/9 |",
        "",
        "## Agent",
        "",
        "| 指标 | 全部 10 对 | 不含 p05（9 对） |",
        "|---|---:|---:|",
        f"| 编辑版整图 AI-generated | {aggregate_all['edited_whole_ai']}/10 | {aggregate_no_p05['edited_whole_ai']}/9 |",
        f"| 原图整图 AI-generated | {aggregate_all['original_whole_ai']}/10 | {aggregate_no_p05['original_whole_ai']}/9 |",
        f"| 20 张整图 AI-generated | {aggregate_all['edited_whole_ai'] + aggregate_all['original_whole_ai']}/20 | {aggregate_no_p05['edited_whole_ai'] + aggregate_no_p05['original_whole_ai']}/18 |",
        f"| 编辑版局部 Tampered | {aggregate_all['edited_local_tampered']}/10 | {aggregate_no_p05['edited_local_tampered']}/9 |",
        f"| 编辑版同时判对（whole AI + local Tampered） | {aggregate_all['edited_both']}/10 | {aggregate_no_p05['edited_both']}/9 |",
        f"| 编辑版局部 Uncertain | {aggregate_all['edited_local_uncertain']}/10 | {aggregate_no_p05['edited_local_uncertain']}/9 |",
        f"| 原图局部误报（Tampered） | {aggregate_all['original_local_fp']}/10 | {aggregate_no_p05['original_local_fp']}/9 |",
        f"| 原图局部 Uncertain | {aggregate_all['original_local_uncertain']}/10 | {aggregate_no_p05['original_local_uncertain']}/9 |",
        f"| 总体弃权（20 张） | {aggregate_all['all_overall_uncertain']}/20 | {aggregate_no_p05['all_overall_uncertain']}/18 |",
        f"| 编辑版调用了 TruFor | {aggregate_all['edited_trufor_called']}/10 | {aggregate_no_p05['edited_trufor_called']}/9 |",
        f"| 专家调用总数 / 平均每图 | {aggregate_all['calls_total']} / {aggregate_all['calls_mean']} | {aggregate_no_p05['calls_total']} / {aggregate_no_p05['calls_mean']} |",
        "",
        "整图 AI 错误 3/20：p05 编辑版（PROBE 0.490，整图 Real）、p07 原图与编辑版（PROBE 0.25，整图 Real）。",
        "",
        "分类：`detected`（局部 Tampered）、`trufor_not_localized`（独立 TruFor 未达到可用证据）、"
        "`agent_not_called`（TruFor 可用但 Agent 未调用）、"
        "`usable_but_missed`（TruFor 可用且 Agent 调用但仍未写 Tampered）、"
        "`detected_without_trufor_evidence`（独立 TruFor 未命中但 Agent 仍写 Tampered）。",
        "",
        "## 逐对",
        "",
        "| pair | 类型 | TF orig→edit score | TF mask mean | TF IoU | Agent 原图 局部/总体 | Agent 编辑版 整图/局部/总体 | both | class |",
        "|---|---|---|---:|---:|---|---|---|---|",
    ]
    for row in rows:
        tf = row["trufor"]
        o, e = row["agent_original"], row["agent_edited"]
        lines.append(
            f"| `{row['pair_id']}` | {row['edit_type']} | "
            f"{tf['original']['score']:.3f}→{tf['edited']['score']:.3f} | "
            f"{tf['original']['mask_mean']:.3f}→{tf['edited']['mask_mean']:.3f} | "
            f"{tf['edited']['iou']:.3f} | "
            f"{o['local']}/{o['overall']} | "
            f"{e['whole']}/{e['local']}/{e['overall']} | "
            f"{'Y' if row['edited_both'] else 'N'} | {row['agent_class']} |")
    lines += [
        "",
        "## 错因分桶",
        "",
    ]
    for cls in ("detected", "detected_without_trufor_evidence",
                "usable_but_missed", "agent_not_called", "trufor_not_localized"):
        ids = [row["pair_id"] for row in rows if row["agent_class"] == cls]
        lines.append(f"- **{cls}**（{len(ids)}）：" + (", ".join(f"`{x}`" for x in ids) if ids else "—"))
    o_fp = [row["pair_id"] for row in rows
            if row["agent_original"]["local"] == "Tampered"]
    lines += [
        "",
        f"原图局部误报：" + (", ".join(f"`{x}`" for x in o_fp) if o_fp else "无"),
        f"编辑版未调用 TruFor：" + (
            ", ".join(f"`{row['pair_id']}`" for row in rows
                      if not row["agent_edited"]["trufor_called"]) or "无"),
        "",
        "## 代表案例",
        "",
        "见 `representatives/index.html`：" + ", ".join(f"`{x}`" for x in reps),
        "",
        "## p05 单列",
        f"p05 mask 覆盖 {rows[4]['mask_fraction']:.1%} 图像面积；TruFor 原/编 score "
        f"{rows[4]['trufor']['original']['score']:.3f}→{rows[4]['trufor']['edited']['score']:.3f}，"
        f"mask mean {rows[4]['trufor']['original']['mask_mean']:.3f}→{rows[4]['trufor']['edited']['mask_mean']:.3f}，"
        f"IoU {rows[4]['trufor']['edited']['iou']:.3f}；Agent 编辑版 "
        f"{rows[4]['agent_edited']['whole']}/{rows[4]['agent_edited']['local']}/{rows[4]['agent_edited']['overall']}。"
        "大范围编辑未形成高响应，且 PROBE 在编辑后把整图判成 Real；不能与小块编辑平均。",
        "",
        "## 边界",
        "",
        "本轮只跑独立 TruFor 和当前 base Agent；未改规则、未训练、未扩样。"
        "p05 mask 覆盖 45.5% 图像面积，已在大表和小表中单列；其分数/响应不与小块编辑直接平均。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"agent_all": aggregate_all,
                      "agent_excluding_p05": aggregate_no_p05,
                      "representatives": reps,
                      "report": str(out / "report.md")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
