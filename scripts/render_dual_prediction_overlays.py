"""Render prediction-box overlays for an existing dual-expert batch.

This is review-only.  It reads the original records/evidence (never rewrites
them) and writes overlays on the original image and on the TruFor anomaly map.
The original 24-session batch remains the control; this script writes to a new
directory by default.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from utils.review_overlays import render_record_overlays  # noqa: E402

DEFAULT_BATCH = ROOT / "sft_data/g8_acceptance/dual_20261107"
DEFAULT_OUT = ROOT / "sft_data/g8_acceptance/dual_20261107_prediction_overlays_20261108"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", default=str(DEFAULT_BATCH))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    batch = Path(args.batch)
    out = Path(args.out)
    records = json.load(open(batch / "records.json", encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(records, 1):
        overlays = render_record_overlays(record, str(out))
        rows.append({
            "variant_id": record["variant_id"],
            "ground_truth": record["ground_truth"],
            "verdict": record["final_verdict"],
            "whole_image_generation": record.get("whole_image_generation"),
            "local_manipulation": record.get("local_manipulation"),
            "suspicious_regions": record.get("suspicious_regions") or [],
            "overlays": overlays,
            "source_page": f"../dual_20261107/{index:02d}__{record['variant_id']}.html",
        })
    (out / "overlay_manifest.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cards = []
    empty = '<p class="muted">无预测框，未生成 overlay</p>'
    for row in rows:
        imgs = "".join(
            f'<figure style="display:inline-block;vertical-align:top;max-width:360px;'
            f'margin:4px 12px 12px 0"><img src="{html.escape(os.path.relpath(ROOT / p, out))}" '
            f'style="max-width:350px;max-height:350px"><figcaption class="muted">'
            f'{html.escape(os.path.basename(p))}</figcaption></figure>'
            for p in row["overlays"])
        regions = "；".join(
            f"{r.get('bbox')} — {r.get('reason', '')}" for r in row["suspicious_regions"])
        cards.append(
            f'<section style="border-top:1px solid #ccc;padding:8px 0">'
            f'<h2>{html.escape(row["variant_id"])}</h2>'
            f'<p>真值 {html.escape(str(row["ground_truth"]))}｜verdict '
            f'{html.escape(str(row["verdict"]))}｜whole '
            f'{html.escape(str(row["whole_image_generation"]))}｜local '
            f'{html.escape(str(row["local_manipulation"]))}</p>'
            f'<p class="muted">regions: {html.escape(regions or "无")}</p>'
            f'{imgs or empty}'
            f'</section>')
    index = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Prediction-box overlay check (not acceptance)</title>
<style>body{{font-family:sans-serif;max-width:1200px;margin:24px auto}}
.muted{{color:#666}}</style></head><body>
<h1>预测框核验（原图 + TruFor anomaly map）</h1>
<p><b>这不是人工验收结论。</b>原 24 条会话保持为对照，本目录只叠加模型写出的
suspicious_regions；全图框标为 GLOBAL RANGE，不表示已完成局部定位。异常图的坐标
按 content_box 映射到原图 [0,1000] 坐标；旧批次的控制图没有新标题/色条。</p>
<p>来源批次：<code>{html.escape(str(batch))}</code>；共 {len(rows)} 条，
有预测框 {sum(bool(r['overlays']) for r in rows)} 条。</p>
{''.join(cards)}
</body></html>"""
    (out / "index.html").write_text(index, encoding="utf-8")
    print(json.dumps({"out": str(out), "sessions": len(rows),
                      "with_overlays": sum(bool(r["overlays"]) for r in rows)},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
