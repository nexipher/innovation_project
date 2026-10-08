#!/usr/bin/env python3
"""
G4-f: the review workstation.

Human review of `final_v2` needs everything on one page: the source image and
the re-encoded variant the model actually saw, the expert artifacts, the
Evidence Bundle with its calibration and applicability, how the run compares
with its tool-free baseline, and the exact answer the sample would train the
model to give.  Without the comparison the reviewer can only check formatting,
which is precisely the review this project has already learned is not enough.

The workstation is static HTML with no server and no external assets: open
`sft_data/review/index.html` in a browser.  Images are referenced by relative
path, so nothing is copied and the page keeps working from the repository.

Alongside the pages it writes the worklist the review protocol calls for:

  worklist.csv       one row per sample, with the columns to fill in
  review_sets.json   the full first pass, the stratified spot check, and the
                     double-review set (conflict / Uncertain / multi-tool /
                     high confidence)

Decisions are recorded separately with `scripts/record_review.py`.

Usage:
  python scripts/build_review_workstation.py
  python scripts/build_review_workstation.py --stratified-per-stratum 3
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

FINAL_V2_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
BUCKET_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                "sft_honest_abstention.json")
VARIANTS_MANIFEST = os.path.join(PROJECT_ROOT, "sft_data", "variants", "manifest.json")
TRAJECTORIES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "sft_data", "review")

HIGH_CONFIDENCE = 0.8
STRATIFIED_PER_STRATUM = 2

CHECKLIST = [
    ("observation", "<observation> 里的每条视觉陈述，都能在图上直接核验吗？"),
    ("evidence_fidelity", "<forensic_evidence> 与真实 token 一致吗（evidence_id、测量范围、原始数值、校准概率）？"),
    ("reasoning", "<reasoning> 只引用了合格证据吗？反证与失效条件是否如实写出？"),
    ("container", "有没有把 PNG/JPEG 容器当成真假的理由？（这是必须拒绝的错误）"),
    ("verdict", "<verdict> 与后验、与证据强度一致吗？高置信是否有证据支撑？"),
    ("gain", "这条轨迹相对 no-tool 基线的增益是真的吗（ΔBrier 为负且结论正确）？"),
    ("teaching", "训练这条样本会不会教出坏习惯（格式捷径 / 无理由弃权 / 模板化）？"),
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_samples(final_v2_dir: str = FINAL_V2_DIR) -> List[dict]:
    samples = []
    for name in BUCKET_FILES:
        path = os.path.join(final_v2_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for sample in json.load(handle):
                sample["_bucket_file"] = name.replace("sft_", "").replace(".json", "")
                samples.append(sample)
    return samples


def load_trajectories(directory: str = TRAJECTORIES_DIR) -> Dict[str, dict]:
    records = {}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name == "manifest.json":
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        records[record["trajectory_id"]] = record
    return records


def load_variant_entries(path: str = VARIANTS_MANIFEST) -> Dict[str, dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {entry["variant_id"]: {**entry, "path": key}
            for key, entry in manifest.get("variants", {}).items()}


def load_split(path: Optional[str] = None) -> dict:
    path = path or os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Review sets
# ---------------------------------------------------------------------------

def double_review_reasons(sample: dict, trajectory: Optional[dict]) -> List[str]:
    """Why this sample needs a second reader, if it does."""
    reasons = []
    verdict = (sample.get("final_verdict") or {}).get("verdict")
    confidence = (sample.get("final_verdict") or {}).get("confidence") or 0.0
    tools = (sample.get("metadata") or {}).get("tools_served") or []
    conflict = (trajectory or {}).get("conflict_score") or 0.0
    reasons_ = (trajectory or {}).get("policy_reasons") or []

    if verdict == "Uncertain":
        reasons.append("uncertain")
    if len(tools) > 1:
        reasons.append("multi_tool")
    if confidence >= HIGH_CONFIDENCE and verdict in ("Real", "Fake"):
        reasons.append("high_confidence")
    if conflict > 0.5 or "conflict_unresolved" in reasons_:
        reasons.append("conflict")
    return reasons


def stratified_sample(samples: List[dict], per_stratum: int) -> List[str]:
    """A fixed number per (treatment x generator) so the grid is covered."""
    strata: Dict[tuple, List[str]] = defaultdict(list)
    for sample in samples:
        generator = sample.get("source_model") or "?"
        treatment = (sample.get("metadata") or {}).get("treatment") or "?"
        strata[(treatment, generator)].append(sample["id"])
    picked = []
    for key in sorted(strata):
        picked.extend(sorted(strata[key])[:per_stratum])
    return picked


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _rel(path: str, html_path: str) -> str:
    """Relative URL from a page to an image, so nothing needs copying."""
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    return os.path.relpath(absolute, os.path.dirname(html_path)).replace(os.sep, "/")


def _img(path: str, page_path: str, caption: str, missing: set) -> str:
    if not path:
        return f'<figure class="missing"><figcaption>{html.escape(caption)}（未提供）</figcaption></figure>'
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(absolute):
        missing.add(path)
        return f'<figure class="missing"><figcaption>{html.escape(caption)}（文件缺失）</figcaption></figure>'
    return (f'<figure><img src="{_rel(path, page_path)}" alt="{html.escape(caption)}">'
            f'<figcaption>{html.escape(caption)}</figcaption></figure>')


def _token_rows(tokens: List[dict]) -> str:
    rows = []
    for token in tokens:
        likelihood = token.get("calibrated_likelihood") or {}
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(str(token.get('evidence_id')))}</code></td>"
            f"<td>{html.escape(str(token.get('source')))}</td>"
            f"<td>{html.escape(str(token.get('measurement_scope')))}</td>"
            f"<td>{token.get('raw_metric')}</td>"
            f"<td>{likelihood.get('Real')} / {likelihood.get('Fake')}</td>"
            f"<td>{html.escape(str(token.get('support')))}</td>"
            f"<td>{html.escape(str(token.get('applicability')))}</td>"
            f"<td class='note'>{html.escape(str(token.get('applicability_conditions') or ''))}</td>"
            "</tr>")
    if not rows:
        return '<tr><td colspan="8">本轨迹未调用工具</td></tr>'
    return ("<table><thead><tr><th>evidence_id</th><th>专家</th><th>测量范围</th>"
            "<th>原始数值</th><th>P(Real)/P(Fake)</th><th>方向</th><th>适用性</th>"
            "<th>适用条件</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def render_sample_page(sample: dict, trajectory: Optional[dict],
                       baseline: Optional[dict], source_path: str,
                       index_path: str, page_path: str, missing: set,
                       double_reasons: List[str]) -> str:
    metadata = sample.get("metadata") or {}
    verdict = sample.get("final_verdict") or {}
    tokens = sample.get("evidence_chain") or []

    images = [_img(sample.get("image_path"), page_path,
                   f"模型实际输入的变体（{metadata.get('treatment')}）", missing),
              _img(source_path, page_path, "数据集源图（未经处理）", missing)]
    for token in tokens:
        if token.get("diagnostic_region_image"):
            images.append(_img(token["diagnostic_region_image"], page_path,
                               f"诊断区域（{token.get('source')}）", missing))
        for artifact in token.get("visual_artifacts") or []:
            images.append(_img(artifact, page_path,
                               f"专家产物 {os.path.basename(artifact)}", missing))

    baseline_row = ""
    if baseline:
        baseline_row = (
            "<tr><th>no-tool 基线</th>"
            f"<td>{html.escape(str(baseline.get('final_verdict')))}</td>"
            f"<td>{baseline.get('posterior')}</td>"
            f"<td>{baseline.get('scores', {}).get('posterior_brier')}</td>"
            f"<td>{baseline.get('counters', {}).get('weighted_cost')}</td></tr>")

    scores = (trajectory or {}).get("scores") or {}
    answer = next((turn["value"] for turn in sample.get("conversations", [])
                   if turn.get("from") == "gpt"), "")
    checklist = "".join(
        f'<li><label><input type="checkbox"> <strong>{html.escape(name)}</strong>'
        f' — {html.escape(text)}</label></li>' for name, text in CHECKLIST)

    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>审核 {html.escape(sample['id'])}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 1200px; color: #222; }}
 h1 {{ font-size: 18px; }} h2 {{ font-size: 15px; margin-top: 28px; border-bottom: 1px solid #ddd; }}
 img {{ max-width: 420px; max-height: 420px; border: 1px solid #ccc; }}
 figure {{ display: inline-block; margin: 6px 12px 6px 0; vertical-align: top; }}
 figcaption {{ font-size: 12px; color: #555; max-width: 420px; }}
 .missing {{ background: #f6f6f6; color: #999; padding: 20px; font-size: 12px; }}
 table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; vertical-align: top; }}
 th {{ background: #f4f4f4; }}
 .note {{ color: #666; font-size: 12px; }}
 .answer {{ background: #fafafa; border: 1px solid #eee; padding: 10px; white-space: pre-wrap;
            font-size: 13px; }}
 .flag {{ background: #fff3cd; border-left: 4px solid #e0a800; padding: 8px 12px; }}
 .ok {{ color: #1a7f37; }}
 ul.checklist {{ list-style: none; padding-left: 0; font-size: 13px; }}
 ul.checklist li {{ margin: 4px 0; }}
</style></head><body>
<p><a href="../{os.path.basename(index_path)}">← 返回索引</a></p>
<h1>{html.escape(sample['id'])}</h1>
<p>桶：<strong>{html.escape(sample.get('type', ''))}</strong> ｜ 真值：<strong>{html.escape(sample['ground_truth'])}</strong>
 ｜ 策略：{html.escape(str(metadata.get('policy')))} ｜ 处理：{html.escape(str(metadata.get('treatment')))}
 ｜ 生成器：{html.escape(str(sample.get('source_model')))} ｜ 分区：{html.escape(str(metadata.get('split')))}</p>
{''.join(f'<div class="flag">需双审：{html.escape(r)}</div>' for r in double_reasons)}

<h2>图像证据</h2>
{''.join(images)}

<h2>Evidence Bundle</h2>
{_token_rows(tokens)}

<h2>轨迹与基线对比</h2>
<table><thead><tr><th>来源</th><th>判定</th><th>后验</th><th>Brier</th><th>加权成本</th></tr></thead>
<tbody>
<tr><th>本轨迹</th><td>{html.escape(str(verdict.get('verdict')))}</td><td>{verdict.get('posterior')}</td>
<td>{scores.get('posterior_brier')}</td><td>{metadata.get('counters', {}).get('weighted_cost')}</td></tr>
{baseline_row}
</tbody></table>
<p>准入判定：<strong>{html.escape(str(metadata.get('admission_category')))}</strong>
 ｜ 策略原因：{html.escape(', '.join((trajectory or {}).get('policy_reasons') or []))}</p>

<h2>该样本将教模型说的话</h2>
<pre class="answer">{html.escape(answer)}</pre>

<h2>审核清单（逐项勾选）</h2>
<ul class="checklist">{checklist}</ul>
<p class="note">记录方式：<code>python scripts/record_review.py --sample {html.escape(sample['id'])}
 --reviewer &lt;你&gt; --decision accept|revise|reject|format_only --notes "…"</code></p>
</body></html>
"""


def render_index(samples: List[dict], sets: dict, page_path: str,
                 missing_count: int) -> str:
    rows = []
    for sample in sorted(samples, key=lambda s: s["id"]):
        metadata = sample.get("metadata") or {}
        verdict = sample.get("final_verdict") or {}
        tags = ", ".join(sets["double_review"].get(sample["id"], [])) or "—"
        rows.append(
            "<tr>"
            f'<td><a href="samples/{html.escape(sample["id"])}.html">{html.escape(sample["id"])}</a></td>'
            f"<td>{html.escape(sample.get('type', ''))}</td>"
            f"<td>{html.escape(sample['ground_truth'])}</td>"
            f"<td>{html.escape(str(metadata.get('policy')))}</td>"
            f"<td>{html.escape(str(metadata.get('treatment')))}</td>"
            f"<td>{html.escape(str(sample.get('source_model')))}</td>"
            f"<td>{html.escape(str(verdict.get('verdict')))}</td>"
            f"<td>{verdict.get('confidence')}</td>"
            f"<td class='note'>{html.escape(tags)}</td>"
            f"<td></td></tr>")
    checklist = "".join(f"<li>{html.escape(text)}</li>" for _, text in CHECKLIST)
    return f"""<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>final_v2 审核工作台</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 24px; color: #222; }}
 table {{ border-collapse: collapse; font-size: 13px; width: 100%; }}
 th, td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: left; }}
 th {{ background: #f4f4f4; position: sticky; top: 0; }}
 .note {{ color: #666; }}
 .warn {{ background: #fff3cd; border-left: 4px solid #e0a800; padding: 8px 12px; }}
</style></head><body>
<h1>final_v2 人工审核工作台</h1>
<p>样本 {len(samples)} 条 ｜ 全量首审 + 双审 {len(sets['double_review'])} 条 +
分层抽查 {len(sets['stratified'])} 条</p>
<div class="warn">审核不能只看格式。逐条打开样本页，看源图与变体图、Evidence Bundle、专家产物、
与 no-tool 基线的对比，再判断这条样本会教给模型什么。</div>
<h2>审核要点</h2><ul>{checklist}</ul>
<p>记录：<code>python scripts/record_review.py --sample &lt;id&gt; --reviewer &lt;你&gt;
 --decision accept|revise|reject|format_only --notes "…"</code> ｜
查看进度：<code>python scripts/record_review.py --summary</code></p>
{f'<div class="warn">有 {missing_count} 张图片文件缺失，已在对应样本页标注。</div>' if missing_count else ''}
<table><thead><tr><th>样本</th><th>桶</th><th>真值</th><th>策略</th><th>处理</th><th>生成器</th>
<th>判定</th><th>置信度</th><th>需双审</th><th>你的结论</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(final_v2_dir: str = FINAL_V2_DIR, output_dir: str = OUTPUT_DIR,
          stratified_per_stratum: int = STRATIFIED_PER_STRATUM,
          trajectories_dir: str = TRAJECTORIES_DIR,
          variants_manifest: str = VARIANTS_MANIFEST,
          split_path: Optional[str] = None) -> dict:
    samples = load_samples(final_v2_dir)
    if not samples:
        raise SystemExit(f"no samples in {final_v2_dir}")
    trajectories = load_trajectories(trajectories_dir)
    variants = load_variant_entries(variants_manifest)
    split = load_split(split_path)

    os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)
    index_path = os.path.join(output_dir, "index.html")
    missing: set = set()
    double_review: Dict[str, List[str]] = {}

    for sample in samples:
        metadata = sample.get("metadata") or {}
        trajectory_id = metadata.get("trajectory_id")
        trajectory = trajectories.get(trajectory_id)
        variant = variants.get(trajectory_id.split("__")[-1] if trajectory_id else "")
        baseline = trajectories.get(
            f"no-tool__{trajectory_id.split('__')[-1]}") if trajectory_id else None
        source_path = ""
        if variant:
            source_path = (split.get("sources", {})
                           .get(variant["source_id"], {}).get("path", ""))
        reasons = double_review_reasons(sample, trajectory)
        if reasons:
            double_review[sample["id"]] = reasons

        page_path = os.path.join(output_dir, "samples", f"{sample['id']}.html")
        with open(page_path, "w", encoding="utf-8") as handle:
            handle.write(render_sample_page(sample, trajectory, baseline, source_path,
                                            index_path, page_path, missing, reasons))

    sets = {
        "full": [s["id"] for s in samples],
        "double_review": double_review,
        "stratified": stratified_sample(samples, stratified_per_stratum),
    }
    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(render_index(samples, sets, index_path, len(missing)))
    with open(os.path.join(output_dir, "review_sets.json"), "w", encoding="utf-8") as handle:
        json.dump(sets, handle, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "worklist.csv"), "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "bucket", "ground_truth", "policy", "treatment",
                         "generator", "verdict", "confidence", "needs_double_review",
                         "reviewer", "decision", "notes"])
        for sample in sorted(samples, key=lambda s: s["id"]):
            metadata = sample.get("metadata") or {}
            verdict = sample.get("final_verdict") or {}
            writer.writerow([
                sample["id"], sample.get("type"), sample["ground_truth"],
                metadata.get("policy"), metadata.get("treatment"),
                sample.get("source_model"), verdict.get("verdict"),
                verdict.get("confidence"),
                "|".join(double_review.get(sample["id"], [])), "", "", "",
            ])

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "samples": len(samples),
        "buckets": dict(Counter(s.get("type") for s in samples)),
        "needs_double_review": len(double_review),
        "double_review_reasons": dict(Counter(
            r for reasons in double_review.values() for r in reasons)),
        "stratified": len(sets["stratified"]),
        "missing_images": len(missing),
        "index": os.path.relpath(index_path, PROJECT_ROOT).replace(os.sep, "/"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--stratified-per-stratum", type=int, default=STRATIFIED_PER_STRATUM)
    parser.add_argument("--trajectories-dir", default=TRAJECTORIES_DIR)
    parser.add_argument("--variants-manifest", default=VARIANTS_MANIFEST)
    parser.add_argument("--split-path", default=None)
    args = parser.parse_args()

    report = build(args.final_v2_dir, args.output_dir, args.stratified_per_stratum,
                   args.trajectories_dir, args.variants_manifest, args.split_path)
    print(f"samples: {report['samples']}  buckets: {report['buckets']}")
    print(f"double review: {report['needs_double_review']}  {report['double_review_reasons']}")
    print(f"stratified spot check: {report['stratified']}  missing images: {report['missing_images']}")
    print(f"Open: {report['index']}")


if __name__ == "__main__":
    main()
