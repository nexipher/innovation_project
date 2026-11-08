"""Compare two dual-expert acceptance batches on the five review concerns.

This is a *descriptive* comparison, not an acceptance verdict.  The heuristics
are deliberately narrow and their matched sentences are printed beside every
count so a person can check them.  Box/text mismatch is a candidate list only:
the overlay images are the authoritative view.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")

IMAGE_MISSING = re.compile(
    r"(?:未|没有|无法|不能|未能|看不到|\bdid not\b|\bdidn't\b|\bno\b|\bnot\b|"
    r"\bwithout\b|\bcannot\b|\bcan't\b).{0,24}"
    r"(?:提供|看到|获得|显示|读取|标注|provide|show|see|include|contain).{0,24}"
    r"(?:异常图|置信度图|可靠性图|热图|map|images?)", re.I)
# Only phrases that join the two experts' conclusions.  Generic agreement
# wording ("高度吻合") is usually the model comparing its own visual observation
# with a map, not PROBE with TruFor, so it is tracked separately.
MUTUAL_STRONG = re.compile(
    r"(?:相互验证|互相验证|相互印证|互相印证|双重验证|交叉验证|交叉印证|"
    r"共同支持|彼此印证|相互支持|同时高分|均高|"
    r"mutual(?:ly)? valid|cross.?validat|corroborat|high agreement|both high)", re.I)
VISUAL_AGREEMENT = re.compile(
    r"(?:高度吻合|高度一致|高度符合|consistent with each other)", re.I)
NEGATION = re.compile(
    r"(?:不|无需|无须|并非|不是|未|没有|无|no|not|without|cannot|can't)\s*$",
    re.I)
MUTUAL_WEAK = re.compile(r"(?:方向一致|结果一致|二者一致|两个专家一致|无冲突|"
                         r"consistent|in agreement|no conflict|agree)", re.I)
GENERIC_UNCERTAIN = re.compile(
    r"(?:证据不足|无法判断|无法确定|无法证明|不能确定|不足以判断|信息不足|"
    r"缺乏明确证据|无法确认|insufficient evidence|cannot determine|cannot prove|"
    r"unable to determine|not enough evidence|uncertain)", re.I)
LIMITATION = re.compile(
    r"(?:失败|未能执行|RuntimeError|缺失|未测量|没有测量|未调用|低可靠|"
    r"可靠性低|置信度低|适用性|适用范围|图文|无法对应|对应不上|"
    r"未提供图|没有图|无法看到图|阈值|映射|坐标|confidence_map|anomaly_map|"
    r"具体可疑区域|可靠性(?:地图|图|评估|信息)|"
    r"Not applicable|未达到|低于阈值|高于阈值|failed|failure|missing|not measured|"
    r"not called|low reliability|low confidence|threshold|mapping|coordinates|"
    r"confidence map|anomaly map|未被.{0,10}(测量|评估)|未.{0,10}(测量|评估)|"
    r"未校准|uncalibrated|not measured|not assessed)", re.I)
# A complaint that a map does not carry coordinates/reliability is not the same
# as claiming the map was never delivered.  The first review confused the two.
COORDINATE_COMPLAINT = re.compile(
    r"(?:具体)?(?:坐标|bbox|可靠性区域|可靠性信息|明确标注|可靠性地图)", re.I)

LOCATION_RULES = (
    ("头部", "top", "头"),
    ("顶部", "top", "顶"),
    ("上方", "top", "上"),
    ("天空", "top", "天"),
    ("底部", "bottom", "底"),
    ("下部", "bottom", "下"),
    ("下方", "bottom", "下"),
    ("左侧", "left", "左"),
    ("左边", "left", "左"),
    ("右侧", "right", "右"),
    ("右边", "right", "右"),
    ("中央", "center", "中"),
    ("中间", "center", "中"),
)


def _closing_text(trace: dict) -> str:
    assistant = [turn.get("value", "") for turn in trace.get("conversations", [])
                 if turn.get("from") == "gpt"]
    return assistant[-1] if assistant else ""


def _review_text(record: dict, trace: dict) -> str:
    return "\n".join([str(record.get("report") or ""), _closing_text(trace)])


def _negated(text: str, match) -> bool:
    """A phrase such as "无需交叉验证" is not a cross-task validation claim."""
    prefix = text[max(0, match.start() - 12):match.start()]
    return bool(NEGATION.search(prefix))


def _load(batch: Path):
    records = json.load(open(batch / "records.json", encoding="utf-8"))
    rows = []
    for record in records:
        trace = json.load(open(batch / record["trace_path"], encoding="utf-8"))
        rows.append((record, trace, _review_text(record, trace)))
    return rows


def _box_candidates(record: dict, text: str):
    candidates = []
    for index, region in enumerate(record.get("suspicious_regions") or [], 1):
        bbox = region.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]
        cy, cx = (ymin + ymax) / 2.0, (xmin + xmax) / 2.0
        reason = str(region.get("reason") or "")
        combined = reason + "\n" + text
        for word, side, _short in LOCATION_RULES:
            if word not in combined:
                continue
            bad = ((side == "top" and cy > 500) or (side == "bottom" and cy < 500)
                   or (side == "left" and cx > 500) or (side == "right" and cx < 500)
                   or (side == "center" and (abs(cy - 500) > 250 or abs(cx - 500) > 250)))
            if bad:
                candidates.append({
                    "variant_id": record["variant_id"],
                    "region_index": index,
                    "bbox": list(bbox),
                    "centre": [round(cy, 1), round(cx, 1)],
                    "location_word": word,
                    "why": f"框中心在 {[round(cy, 1), round(cx, 1)]}，文字却提到 {word}",
                })
                break
    return candidates


def measure(batch: Path):
    rows = _load(batch)
    image_missing, coordinate_complaints, map_content_claims = [], [], []
    mutual_strong, mutual_strong_negated, visual_agreement = [], [], []
    mutual_weak = []
    overall_uncertain = []
    generic_uncertain = []
    boxes = []
    text_without_box = []
    full_boxes_detail = []
    confidences = []
    whole_correct = local_correct = both_correct = 0
    local_false_positive = 0
    joint_issues = 0
    local_without_trufor = 0
    call_counts = Counter()
    first_tool = Counter()
    session_rows = []
    for record, trace, text in rows:
        delivered_images = max(
            [c.get("images_attached") or 0
             for c in (trace.get("metadata") or {}).get("model_calls") or []]
            or [0])
        has_maps = any(token.get("visual_artifacts")
                       for token in trace.get("evidence_chain") or [])
        image_match = IMAGE_MISSING.search(text)
        if delivered_images >= 3 and has_maps and image_match:
            window = text[max(0, image_match.start() - 40):image_match.end() + 40]
            item = {"variant_id": record["variant_id"],
                    "match": image_match.group(0), "sentence": text[-600:]}
            if COORDINATE_COMPLAINT.search(window):
                coordinate_complaints.append(item)
            elif re.search(r"(?:篡改|编辑)区域", image_match.group(0)):
                map_content_claims.append(item)
            else:
                image_missing.append(item)
        mutual_match = MUTUAL_STRONG.search(text)
        if mutual_match:
            item = {"variant_id": record["variant_id"],
                    "match": mutual_match.group(0)}
            if _negated(text, mutual_match):
                mutual_strong_negated.append(item)
            else:
                mutual_strong.append(item)
        else:
            visual_match = VISUAL_AGREEMENT.search(text)
            if visual_match:
                visual_agreement.append({"variant_id": record["variant_id"],
                                         "match": visual_match.group(0)})
            elif MUTUAL_WEAK.search(text):
                mutual_weak.append({"variant_id": record["variant_id"],
                                    "match": MUTUAL_WEAK.search(text).group(0)})
        verdict = record.get("final_verdict")
        if verdict == "Uncertain":
            overall_uncertain.append(record["variant_id"])
            report = str(record.get("report") or "")
            if GENERIC_UNCERTAIN.search(report) and not LIMITATION.search(report):
                generic_uncertain.append({
                    "variant_id": record["variant_id"],
                    "sentence": report[-600:],
                })
        boxes.extend(_box_candidates(record, text))
        region_claim = re.search(r"(?:高响应|高亮|候选区域|异常区域|局部区域)", text)
        negated_claim = re.search(
            r"(?:无|没有|未见|不存在).{0,12}(?:高响应|高亮|候选区域|异常区域|局部区域)",
            text)
        if (not (record.get("suspicious_regions") or [])
                and region_claim and not negated_claim):
            text_without_box.append({
                "variant_id": record["variant_id"],
                "sentence": text[-500:],
            })
        for region in record.get("suspicious_regions") or []:
            if region.get("bbox") == [0, 0, 1000, 1000]:
                full_boxes_detail.append({
                    "variant_id": record["variant_id"],
                    "scope": region.get("scope"),
                    "reason": region.get("reason"),
                })
        confidences.append(record.get("confidence"))
        whole_truth = record["task_truth"]["whole_image_generation"]
        local_truth = record["task_truth"]["local_manipulation"]
        whole_ok = record.get("whole_image_generation") == whole_truth
        local_ok = record.get("local_manipulation") == local_truth
        whole_correct += int(whole_ok)
        local_correct += int(local_ok)
        both_correct += int(whole_ok and local_ok)
        if (local_truth in ("Not tampered", "Not applicable")
                and record.get("local_manipulation") == "Tampered"):
            local_false_positive += 1
        issues = record.get("joint_verdict_issues") or []
        joint_issues += bool(issues)
        if any("without a TruFor measurement" in str(item) for item in issues):
            local_without_trufor += 1
        sources = [str(e.get("source")) for e in trace.get("evidence_chain", [])
                   if e.get("source") and e.get("status") != "failed"]
        call_counts[len(sources)] += 1
        if sources:
            first_tool[sources[0]] += 1
        session_rows.append({
            "variant_id": record["variant_id"],
            "truth": record.get("task_truth"),
            "verdict": record.get("final_verdict"),
            "whole": record.get("whole_image_generation"),
            "local": record.get("local_manipulation"),
            "confidence": record.get("confidence"),
            "calls": len(sources),
            "first_tool": sources[0] if sources else None,
            "issues": record.get("joint_verdict_issues") or [],
        })
    numeric = [float(c) for c in confidences if isinstance(c, (int, float))]
    return {
        "batch": str(batch),
        "sessions": len(rows),
        "image_missing": image_missing,
        "coordinate_complaints": coordinate_complaints,
        "map_content_claims": map_content_claims,
        "mutual_strong": mutual_strong,
        "mutual_strong_negated": mutual_strong_negated,
        "visual_agreement": visual_agreement,
        "mutual_weak": mutual_weak,
        "overall_uncertain": overall_uncertain,
        "generic_uncertain": generic_uncertain,
        "box_text_candidates": boxes,
        "text_region_without_box": text_without_box,
        "full_boxes_detail": full_boxes_detail,
        "confidence": {
            "values": Counter(confidences),
            "n_095": sum(c == 0.95 for c in numeric),
            "n_high": sum(c >= 0.9 for c in numeric),
            "mean": round(statistics.mean(numeric), 4) if numeric else None,
            "stdev": round(statistics.pstdev(numeric), 4) if numeric else None,
        },
        "full_image_boxes": len(full_boxes_detail),
        "regions": sum(len(record.get("suspicious_regions") or [])
                       for record, _trace, _text in rows),
        "call_counts": dict(call_counts),
        "first_tool": dict(first_tool),
        "whole_correct": whole_correct,
        "local_correct": local_correct,
        "both_correct": both_correct,
        "local_false_positive": local_false_positive,
        "joint_issues": joint_issues,
        "local_without_trufor": local_without_trufor,
        "session_rows": session_rows,
    }


def _markdown(control, new):
    def row(label, key, extract):
        c = extract(control[key])
        n = extract(new[key])
        if isinstance(c, (int, float)) and isinstance(n, (int, float)):
            delta = f"{n - c:+g}"
        else:
            delta = "—"
        return f"| {label} | {c} | {n} | {delta} |"
    lines = [
        "# Dual-expert control vs new batch — descriptive comparison",
        "",
        "> This is **not** a human acceptance verdict. Heuristic matches are listed",
        "> below; box/text mismatch is a candidate list and must be checked against",
        "> the overlay images.  Negated phrases (\"无需交叉验证\") and visual/map",
        "> agreement (\"与观察高度吻合\") are separated from cross-expert claims;",
        "> a complaint that a map lacks coordinates is not a missing-image claim.",
        "",
        "| 指标 | 旧 24 条（控制） | 新批次 | Δ（新−旧） |",
        "| --- | ---: | ---: | ---: |",
        row("sessions", "sessions", lambda x: x),
        row("整图判断完全正确", "whole_correct", lambda x: x),
        row("局部判断完全正确", "local_correct", lambda x: x),
        row("两维都正确", "both_correct", lambda x: x),
        row("Real/Not-tampered 被误判 Tampered", "local_false_positive", lambda x: x),
        row("joint_verdict_issues 会话", "joint_issues", lambda x: x),
        row("局部结论缺 TruFor 测量", "local_without_trufor", lambda x: x),
        row("expert call 数为 2 的会话", "call_counts", lambda x: x.get(2, 0)),
        row("TruFor 先调用的会话", "first_tool", lambda x: x.get("trufor_expert", 0)),
        row("图片误读候选（有图却称未提供/看不到）", "image_missing", len),
        row("抱怨 map 没给坐标/可靠性（不是未投送）", "coordinate_complaints", len),
        row("强跨任务相互验证措辞（排除否定句）", "mutual_strong", len),
        row("其中被否定的相互验证句（只列不计）", "mutual_strong_negated", len),
        row("视觉观察与 map 的相似措辞（非跨专家）", "visual_agreement", len),
        row("弱‘方向/结果一致’措辞", "mutual_weak", len),
        row("总体 Uncertain", "overall_uncertain", len),
        row("笼统 Uncertain（无具体限制）", "generic_uncertain", len),
        row("框文不一致候选", "box_text_candidates", len),
        row("文字提到异常区/高响应但无框", "text_region_without_box", len),
        row("可疑区域数", "regions", lambda x: x),
        row("全图 bbox 数", "full_image_boxes", lambda x: x),
        row("confidence=0.95 会话", "confidence", lambda x: x["n_095"]),
        row("confidence>=0.9 会话", "confidence", lambda x: x["n_high"]),
        row("confidence mean", "confidence", lambda x: x["mean"]),
        row("confidence stdev", "confidence", lambda x: x["stdev"]),
        "",
        "## 候选明细",
        "",
    ]
    for title, key in (
        ("图片误读（明确称图未提供）", "image_missing"),
        ("map 缺坐标/可靠性抱怨（不是未投送）", "coordinate_complaints"),
        ("强相互验证（排除否定句）", "mutual_strong"),
        ("被否定的相互验证句（不计为违规）", "mutual_strong_negated"),
        ("视觉观察与 map 的相似措辞（非跨专家）", "visual_agreement"),
        ("弱一致措辞", "mutual_weak"),
        ("笼统 Uncertain", "generic_uncertain"),
        ("框文不一致", "box_text_candidates"),
        ("文字提到异常区/高响应但无框", "text_region_without_box"),
    ):
        lines.append(f"### 新批次：{title}")
        items = new[key]
        if not items:
            lines.append("无自动命中。")
        for item in items:
            lines.append(f"- `{item.get('variant_id')}`: {item.get('match') or item.get('why') or item.get('sentence', '')[:180]}")
        lines.append("")
    lines.append("### 全图 bbox 明细（人工核对这些框是否被写成局部定位）")
    if not new["full_boxes_detail"]:
        lines.append("无全图 bbox。")
    for item in new["full_boxes_detail"]:
        lines.append(f"- `{item['variant_id']}` scope={item.get('scope')}: "
                     f"{str(item.get('reason') or '')[:200]}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--control", default=str(
        ROOT / "sft_data/g8_acceptance/dual_20261107"))
    parser.add_argument("--new", default=str(
        ROOT / "sft_data/g8_acceptance/dual_20261108"))
    parser.add_argument("--out", default=str(
        ROOT / "sft_data/g8_acceptance/dual_20261108/comparison"))
    args = parser.parse_args()
    control = measure(Path(args.control))
    new = measure(Path(args.new))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "comparison.json").write_text(
        json.dumps({"control": control, "new": new}, ensure_ascii=False,
                   indent=2, default=str) + "\n", encoding="utf-8")
    (out / "comparison.md").write_text(_markdown(control, new) + "\n",
                                       encoding="utf-8")
    print(json.dumps({
        "control": control["sessions"], "new": new["sessions"],
        "image_missing": [len(control["image_missing"]), len(new["image_missing"])],
        "coordinate_complaints": [len(control["coordinate_complaints"]),
                                  len(new["coordinate_complaints"])],
        "mutual_strong": [len(control["mutual_strong"]), len(new["mutual_strong"])],
        "mutual_strong_negated": [len(control["mutual_strong_negated"]),
                                  len(new["mutual_strong_negated"])],
        "visual_agreement": [len(control["visual_agreement"]),
                             len(new["visual_agreement"])],
        "generic_uncertain": [len(control["generic_uncertain"]), len(new["generic_uncertain"])],
        "box_candidates": [len(control["box_text_candidates"]), len(new["box_text_candidates"])],
        "confidence_095": [control["confidence"]["n_095"], new["confidence"]["n_095"]],
        "out": str(out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
