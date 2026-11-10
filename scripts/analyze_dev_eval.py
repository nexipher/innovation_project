"""Analyze the 200-session g8 dev evaluation.

Usage:
  python scripts/analyze_dev_eval.py \
    --batch sft_data/g8_dev_eval_20261110

Outputs SUMMARY.md, report.json, review_queue.json, REVIEW_GUIDE.md and
review_records.csv in the batch directory.  Automatic explanation matches are
candidates only; every candidate points to the turn and sentence.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
DEFAULT_BATCH = ROOT / "sft_data/g8_dev_eval_20261110"
GROUPS = ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong")
MUTUAL_RE = re.compile(
    r"(?:相互验证|互相验证|相互印证|互相印证|双重验证|交叉验证|交叉印证|"
    r"共同支持|彼此印证|相互支持|同时高分|均高|mutual(?:ly)? valid|"
    r"cross.?validat|corroborat|high agreement|both high)", re.I)
PROCESS_RE = re.compile(
    r"(?:后期添加|后期合成|后期编辑|添加了|被添加|被合成|抠图|合成到|"
    r"拼接到|移花接木|added later|pasted|composited)", re.I)
RELIABILITY_RE = re.compile(
    r"(?:可靠性(?:图|度)|confidence\s*map)[^。；\n]{0,80}"
    r"(?:红|高)[^。；\n]{0,80}(?:背景|非主体|边缘)|"
    r"(?:背景|非主体)[^。；\n]{0,80}(?:可靠性(?:图|度)|confidence\s*map)", re.I)
LIMITATION_RE = re.compile(
    r"(?:仅.{0,8}候选|候选.{0,8}(?:非|不是|不构成)|未校准|缺失|失败|"
    r"无法确认|不能确认|不构成证明|非证明|无法排除)", re.I)
NEGATION_RE = re.compile(
    r"(?:不|无需|无须|并非|不是|未|没有|无|no|not|without|cannot|can't)\s*$", re.I)


def load_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 28):start]
    cut = max((prefix.rfind(ch) for ch in "。；;！？\n"), default=-1)
    prefix = prefix[cut + 1:] if cut >= 0 else prefix
    if NEGATION_RE.search(prefix):
        return True
    return bool(re.search(
        r"(?:不构成|不能|不应|不要|不可|无法|并非|不是|没有|无需|无须)",
        prefix, re.I))


def expected_overall(truth: dict) -> str:
    if (truth.get("whole_image_generation") == "Real"
            and truth.get("local_manipulation") == "Tampered"):
        return "Fake"
    if truth.get("whole_image_generation") == "Real":
        return "Real"
    return "Fake"


def truth_class(truth: dict) -> str:
    if truth.get("local_manipulation") == "Tampered":
        return "local_tampered"
    if truth.get("whole_image_generation") == "AI-generated":
        return "whole_generated"
    return "real"


def classify_overall(record: dict) -> str:
    truth = record.get("task_truth") or {}
    expected = expected_overall(truth)
    actual = record.get("final_verdict")
    if actual == expected:
        return "correct"
    if actual in ("Real", "Fake"):
        return "wrong"
    if actual == "Uncertain":
        return "abstain"
    return "no_conclusion"


def ratio(num, den):
    return None if not den else round(num / den, 4)


def prf(tp, fp, fn):
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = round(2 * precision * recall / (precision + recall), 4)
    return precision, recall, f1


def bbox_pixels(bbox, width, height):
    try:
        ymin, xmin, ymax, xmax = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None
    x0 = max(0, min(width, round(xmin / 1000.0 * width)))
    x1 = max(0, min(width, round(xmax / 1000.0 * width)))
    y0 = max(0, min(height, round(ymin / 1000.0 * height)))
    y1 = max(0, min(height, round(ymax / 1000.0 * height)))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def localization(record, np):
    """Compare predicted normalized bboxes against the shared AutoSplice mask."""
    mask_path = record.get("mask_path")
    if not mask_path:
        return None
    try:
        from PIL import Image
        mask = np.array(Image.open(ROOT / mask_path).convert("L")) > 0
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    h, w = mask.shape
    ys, xs = mask.nonzero()
    if len(xs) == 0:
        return {"error": "empty_mask"}
    mask_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    regions = [r for r in record.get("suspicious_regions") or []
               if isinstance(r, dict) and r.get("bbox")]
    values = []
    for region in regions:
        pix = bbox_pixels(region.get("bbox"), w, h)
        if not pix:
            values.append({"bbox": region.get("bbox"), "invalid": True})
            continue
        x0, y0, x1, y1 = pix
        center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
        center_in = bool(mask[min(h - 1, int(center_y)), min(w - 1, int(center_x))])
        inter_x0, inter_y0 = max(x0, mask_bbox[0]), max(y0, mask_bbox[1])
        inter_x1, inter_y1 = min(x1, mask_bbox[2]), min(y1, mask_bbox[3])
        inter = max(0, inter_x1 - inter_x0) * max(0, inter_y1 - inter_y0)
        bbox_area = (x1 - x0) * (y1 - y0)
        mask_area = int(mask.sum())
        bbox_iou = inter / (bbox_area + mask_area - inter) if bbox_area else 0.0
        # Coverage/precision use actual mask pixels, not the mask bbox.
        mask_in_bbox = int(mask[y0:y1, x0:x1].sum())
        mask_cov = mask_in_bbox / mask_area if mask_area else 0.0
        bbox_prec = mask_in_bbox / bbox_area if bbox_area else 0.0
        values.append({"bbox": region.get("bbox"), "center_in_mask": center_in,
                       "bbox_mask_iou": round(bbox_iou, 4),
                       "mask_coverage": round(mask_cov, 4),
                       "bbox_precision": round(bbox_prec, 4)})
    valid = [v for v in values if not v.get("invalid")]
    best = max(valid, key=lambda v: v["bbox_mask_iou"]) if valid else None
    return {
        "mask_path": mask_path,
        "mask_shared_by_base": bool(record.get("mask_shared_by_base")),
        "base_variant_count": record.get("base_variant_count"),
        "regions": values,
        "n_regions": len(values),
        "n_valid_regions": len(valid),
        "best": best,
        "any_center_in_mask": any(v["center_in_mask"] for v in valid),
        "best_iou": best["bbox_mask_iou"] if best else None,
        "best_mask_coverage": best["mask_coverage"] if best else None,
        "best_bbox_precision": best["bbox_precision"] if best else None,
    }


def scan_turns(trace: dict):
    turns = []
    for index, turn in enumerate(trace.get("conversations") or []):
        if turn.get("from") != "gpt":
            continue
        text = str(turn.get("value") or "")
        for match in MUTUAL_RE.finditer(text):
            if negated(text, match.start()):
                continue
            turns.append({"turn": index, "kind": "mutual_wording",
                          "phrase": match.group(0),
                          "sentence": text[max(0, match.start() - 80):match.end() + 100]})
        for match in PROCESS_RE.finditer(text):
            if negated(text, match.start()):
                continue
            turns.append({"turn": index, "kind": "edit_process_claim",
                          "phrase": match.group(0),
                          "sentence": text[max(0, match.start() - 80):match.end() + 100]})
        for match in RELIABILITY_RE.finditer(text):
            turns.append({"turn": index, "kind": "reliability_candidate",
                          "phrase": match.group(0),
                          "sentence": text[max(0, match.start() - 80):match.end() + 100]})
    # Keep one item per (turn, kind, phrase), deterministic.
    seen, out = set(), []
    for item in turns:
        key = (item["turn"], item["kind"], item["phrase"])
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def analyse(batch: Path):
    import numpy as np
    manifest = load_json(batch / "manifest.json") or {}
    records = load_json(batch / "records.json") or []
    by_id = {row.get("sample_id"): row for row in records}
    failures = []
    failure_path = batch / "failures.jsonl"
    if failure_path.exists():
        for line in failure_path.read_text(encoding="utf-8").splitlines():
            try:
                failures.append(json.loads(line))
            except ValueError:
                pass
    samples = manifest.get("samples") or []
    missing = [s["sample_id"] for s in samples if s["sample_id"] not in by_id]
    extra = [sid for sid in by_id if sid not in {s["sample_id"] for s in samples}]
    rows = []
    for sample in samples:
        record = by_id.get(sample["sample_id"])
        if not record:
            continue
        record = dict(record)
        record["_sample"] = sample
        record["_overall"] = classify_overall(record)
        record["_truth_class"] = truth_class(record.get("task_truth") or {})
        record["_expected"] = expected_overall(record.get("task_truth") or {})
        trace = load_json(batch / record.get("trace_path", "" ), {}) or {}
        record["_trace"] = trace
        record["_matches"] = scan_turns(trace)
        record["_loc"] = localization(record, np) if record.get("mask_path") else None
        rows.append(record)

    def overall_stats(subset):
        n = len(subset)
        counts = collections.Counter(r["_overall"] for r in subset)
        correct = counts["correct"]
        wrong = counts["wrong"]
        abstain = counts["abstain"]
        no_conclusion = counts["no_conclusion"]
        concluded = correct + wrong
        return {
            "n": n, "correct": correct, "wrong": wrong, "abstain": abstain,
            "no_conclusion": no_conclusion,
            "accuracy": ratio(correct, n),
            "coverage": ratio(concluded, n),
            "concluded_accuracy": ratio(correct, concluded),
        }

    overall_by_class = {}
    for cls in ("real", "whole_generated", "local_tampered"):
        overall_by_class[cls] = overall_stats([r for r in rows if r["_truth_class"] == cls])
    overall_all = overall_stats(rows)
    class_recalls = [s["accuracy"] for s in overall_by_class.values()
                     if s["n"] and s["accuracy"] is not None]
    balanced_accuracy = round(sum(class_recalls) / len(class_recalls), 4) if class_recalls else None
    tp = sum(1 for r in rows if r["_expected"] == "Fake" and r.get("final_verdict") == "Fake")
    fp = sum(1 for r in rows if r["_expected"] == "Real" and r.get("final_verdict") == "Fake")
    fn = sum(1 for r in rows if r["_expected"] == "Fake" and r.get("final_verdict") != "Fake")
    fake_p, fake_r, fake_f1 = prf(tp, fp, fn)

    def definite(value, legal):
        return value in legal

    whole_stats = {}
    for label in ("Real", "AI-generated"):
        subset = [r for r in rows if (r.get("task_truth") or {}).get("whole_image_generation") == label]
        exact = sum(1 for r in subset if r.get("whole_image_generation") == label)
        definite_n = sum(1 for r in subset if r.get("whole_image_generation") in ("Real", "AI-generated"))
        whole_stats[label] = {
            "n": len(subset), "correct": exact,
            "accuracy": ratio(exact, len(subset)),
            "definite_coverage": ratio(definite_n, len(subset)),
            "uncertain": sum(1 for r in subset if r.get("whole_image_generation") == "Uncertain"),
            "missing": sum(1 for r in subset if r.get("whole_image_generation") not in
                            ("Real", "AI-generated", "Uncertain")),
        }
    whole_tp = sum(1 for r in rows if (r.get("task_truth") or {}).get("whole_image_generation") == "AI-generated"
                   and r.get("whole_image_generation") == "AI-generated")
    whole_fp = sum(1 for r in rows if (r.get("task_truth") or {}).get("whole_image_generation") == "Real"
                   and r.get("whole_image_generation") == "AI-generated")
    whole_fn = sum(1 for r in rows if (r.get("task_truth") or {}).get("whole_image_generation") == "AI-generated"
                   and r.get("whole_image_generation") != "AI-generated")
    whole_p, whole_r, whole_f1 = prf(whole_tp, whole_fp, whole_fn)
    whole_by_group = {}
    for group in ("Real",) + GROUPS:
        subset = [r for r in rows if r.get("group") == group]
        if not subset:
            continue
        whole_by_group[group] = {
            "n": len(subset),
            "correct": sum(1 for r in subset if r.get("whole_image_generation") ==
                           (r.get("task_truth") or {}).get("whole_image_generation")),
            "uncertain": sum(1 for r in subset if r.get("whole_image_generation") == "Uncertain"),
        }

    auto = [r for r in rows if r.get("family") == "AutoSplice_q95"]
    auto_forged = [r for r in auto if r.get("source_role") == "autosplice_forged"]
    auto_real = [r for r in auto if r.get("source_role") == "autosplice_authentic"]
    def local_count(subset, value):
        return sum(1 for r in subset if r.get("local_manipulation") == value)
    tp = local_count(auto_forged, "Tampered")
    fp = local_count(auto_real, "Tampered")
    fn = sum(1 for r in auto_forged if r.get("local_manipulation") != "Tampered")
    local_p, local_r, local_f1 = prf(tp, fp, fn)
    local_stats = {
        "forged": {
            "n": len(auto_forged),
            "tampered": tp,
            "not_tampered": local_count(auto_forged, "Not tampered"),
            "uncertain": local_count(auto_forged, "Uncertain"),
            "not_applicable": local_count(auto_forged, "Not applicable"),
            "missing": sum(1 for r in auto_forged if r.get("local_manipulation") not in
                            ("Tampered", "Not tampered", "Uncertain", "Not applicable")),
        },
        "authentic": {
            "n": len(auto_real),
            "tampered": fp,
            "not_tampered": local_count(auto_real, "Not tampered"),
            "uncertain": local_count(auto_real, "Uncertain"),
            "not_applicable": local_count(auto_real, "Not applicable"),
            "missing": sum(1 for r in auto_real if r.get("local_manipulation") not in
                            ("Tampered", "Not tampered", "Uncertain", "Not applicable")),
        },
        "precision": local_p, "recall": local_r, "f1": local_f1,
        "abstain": local_count(auto_forged, "Uncertain") + local_count(auto_real, "Uncertain"),
        "no_trufor": sum(1 for r in auto if not any(
            (e.get("source") == "trufor_expert") for e in r.get("evidence") or [])),
        "trufor_failed": sum(1 for r in auto if any(
            e.get("source") == "trufor_expert" and e.get("status") == "failed"
            for e in r.get("evidence") or [])),
    }

    genimage_generated = [r for r in rows if r.get("group") in GROUPS]
    generated_local = collections.Counter(r.get("local_manipulation") for r in genimage_generated)
    genimage_real = [r for r in rows if r.get("group") == "Real"]
    genimage_real_local = collections.Counter(r.get("local_manipulation") for r in genimage_real)

    loc_rows = [r for r in auto_forged if r.get("_loc") and not r["_loc"].get("error")]
    loc_with_bbox = [r for r in loc_rows if r["_loc"].get("n_valid_regions", 0) > 0]
    loc_metrics = {
        "forged_with_valid_mask": len(loc_rows),
        "forged_with_bbox": len(loc_with_bbox),
        "bbox_present_rate": ratio(len(loc_with_bbox), len(auto_forged)),
        "any_center_in_mask": sum(1 for r in loc_rows if r["_loc"].get("any_center_in_mask")),
        "iou_ge_0_1": sum(1 for r in loc_with_bbox if (r["_loc"].get("best_iou") or 0) >= 0.1),
        "iou_ge_0_3": sum(1 for r in loc_with_bbox if (r["_loc"].get("best_iou") or 0) >= 0.3),
        "iou_ge_0_5": sum(1 for r in loc_with_bbox if (r["_loc"].get("best_iou") or 0) >= 0.5),
        "mean_best_iou": round(sum((r["_loc"].get("best_iou") or 0) for r in loc_with_bbox) /
                               len(loc_with_bbox), 4) if loc_with_bbox else None,
        "mean_mask_coverage": round(sum((r["_loc"].get("best_mask_coverage") or 0)
                                        for r in loc_with_bbox) / len(loc_with_bbox), 4)
        if loc_with_bbox else None,
        "multi_variant_bases": sum(1 for r in auto_forged
                                   if (r.get("base_variant_count") or 0) > 1),
        "shared_mask_bases": sum(1 for r in auto_forged if r.get("mask_shared_by_base")),
    }

    first_tool = collections.Counter()
    requested = executed = valid_measurements = failed_calls = duplicates = 0
    for r in rows:
        evidence = r.get("evidence") or []
        first_tool[evidence[0].get("source") if evidence else "none"] += 1
        trace = r.get("_trace") or {}
        for call in (trace.get("metadata") or {}).get("model_calls") or []:
            text = str(call.get("raw_output") or "")
            requested += len(re.findall(r"<call_(?:probe|trufor)(?:\s|>)", text))
        executed += int((r.get("counters") or {}).get("expert_calls") or 0)
        valid_measurements += sum(1 for e in evidence if e.get("status", "measured") != "failed")
        failed_calls += sum(1 for e in evidence if e.get("status") == "failed")
        duplicates += int((r.get("counters") or {}).get("skipped_duplicate_calls") or 0)
    preclose = sum(1 for r in rows if (r.get("counters") or {}).get("preclose_checks"))
    coverage_corr = sum(1 for r in rows if (r.get("coverage_check") or {}).get("status") == "pass_after_correction")
    certainty_corr = sum(1 for r in rows if (r.get("certainty_check") or {}).get("status") == "pass_after_correction")
    coverage_failed = sum(1 for r in rows if (r.get("coverage_check") or {}).get("status") in ("failed", "not_corrected"))
    certainty_failed = sum(1 for r in rows if (r.get("certainty_check") or {}).get("status") in ("failed", "not_corrected"))
    untried_cited = sum(1 for r in rows if (r.get("counters") or {}).get("abstention_cited_untried_expert"))
    definite_without = sum(1 for r in rows if (r.get("coverage_check") or {}).get("issues")
                           and any(i.get("code") in ("whole_image_without_probe",
                                                     "local_manipulation_without_trufor")
                                   for i in (r.get("coverage_check") or {}).get("issues") or []))
    single_expert = sum(1 for r in rows if len({e.get("source") for e in r.get("evidence") or []
                                                if e.get("status", "measured") != "failed"}) == 1)
    call_stats = {
        "first_tool": dict(first_tool),
        "requested_calls": requested, "executed_calls": executed,
        "valid_measurements": valid_measurements, "failed_calls": failed_calls,
        "duplicate_requests": duplicates,
        "preclose_sessions": preclose,
        "coverage_corrections": coverage_corr, "coverage_failed": coverage_failed,
        "certainty_corrections": certainty_corr, "certainty_failed": certainty_failed,
        "abstention_cited_untried": untried_cited,
        "definite_without_measurement": definite_without,
        "single_expert_sessions": single_expert,
    }

    certainty_counts = collections.Counter(r.get("certainty") for r in rows)
    certainty_accuracy = {}
    for value in ("High", "Medium", "Low", None):
        subset = [r for r in rows if r.get("certainty") == value]
        if subset:
            certainty_accuracy["null" if value is None else value] = {
                "n": len(subset),
                "overall_correct": sum(1 for r in subset if r["_overall"] == "correct"),
                "overall_accuracy": ratio(sum(1 for r in subset if r["_overall"] == "correct"), len(subset)),
                "whole_correct": sum(1 for r in subset if r.get("whole_image_generation") ==
                                     (r.get("task_truth") or {}).get("whole_image_generation")),
                "local_correct": sum(1 for r in subset if r.get("local_manipulation") ==
                                     (r.get("task_truth") or {}).get("local_manipulation")),
            }
    high_conflict = [{
        "sample_id": r.get("sample_id"),
        "page": r.get("page"),
        "report": r.get("report"),
        "matches": r.get("_matches"),
    } for r in rows if r.get("certainty") == "High"
        and LIMITATION_RE.search(str(r.get("report") or ""))]

    quality = {
        "three_panel_sessions": sum(1 for r in rows if any(
            e.get("source") == "trufor_expert" and e.get("artifact_layout")
            and e.get("diagnostic_region_image") for e in r.get("evidence") or [])),
        "layout_recorded_sessions": sum(1 for r in rows if any(
            e.get("source") == "trufor_expert" and e.get("model_image_layout")
            for e in r.get("evidence") or [])),
        "bbox_flag_sessions": sum(1 for r in rows if r.get("suspicious_region_review_flags")),
        "mutual_candidates": sum(1 for r in rows for m in r.get("_matches") or []
                                 if m["kind"] == "mutual_wording"),
        "edit_process_candidates": sum(1 for r in rows for m in r.get("_matches") or []
                                       if m["kind"] == "edit_process_claim"),
        "reliability_candidates": sum(1 for r in rows for m in r.get("_matches") or []
                                      if m["kind"] == "reliability_candidate"),
        "high_limitation_candidates": len(high_conflict),
    }

    # Priority queue: every error, abstention, no-conclusion, tool failure,
    # correction; then a fixed random sample of overall-correct + High.
    queue = {}
    for r in rows:
        reasons = []
        if r["_overall"] in ("wrong", "abstain", "no_conclusion"):
            reasons.append(r["_overall"])
        if (r.get("counters") or {}).get("failed_experts"):
            reasons.append("tool_failure")
        if (r.get("coverage_check") or {}).get("status") in ("pass_after_correction", "failed", "not_corrected"):
            reasons.append("coverage_correction")
        if (r.get("certainty_check") or {}).get("status") in ("pass_after_correction", "failed", "not_corrected"):
            reasons.append("certainty_correction")
        if reasons:
            queue[r["sample_id"]] = {
                "reasons": reasons, "page": r.get("page"),
                "matches": r.get("_matches") or [],
                "localization": r.get("_loc"),
            }
    correct_high = sorted([r for r in rows if r["_overall"] == "correct"
                           and r.get("certainty") == "High"], key=lambda r: r["sample_id"])
    rng = random.Random(f"{manifest.get('seed', 20261110)}:review:correct_high")
    for r in rng.sample(correct_high, min(20, len(correct_high))):
        item = queue.setdefault(r["sample_id"], {
            "reasons": [], "page": r.get("page"), "matches": r.get("_matches") or [],
            "localization": r.get("_loc")})
        item["reasons"].append("correct_high_sample")
    review_queue = [{"sample_id": sid, **value} for sid, value in sorted(queue.items())]

    # The trace payloads are already on disk; keep report.json compact.
    for row in rows:
        row.pop("_trace", None)
        row.pop("_sample", None)

    return {
        "batch": str(batch),
        "manifest": {
            "seed": manifest.get("seed"), "counts": manifest.get("counts"),
            "sha256": hashlib.sha256((batch / "manifest.json").read_bytes()).hexdigest()
            if (batch / "manifest.json").exists() else None,
        },
        "sessions": {
            "expected": len(samples), "records": len(rows),
            "missing": missing, "extra": extra, "failures": failures,
        },
        "overall": {"all": overall_all, "by_truth_class": overall_by_class,
                    "balanced_accuracy": balanced_accuracy,
                    "fake_precision": fake_p, "fake_recall": fake_r, "fake_f1": fake_f1},
        "whole": {"by_truth": whole_stats, "by_group": whole_by_group,
                  "ai_precision": whole_p, "ai_recall": whole_r, "ai_f1": whole_f1},
        "local": {"autosplice": local_stats,
                  "genimage_generated_outputs": dict(generated_local),
                  "genimage_real_outputs": dict(genimage_real_local),
                  "localization": loc_metrics},
        "calls": call_stats,
        "certainty": {"distribution": {("null" if k is None else k): v
                                       for k, v in certainty_counts.items()},
                      "accuracy_by_value": certainty_accuracy,
                      "high_limitation_candidates": high_conflict},
        "quality": quality,
        "review_queue": review_queue,
        "records": rows,
    }


def fmt(value):
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(report: dict) -> str:
    o = report["overall"]
    lines = [
        "# g8 200-source development evaluation — automatic report",
        "",
        f"- batch: `{report['batch']}`",
        f"- manifest seed: `{report['manifest'].get('seed')}`；"
        f"sessions: {report['sessions']['records']}/{report['sessions']['expected']}",
        f"- manifest sha256: `{report['manifest'].get('sha256')}`",
        "",
        "> 自动结果描述性报告，不是人工验收；Uncertain 不计为 Fake 正例。",
        "",
        "## 1. 总体分类（按真实类别）",
        "",
        "| 真值类别 | n | 正确 | 错误 | 弃权 | 未形成结论 | 准确率 | 覆盖率 | 已结案准确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {"real": "真实图（Real）", "whole_generated": "整图生成",
              "local_tampered": "局部篡改"}
    for key in ("real", "whole_generated", "local_tampered"):
        s = o["by_truth_class"][key]
        lines.append(
            f"| {labels[key]} | {s['n']} | {s['correct']} | {s['wrong']} | "
            f"{s['abstain']} | {s['no_conclusion']} | {fmt(s['accuracy'])} | "
            f"{fmt(s['coverage'])} | {fmt(s['concluded_accuracy'])} |")
    a = o["all"]
    lines += [
        "",
        f"- 总体准确率：**{fmt(a['accuracy'])}**（{a['correct']}/{a['n']}）",
        f"- 平衡准确率（三类 recall 均值）：**{fmt(o['balanced_accuracy'])}**",
        f"- 覆盖率：**{fmt(a['coverage'])}**；已结案样本准确率：**{fmt(a['concluded_accuracy'])}**",
        f"- Fake 类（整图生成 + 局部篡改）precision={fmt(o['fake_precision'])}，"
        f"recall={fmt(o['fake_recall'])}，F1={fmt(o['fake_f1'])}；"
        "Uncertain 不计入 TP/FP/FN 的正例。",
        "",
        "## 2. 整图生成分项",
        "",
        "| 真值 | n | 正确 | 准确率 | 确定测量覆盖 | Uncertain | 缺失/非法 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("Real", "AI-generated"):
        s = report["whole"]["by_truth"][key]
        lines.append(f"| {key} | {s['n']} | {s['correct']} | {fmt(s['accuracy'])} | "
                     f"{fmt(s['definite_coverage'])} | {s['uncertain']} | {s['missing']} |")
    lines += [
        "",
        f"- AI-generated precision={fmt(report['whole']['ai_precision'])}，"
        f"recall={fmt(report['whole']['ai_recall'])}，F1={fmt(report['whole']['ai_f1'])}。",
        "",
        "| GenImage 组 | n | 整图正确 | Uncertain |",
        "| --- | ---: | ---: | ---: |",
    ]
    for group, s in report["whole"]["by_group"].items():
        lines.append(f"| {group} | {s['n']} | {s['correct']} | {s['uncertain']} |")
    lines += [
        "",
        "## 3. 局部编辑分项（AutoSplice）",
        "",
        "| 真值 | n | Tampered | Not tampered | Uncertain | Not applicable | 缺失 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in (("forged", "真实篡改"), ("authentic", "真实真图")):
        s = report["local"]["autosplice"][key]
        lines.append(f"| {label} | {s['n']} | {s['tampered']} | {s['not_tampered']} | "
                     f"{s['uncertain']} | {s['not_applicable']} | {s['missing']} |")
    l = report["local"]["autosplice"]
    lines += [
        "",
        f"- 图像级局部：precision={fmt(l['precision'])}，recall={fmt(l['recall'])}，"
        f"F1={fmt(l['f1'])}；弃权 {l['abstain']}；未调用 TruFor {l['no_trufor']}；"
        f"TruFor 失败 {l['trufor_failed']}。",
        f"- GenImage 纯生成图 local 输出：{report['local']['genimage_generated_outputs']}；"
        "真值为 Not applicable，按该口径单独评价，不混入 AutoSplice 局部 P/R。",
        f"- GenImage Real local 输出：{report['local']['genimage_real_outputs']}。",
        "",
        "### 区域定位（AutoSplice，base 共享 mask）",
        "",
        f"- 有 mask 的伪造源：{report['local']['localization']['forged_with_valid_mask']}；"
        f"输出 bbox：{report['local']['localization']['forged_with_bbox']}"
        f"（{fmt(report['local']['localization']['bbox_present_rate'])}）。",
        f"- bbox 中心落 mask：{report['local']['localization']['any_center_in_mask']}；"
        f"IoU≥0.1/0.3/0.5：{report['local']['localization']['iou_ge_0_1']}/"
        f"{report['local']['localization']['iou_ge_0_3']}/"
        f"{report['local']['localization']['iou_ge_0_5']}；"
        f"mean best IoU={fmt(report['local']['localization']['mean_best_iou'])}，"
        f"mean mask coverage={fmt(report['local']['localization']['mean_mask_coverage'])}。",
        f"- 共享 mask 的伪造源 {report['local']['localization']['shared_mask_bases']} 个；"
        f"同 base 多 variant（本批只选一个输入）{report['local']['localization']['multi_variant_bases']} 个。",
        "",
        "## 4. 调用行为",
        "",
        f"- 首个工具：{report['calls']['first_tool']}",
        f"- 请求调用 {report['calls']['requested_calls']}，实际执行 "
        f"{report['calls']['executed_calls']}，有效测量 {report['calls']['valid_measurements']}，"
        f"失败 {report['calls']['failed_calls']}，重复请求 {report['calls']['duplicate_requests']}。",
        f"- 结案前补测 {report['calls']['preclose_sessions']}；覆盖纠正 "
        f"{report['calls']['coverage_corrections']}（失败 {report['calls']['coverage_failed']}）；"
        f"certainty 纠正 {report['calls']['certainty_corrections']}"
        f"（失败 {report['calls']['certainty_failed']}）。",
        f"- 单专家会话 {report['calls']['single_expert_sessions']}；弃权仍引用未尝试专家 "
        f"{report['calls']['abstention_cited_untried']}；未测分项越界 "
        f"{report['calls']['definite_without_measurement']}。",
        "",
        "## 5. 解释质量候选",
        "",
        "> 自动命中只是候选；`review_queue.json`/`REVIEW_GUIDE.md` 给出原句和轮次，"
        "需要人工按原对话核对。",
        f"- 三图会话 {report['quality']['three_panel_sessions']}；layout 记录 "
        f"{report['quality']['layout_recorded_sessions']}；框核验标记会话 "
        f"{report['quality']['bbox_flag_sessions']}。",
        f"- 跨任务相互验证候选 {report['quality']['mutual_candidates']}；"
        f"编辑过程断言候选 {report['quality']['edit_process_candidates']}；"
        f"可靠性可能误读候选 {report['quality']['reliability_candidates']}。",
        "",
        "## 6. Certainty",
        "",
        f"- 分布：{report['certainty']['distribution']}",
        "",
        "| certainty | n | 总体正确 | 总体准确率 | 整图正确 | 局部正确 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("High", "Medium", "Low", "null"):
        s = report["certainty"]["accuracy_by_value"].get(key)
        if s:
            lines.append(f"| {key} | {s['n']} | {s['overall_correct']} | "
                         f"{fmt(s['overall_accuracy'])} | {s['whole_correct']} | {s['local_correct']} |")
    lines += [
        "",
        f"- High 且报告含“候选/未校准/缺失/失败/无法确认”等限制的候选："
        f"{report['quality']['high_limitation_candidates']} 条（见 review queue）。",
        "",
        "## 7. 失败与复核入口",
        "",
        f"- 缺失记录：{len(report['sessions']['missing'])}；额外记录："
        f"{len(report['sessions']['extra'])}；运行失败记录：{len(report['sessions']['failures'])}。",
        f"- 优先复核：错误、弃权、未形成结论、工具失败、纠正案例，再固定随机抽查 "
        f"{sum(1 for x in report['review_queue'] if 'correct_high_sample' in x['reasons'])} "
        "条总体正确且 High 的会话。",
        "- `index.html`：原始对话、投送图片、三图、预测框、专家结果、纠正、停止原因。",
        "- `review_records.csv` / `records.json`：可导出审核记录。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=str(DEFAULT_BATCH))
    args = parser.parse_args()
    batch = Path(args.batch).resolve()
    report = analyse(batch)
    (batch / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (batch / "SUMMARY.md").write_text(markdown(report), encoding="utf-8")
    queue = report["review_queue"]
    (batch / "review_queue.json").write_text(
        json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    guide = [
        "# g8 200-source development evaluation — 人工复核指引", "",
        "> 自动统计不是验收结论；`index.html` 和原始 trace 是权威。",
        "", "## 优先复核顺序", "",
        "1. 全部总体错误、弃权、未形成结论。",
        "2. 全部工具失败和 coverage/certainty 纠正案例。",
        "3. 固定随机抽查约 20 条“总体正确且 certainty=High”的会话，检查结论正确但解释错误。",
        "", f"固定队列见 `review_queue.json`（{len(queue)} 条）；每条包含 reasons。", "",
        "## 重点", "",
        "- 总体误判：真实图被 PROBE 高分带成 Fake；局部篡改被保守写成 Uncertain。",
        "- 局部任务：AutoSplice 的 Tampered/Not tampered/Uncertain/未调用/失败分开看；",
        "  GenImage 纯生成图的 local Not applicable 按单独口径看。",
        "- 三图同位置、可靠性图背景红是否被误套、框文矛盾、跨任务相互验证措辞、编辑过程断言。",
        "- 报告的主要限制是否与 certainty 协调；High 是否机械集中。",
        "- 区域定位只在使用对应 mask 时评价；本批 AutoSplice mask 为 base 共享，且每源只选一个 variant。",
        "", "## 导出", "",
        "- `records.json` / `review_records.csv`：逐样本结论与审计字段。",
        "- `SUMMARY.md`：自动统计。",
        "- `index.html`：逐会话工作台。",
    ]
    (batch / "REVIEW_GUIDE.md").write_text("\n".join(guide) + "\n", encoding="utf-8")
    print(json.dumps({"batch": str(batch), "records": report["sessions"]["records"],
                      "overall": report["overall"], "review_queue": len(queue)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
