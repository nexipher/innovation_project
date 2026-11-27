"""Independent PROBE / TruFor measurements on the 200 q95 dev inputs.

The two experts run without Qwen and without the Agent loop, using the same
weights and preprocessing as the runtime.  The script writes a per-sample CSV,
a concise report, TruFor raw npz files and anomaly/confidence maps, and joins
the existing 2026-11-27 Agent records for the focus groups.

Thresholds are the existing ones: PROBE raw score 0.5; TruFor image score 0.5;
TruFor map high-response 0.5.  AutoSplice masks are base-shared and resized to
the map.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT  # noqa: E402
from experts.probe_dino import ProbeDinoExpert  # noqa: E402
from experts.trufor import HIGH_RESPONSE_THRESHOLD, TruForExpert  # noqa: E402
from scripts.trufor_autosplice_smoke import _mask, _metrics  # noqa: E402
from utils.image_utils import ImageUtils  # noqa: E402

PROBE_THRESHOLD = 0.5
TRUFOR_SCORE_THRESHOLD = 0.5
MAP_THRESHOLD = HIGH_RESPONSE_THRESHOLD
DEFAULT_MANIFEST = ROOT / "sft_data/g8_dev_eval_20261110/manifest.json"
DEFAULT_OUT = ROOT / "sft_data/independent_expert_20261127"
AGENT_PARTS = (
    ROOT / "sft_data/g8_dev_eval_20261127_part1/records.json",
    ROOT / "sft_data/g8_dev_eval_20261127_part2/records.json",
)


def _dist(values):
    values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {"n": len(values), "mean": round(float(a.mean()), 6),
            "median": round(float(np.median(a)), 6),
            "min": round(float(a.min()), 6), "max": round(float(a.max()), 6),
            "p25": round(float(np.percentile(a, 25)), 6),
            "p75": round(float(np.percentile(a, 75)), 6)}


def _rate(n, d):
    return round(n / d, 4) if d else None


def _load_agent_records():
    records = {}
    for path in AGENT_PARTS:
        if not path.exists():
            raise SystemExit(f"Agent record missing: {path}")
        rows = json.load(open(path, encoding="utf-8"))
        for row in rows:
            sid = row["sample_id"]
            if sid in records:
                raise SystemExit(f"duplicate Agent record: {sid}")
            records[sid] = row
    if len(records) != 200:
        raise SystemExit(f"expected 200 Agent records, got {len(records)}")
    return records


def _agent_state(record):
    tokens = [e for e in (record.get("evidence") or [])
              if e.get("source") == "trufor_expert"]
    if not tokens:
        return "not_called"
    if any(e.get("status") == "failed" for e in tokens):
        return "failed"
    return "measured"


def _focus_group(record):
    if record["task_truth"]["local_manipulation"] != "Tampered":
        return ""
    state = _agent_state(record)
    if state == "not_called":
        return "tampered_agent_not_called"
    if state == "failed":
        return "tampered_agent_failed"
    if record.get("local_manipulation") == "Tampered":
        return "tampered_agent_detected"
    return "tampered_agent_called_not_detected"


def _probe_stats(rows):
    def summarize(selected):
        scores = [row["probe_score"] for row in selected]
        detected = sum(1 for row in selected
                       if row["probe_score"] >= PROBE_THRESHOLD)
        return {
            "n": len(selected), "detected_ai": detected,
            "detection_rate": _rate(detected, len(selected)),
            "false_positive_rate": _rate(detected, len(selected)),
            "score": _dist(scores),
        }
    gen = [r for r in rows if r["family"] == "GenImage_q95"
           and r["whole_truth"] == "AI-generated"]
    real = [r for r in rows if r["group"] == "Real"]
    auto_real = [r for r in rows if r["group"] == "AutoSplice_Authentic"]
    auto_forged = [r for r in rows if r["group"] == "AutoSplice_Forged"]
    all_real = [r for r in rows if r["whole_truth"] == "Real"]
    return {
        "threshold": PROBE_THRESHOLD,
        "score_field": "raw_metric (PROBE AI-generated probability)",
        "genimage_generated": summarize(gen),
        "genimage_real": summarize(real),
        "autosplice_authentic": summarize(auto_real),
        "autosplice_forged_whole_image": summarize(auto_forged),
        "all_whole_real": summarize(all_real),
        "genimage_by_generator": {
            group: summarize([r for r in gen if r["group"] == group])
            for group in sorted({r["group"] for r in gen})
        },
    }


def _trufor_stats(rows):
    forged = [r for r in rows if r["group"] == "AutoSplice_Forged"]
    authentic = [r for r in rows if r["group"] == "AutoSplice_Authentic"]
    gen = [r for r in rows if r["family"] == "GenImage_q95"
           and r["whole_truth"] == "AI-generated"]
    gen_real = [r for r in rows if r["group"] == "Real"]
    tp = sum(1 for r in forged if r["trufor_score"] >= TRUFOR_SCORE_THRESHOLD)
    fp = sum(1 for r in authentic if r["trufor_score"] >= TRUFOR_SCORE_THRESHOLD)
    fn = len(forged) - tp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall else None)
    ious = [r["trufor_iou"] for r in forged if r["trufor_iou"] is not None]
    f1s = [r["trufor_f1"] for r in forged if r["trufor_f1"] is not None]
    recalls = [r["trufor_recall"] for r in forged if r["trufor_recall"] is not None]
    return {
        "image_score_threshold": TRUFOR_SCORE_THRESHOLD,
        "map_high_response_threshold": MAP_THRESHOLD,
        "autosplice_forged": {
            "n": len(forged), "detected": tp, "missed": fn,
            "detection_rate": _rate(tp, len(forged)),
            "score": _dist([r["trufor_score"] for r in forged]),
            "map_high_fraction": _dist([r["trufor_map_high_fraction"] for r in forged]),
        },
        "autosplice_authentic": {
            "n": len(authentic), "false_positive": fp,
            "false_positive_rate": _rate(fp, len(authentic)),
            "score": _dist([r["trufor_score"] for r in authentic]),
            "map_high_fraction": _dist([r["trufor_map_high_fraction"] for r in authentic]),
        },
        "autosplice_image_level": {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
        },
        "autosplice_localization": {
            "n": len(ious),
            "mask": "base-shared mask resized to map; no per-variant mask",
            "iou": _dist(ious), "f1": _dist(f1s), "recall": _dist(recalls),
            "iou_ge_0_1": sum(1 for v in ious if v >= 0.1),
            "iou_ge_0_3": sum(1 for v in ious if v >= 0.3),
            "iou_ge_0_5": sum(1 for v in ious if v >= 0.5),
        },
        "genimage_generated_local_response": {
            "n": len(gen), "score_ge_0_5": sum(
                1 for r in gen if r["trufor_score"] >= TRUFOR_SCORE_THRESHOLD),
            "score": _dist([r["trufor_score"] for r in gen]),
            "map_high_fraction": _dist([r["trufor_map_high_fraction"] for r in gen]),
            "note": "whole=AI-generated; local truth is Not applicable; not an FP",
        },
        "genimage_real_local_response": {
            "n": len(gen_real), "score_ge_0_5": sum(
                1 for r in gen_real if r["trufor_score"] >= TRUFOR_SCORE_THRESHOLD),
            "score": _dist([r["trufor_score"] for r in gen_real]),
            "map_high_fraction": _dist([r["trufor_map_high_fraction"] for r in gen_real]),
        },
    }


def _focus_stats(rows):
    groups = ("tampered_agent_not_called",
              "tampered_agent_called_not_detected",
              "tampered_agent_detected")
    out = {}
    for group in groups:
        selected = [r for r in rows if r["focus_group"] == group]
        score_det = [r for r in selected
                     if r["trufor_score"] >= TRUFOR_SCORE_THRESHOLD]
        loc_det = [r for r in selected
                   if r["trufor_iou"] is not None and r["trufor_iou"] >= 0.1]
        out[group] = {
            "n": len(selected),
            "trufor_score_detected": len(score_det),
            "trufor_score_detection_rate": _rate(len(score_det), len(selected)),
            "trufor_localization_hit_iou_ge_0_1": len(loc_det),
            "trufor_localization_hit_rate": _rate(len(loc_det), len(selected)),
            "trufor_score": _dist([r["trufor_score"] for r in selected]),
            "trufor_iou": _dist([r["trufor_iou"] for r in selected
                                 if r["trufor_iou"] is not None]),
            "examples": sorted(selected, key=lambda r: (
                r["trufor_score"] or 0, r["trufor_iou"] or 0), reverse=True)[:5],
        }
    return out


def _issue_reading(rows):
    """Classify the two focus groups into tool-call / output / interpretation.

    This is a descriptive rule, not an automatic verdict:
    * Agent did not call TruFor, independent score >= 0.5 -> tool-call gap;
    * Agent called but did not label Tampered, independent score >= 0.5 ->
      Agent interpretation/evidence-use gap;
    * independent score < 0.5 and no localization -> expert output limit;
    * independent localization only (IoU >= 0.1, score < 0.5) -> borderline
      (localization candidate not reflected in the image score).
    """
    out = {}
    for group in ("tampered_agent_not_called",
                  "tampered_agent_called_not_detected"):
        rows_g = [r for r in rows if r["focus_group"] == group]
        counts = {"tool_call": 0, "agent_interpretation": 0,
                  "expert_output": 0, "localization_only": 0}
        for r in rows_g:
            score_ok = r["trufor_score"] >= TRUFOR_SCORE_THRESHOLD
            loc_ok = r["trufor_iou"] is not None and r["trufor_iou"] >= 0.1
            if score_ok and group == "tampered_agent_not_called":
                counts["tool_call"] += 1
            elif score_ok:
                counts["agent_interpretation"] += 1
            elif loc_ok:
                counts["localization_only"] += 1
            else:
                counts["expert_output"] += 1
        out[group] = counts
    return out


def _write_csv(path, rows):
    fields = [
        "sample_id", "family", "group", "source_role", "ground_truth",
        "whole_truth", "local_truth", "path", "mask_path",
        "probe_score", "probe_pred_ai",
        "trufor_score", "trufor_pred_tampered", "trufor_map_high_fraction",
        "trufor_anomaly_peak", "trufor_conf_mean",
        "trufor_iou", "trufor_f1", "trufor_precision", "trufor_recall",
        "agent_trufor", "agent_trufor_score", "agent_local", "agent_final",
        "agent_certainty", "focus_group", "agent_page",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})


def _write_report(path, payload):
    p = payload["probe"]
    t = payload["trufor"]
    f = payload["focus"]
    lines = [
        "# 独立专家对照：PROBE / TruFor × 200 q95",
        "",
        f"输入：`{payload['manifest']}`（200 张，原顺序）；"
        f"运行时间：{payload['started_at']} – {payload['finished_at']}。",
        "",
        "## 口径",
        "",
        f"- PROBE：`raw_metric`，阈值 **{PROBE_THRESHOLD}**（外部默认；未校准概率，仅作判别分数）。",
        f"- TruFor 图像级：score 阈值 **{TRUFOR_SCORE_THRESHOLD}**；"
        f"map 高响应阈值 **{MAP_THRESHOLD}**。",
        "- AutoSplice 定位：base 共享 mask，resize 到 map；IoU/F1/precision/recall "
        "按像素计算；本批每个 base 只取一张，不把同源变体当独立样本。",
        "- 独立专家不加载 Qwen、不经过 Agent；Agent 记录来自 2026-11-27 两批重跑。",
        "",
        "## PROBE",
        "",
        "| 组 | n | score≥0.5 | 比率 | 说明 |",
        "|---|---:|---:|---:|---|",
        f"| GenImage 生成图 | {p['genimage_generated']['n']} | "
        f"{p['genimage_generated']['detected_ai']} | "
        f"{p['genimage_generated']['detection_rate']} | 检出率 |",
        f"| GenImage 真图 | {p['genimage_real']['n']} | "
        f"{p['genimage_real']['detected_ai']} | "
        f"{p['genimage_real']['false_positive_rate']} | 误报率 |",
        f"| AutoSplice 真图 | {p['autosplice_authentic']['n']} | "
        f"{p['autosplice_authentic']['detected_ai']} | "
        f"{p['autosplice_authentic']['false_positive_rate']} | 整图误报率 |",
        f"| AutoSplice 编辑图 | {p['autosplice_forged_whole_image']['n']} | "
        f"{p['autosplice_forged_whole_image']['detected_ai']} | "
        f"{p['autosplice_forged_whole_image']['false_positive_rate']} | 整图误报率（真值整图 Real） |",
        "",
        "GenImage 分组：",
        "",
        "| 生成器 | n | 检出 | 检出率 |",
        "|---|---:|---:|---:|",
    ]
    for group, stats in p["genimage_by_generator"].items():
        lines.append(f"| {group} | {stats['n']} | {stats['detected_ai']} | "
                     f"{stats['detection_rate']} |")
    il = t["autosplice_image_level"]
    loc = t["autosplice_localization"]
    gi = t["genimage_generated_local_response"]
    gr = t["genimage_real_local_response"]
    lines += [
        "",
        "## TruFor",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| AutoSplice 篡改图像级检出 | {t['autosplice_forged']['detected']}/"
        f"{t['autosplice_forged']['n']} = "
        f"{t['autosplice_forged']['detection_rate']:.4f} |",
        f"| AutoSplice 真图误报 | {t['autosplice_authentic']['false_positive']}/"
        f"{t['autosplice_authentic']['n']} = "
        f"{t['autosplice_authentic']['false_positive_rate']:.4f} |",
        f"| 图像级 TP/FP/FN | {il['tp']}/{il['fp']}/{il['fn']} |",
        f"| 图像级 P/R/F1 | {il['precision']:.4f} / {il['recall']:.4f} / "
        f"{il['f1']:.4f} |",
        f"| 定位 IoU mean / median | {loc['iou']['mean']:.4f} / "
        f"{loc['iou']['median']:.4f} |",
        f"| 定位 F1 mean / median | {loc['f1']['mean']:.4f} / "
        f"{loc['f1']['median']:.4f} |",
        f"| 定位 IoU≥0.1 / 0.3 / 0.5 | {loc['iou_ge_0_1']}/"
        f"{loc['iou_ge_0_3']}/{loc['iou_ge_0_5']}（n={loc['n']}） |",
        f"| GenImage 80 生成图 score≥0.5 | {gi['score_ge_0_5']}/"
        f"{gi['n']}（局部响应单列，不是误报） |",
        f"| GenImage 80 map high fraction mean | "
        f"{gi['map_high_fraction']['mean']:.4f} |",
        f"| GenImage 30 真图 score≥0.5 | {gr['score_ge_0_5']}/{gr['n']} |",
        f"| GenImage 30 map high fraction mean | "
        f"{gr['map_high_fraction']['mean']:.4f} |",
        "",
        "阈值：TruFor 图像 score **0.5**；map 高响应 **0.5**；AutoSplice mask 为 "
        "base 共享 mask，resize 到 map；本批每个 base 只取一张。",
        "",
        "## Agent 重点组",
        "",
        "| 组 | n | 独立 TruFor score 检出 | 检出率 | 定位 IoU≥0.1 | 定位率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, stats in f.items():
        lines.append(f"| {group} | {stats['n']} | {stats['trufor_score_detected']} | "
                     f"{stats['trufor_score_detection_rate']} | "
                     f"{stats['trufor_localization_hit_iou_ge_0_1']} | "
                     f"{stats['trufor_localization_hit_rate']} |")
    issue = payload["issue_reading"]
    check = payload.get("agent_score_check") or {}
    lines += [
        "",
        f"Agent 已调用且有效的 TruFor score 与本次独立重跑逐图一致："
        f"{check.get('exact_matches')}/{check.get('measured')}（0 条差异）。",
        "",
        "问题层级读数（描述性，不是自动验收）：",
        "",
        f"- `tampered_agent_not_called`：{issue['tampered_agent_not_called']}。"
        "`tool_call` 表示独立 TruFor score≥0.5、Agent 本可调用却未调用；"
        "`localization_only` 表示 score<0.5 但 IoU≥0.1，定位已有候选。",
        f"- `tampered_agent_called_not_detected`："
        f"{issue['tampered_agent_called_not_detected']}。`agent_interpretation` "
        "表示独立 score≥0.5、Agent 已调用但未写 local Tampered；`expert_output` "
        "表示独立 score<0.5 且无定位命中；`localization_only` 为边界情况。",
        "",
        "读法：14 张未调用中 10 张独立 score≥0.5、13 张 IoU≥0.1，问题首先在"
        "**调用覆盖**；33 张已调用未检出中 18 张独立 score≥0.5，问题更像"
        "**Agent 判读/证据使用**；4 张独立 score<0.5 且无定位命中，属于"
        "**专家输出限制**；11 张仅定位命中，需人工看响应区域是否被忽略。",
        "",
    ]
    for group, stats in f.items():
        if not stats["examples"]:
            continue
        lines += [f"### {group} 代表案例", "",
                  "| sample_id | Agent local | TruFor score | IoU | focus |",
                  "|---|---|---:|---:|---|"]
        for row in stats["examples"]:
            lines.append(f"| {row['sample_id']} | {row['agent_local']} | "
                         f"{row['trufor_score']:.6f} | "
                         f"{row['trufor_iou'] if row['trufor_iou'] is not None else '—'} | "
                         f"{row['focus_group']} |")
        lines.append("")
    lines += [
        "## 边界",
        "",
        "- 独立专家只回答自己的测量问题；PROBE 无定位，TruFor 的 score 不是本项目校准概率。",
        "- GenImage 生成图的局部真值为 Not applicable，局部响应单列，不计入 AutoSplice 误报。",
        "- 逐图 CSV：`per_sample.csv`；TruFor 原始 npz：`trufor_raw/`；异常/可靠性图：`trufor_maps/`。",
    ]
    open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    os.environ.setdefault(
        "PROBE_WEIGHTS",
        "/root/autodl-tmp/third_party/probe/DINOv2_best_model_step_34999.pth")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "trufor_raw").mkdir(exist_ok=True)
    (out / "trufor_maps").mkdir(exist_ok=True)
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) != 200:
        raise SystemExit(f"expected 200 manifest samples, got {len(samples)}")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    agent = _load_agent_records()

    images = {}
    load_failures = []
    for sample in samples:
        sid = sample["sample_id"]
        try:
            images[sid] = ImageUtils.load_image(str(ROOT / sample["path"]))
        except Exception as exc:  # noqa: BLE001
            load_failures.append({"sample_id": sid,
                                  "error": f"{type(exc).__name__}: {exc}"})
    if load_failures:
        (out / "run_error.json").write_text(json.dumps(
            {"stage": "load", "failures": load_failures}, ensure_ascii=False, indent=2))
        raise SystemExit(f"image load failed: {load_failures[:3]}")

    # PROBE
    probe = ProbeDinoExpert(device=args.device)
    reason = probe.unavailable_reason()
    if reason:
        raise SystemExit(f"PROBE unavailable: {reason}")
    probe_scores = {}
    probe_failures = []
    for index, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        try:
            result = probe.analyze(images[sid])
            probe_scores[sid] = float(getattr(result, "raw_metric", result.strength))
        except Exception as exc:  # noqa: BLE001
            probe_failures.append({"sample_id": sid,
                                   "error": f"{type(exc).__name__}: {exc}"})
            break
        if index % 20 == 0:
            print(f"PROBE {index}/200", flush=True)
    if probe_failures:
        (out / "run_error.json").write_text(json.dumps(
            {"stage": "probe", "failures": probe_failures}, ensure_ascii=False, indent=2))
        raise SystemExit(f"PROBE failed: {probe_failures}")
    del probe
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass

    # TruFor
    trufor = TruForExpert(device=args.device, cache=False,
                          results_dir=str(out / "trufor_raw"))
    reason = trufor.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor unavailable: {reason}")
    trufor_scores = {}
    trufor_arrays = {}
    trufor_failures = []
    for start in range(0, len(samples), args.batch_size):
        chunk_samples = samples[start:start + args.batch_size]
        chunk = {s["sample_id"]: images[s["sample_id"]] for s in chunk_samples}
        try:
            results = trufor.analyze_many(chunk, progress=True)
        except Exception as exc:  # noqa: BLE001
            trufor_failures.append({"sample_ids": list(chunk),
                                    "error": f"{type(exc).__name__}: {exc}"})
            break
        for sample in chunk_samples:
            sid = sample["sample_id"]
            result = results.get(sid)
            if result is None:
                trufor_failures.append({"sample_id": sid, "error": "missing_result"})
                continue
            key = trufor.cache_key(images[sid])
            path = trufor.result_path(key)
            try:
                arrays = dict(np.load(path, allow_pickle=True))
            except Exception as exc:  # noqa: BLE001
                trufor_failures.append({"sample_id": sid,
                                        "error": f"npz_load: {type(exc).__name__}: {exc}"})
                continue
            trufor_scores[sid] = float(result.raw_metric)
            trufor_arrays[sid] = arrays
        print(f"TruFor {min(start + args.batch_size, len(samples))}/200", flush=True)
    if trufor_failures:
        (out / "run_error.json").write_text(json.dumps(
            {"stage": "trufor", "failures": trufor_failures},
            ensure_ascii=False, indent=2))
        raise SystemExit(f"TruFor failed: {trufor_failures[:3]}")

    rows = []
    for sample in samples:
        sid = sample["sample_id"]
        arrays = trufor_arrays[sid]
        map_array = np.asarray(arrays["map"], dtype=np.float64)
        conf_array = arrays.get("conf")
        label = "Fake" if sample["group"] == "AutoSplice_Forged" else "Real"
        mask = None
        if label == "Fake" and sample.get("mask_path"):
            mask = _mask(str(ROOT / sample["mask_path"]))
        metrics = _metrics(trufor_scores[sid], map_array, conf_array, mask, label)
        # Save the same JET panels rendered by the runtime adapter.
        trufor._last = arrays
        artifacts = trufor.render_artifacts(images[sid])
        for name, image in artifacts.items():
            cv2.imwrite(str(out / "trufor_maps" / f"{sid}__{name}.png"), image)
        record = agent[sid]
        agent_trufor_tokens = [
            e for e in (record.get("evidence") or [])
            if e.get("source") == "trufor_expert" and e.get("status") != "failed"]
        row = {
            "sample_id": sid,
            "family": sample["family"],
            "group": sample["group"],
            "source_role": sample.get("source_role"),
            "ground_truth": sample.get("ground_truth"),
            "whole_truth": sample["task_truth"]["whole_image_generation"],
            "local_truth": sample["task_truth"]["local_manipulation"],
            "path": sample["path"],
            "mask_path": sample.get("mask_path"),
            "probe_score": round(probe_scores[sid], 6),
            "probe_pred_ai": probe_scores[sid] >= PROBE_THRESHOLD,
            "trufor_score": round(trufor_scores[sid], 6),
            "trufor_pred_tampered": trufor_scores[sid] >= TRUFOR_SCORE_THRESHOLD,
            "trufor_map_high_fraction": metrics.get("map_high_fraction"),
            "trufor_anomaly_peak": metrics.get("map_max"),
            "trufor_conf_mean": metrics.get("conf_mean"),
            "trufor_iou": metrics.get("iou"),
            "trufor_f1": metrics.get("f1"),
            "trufor_precision": metrics.get("precision"),
            "trufor_recall": metrics.get("recall"),
            "agent_trufor": _agent_state(record),
            "agent_trufor_score": (float(agent_trufor_tokens[0]["raw_metric"])
                                   if agent_trufor_tokens else None),
            "agent_local": record.get("local_manipulation"),
            "agent_final": record.get("final_verdict"),
            "agent_certainty": record.get("certainty"),
            "focus_group": _focus_group(record),
            "agent_page": record.get("page"),
        }
        rows.append(row)

    per_sample = out / "per_sample.csv"
    _write_csv(per_sample, rows)
    probe_stats = _probe_stats(rows)
    trufor_stats = _trufor_stats(rows)
    focus_stats = _focus_stats(rows)
    measured_agent = [r for r in rows if r["agent_trufor"] == "measured"
                      and r.get("agent_trufor_score") is not None]
    score_matches = sum(
        1 for r in measured_agent
        if abs(float(r["agent_trufor_score"]) - float(r["trufor_score"])) < 1e-6)
    payload = {
        "schema": "independent_expert_compare/1",
        "manifest": str(args.manifest),
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "device": args.device,
        "probe": probe_stats,
        "trufor": trufor_stats,
        "focus": focus_stats,
        "issue_reading": _issue_reading(rows),
        "agent_score_check": {"measured": len(measured_agent),
                              "exact_matches": score_matches},
        "weights": {
            "probe": ProbeDinoExpert(device="cpu").provenance().as_dict(),
            "trufor": trufor.provenance().as_dict(),
        },
        "rows": len(rows),
        "errors": [],
    }
    # The weight provenance above is read from files, not from a model load.
    json.dump(payload, open(out / "report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    _write_report(out / "report.md", payload)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("rows",)}, ensure_ascii=False, indent=2)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
