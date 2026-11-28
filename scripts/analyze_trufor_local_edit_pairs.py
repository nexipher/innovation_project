"""Independent TruFor analysis of the 10 GenImage whole/local edit pairs.

For each pair, runs TruFor on the original and edited image, compares their
image-level scores, measures the anomaly response inside the paired edit mask,
and scores the edited image's high-response map against that mask with IoU.
No Agent, PROBE or rule change is involved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from experts.trufor import TruForExpert  # noqa: E402
from utils.image_utils import ImageUtils  # noqa: E402

BATCH = ROOT / "sft_data/genimage_local_edits_20261128"
MANIFEST = BATCH / "manifest.json"
OUT = BATCH / "analysis/trufor"
HIGH = 0.5
LOCALIZATION_IOU = 0.1
LARGE_EDIT_FRACTION = 0.30


def _region_metrics(score: float, arrays: dict, mask: np.ndarray) -> dict:
    anomaly = np.asarray(arrays["map"], dtype=np.float64)
    h, w = anomaly.shape
    high = anomaly >= HIGH
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    values = anomaly[resized]
    outside = anomaly[~resized]
    inter = int(np.logical_and(resized, high).sum())
    union = int(np.logical_or(resized, high).sum())
    peak_y, peak_x = np.unravel_index(int(np.argmax(anomaly)), anomaly.shape)
    out = {
        "score": round(float(score), 6),
        "anomaly_peak": round(float(anomaly.max()), 6),
        "anomaly_mean": round(float(anomaly.mean()), 6),
        "high_fraction": round(float(high.mean()), 6),
        "mask_pixels": int(resized.sum()),
        "mask_fraction": round(float(resized.mean()), 6),
        "mask_mean": round(float(values.mean()), 6) if values.size else None,
        "mask_p95": round(float(np.percentile(values, 95)), 6) if values.size else None,
        "mask_max": round(float(values.max()), 6) if values.size else None,
        "mask_high_fraction": round(float(high[resized].mean()), 6) if values.size else None,
        "outside_mean": round(float(outside.mean()), 6) if outside.size else None,
        "outside_high_fraction": round(float(high[~resized].mean()), 6) if outside.size else None,
        "peak_in_mask": bool(resized[peak_y, peak_x]),
        "iou": round(inter / union, 6) if union else None,
        "precision": round(inter / int(high.sum()), 6) if high.sum() else None,
        "recall": round(inter / int(resized.sum()), 6) if resized.sum() else None,
    }
    conf = arrays.get("conf")
    if conf is not None:
        conf = np.asarray(conf, dtype=np.float64)
        if conf.shape[:2] != (h, w):
            conf = cv2.resize(conf, (w, h), interpolation=cv2.INTER_LINEAR)
        out["conf_mean"] = round(float(conf.mean()), 6)
        out["conf_mean_mask"] = round(float(conf[resized].mean()), 6) if resized.any() else None
        out["low_conf_fraction"] = round(float((conf < 0.5).mean()), 6)
    return out


def _agg(rows: list[dict]) -> dict:
    edited = [row for row in rows if not row["large_edit"]]
    return {
        "pairs": len(rows),
        "edited_image_detected": sum(row["edited"]["score"] >= HIGH for row in rows),
        "edited_localized": sum(
            (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "edited_detected_and_localized": sum(
            row["edited"]["score"] >= HIGH
            and (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "edited_mask_mean_above_original": sum(
            (row["edited"]["mask_mean"] or 0) > (row["original"]["mask_mean"] or 0)
            for row in rows),
        "edited_mask_high_fraction_ge_0.1": sum(
            (row["edited"]["mask_high_fraction"] or 0) >= 0.1 for row in rows),
        "original_score_fp": sum(row["original"]["score"] >= HIGH for row in rows),
        "original_localized_fp": sum(
            row["original"]["score"] >= HIGH
            and (row["original"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "original_mask_high_fraction_ge_0.1": sum(
            (row["original"]["mask_high_fraction"] or 0) >= 0.1 for row in rows),
        "excluding_p05": {
            "pairs": len(edited),
            "edited_image_detected": sum(row["edited"]["score"] >= HIGH for row in edited),
            "edited_localized": sum(
                (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU for row in edited),
            "edited_detected_and_localized": sum(
                row["edited"]["score"] >= HIGH
                and (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU for row in edited),
            "edited_mask_high_fraction_ge_0.1": sum(
                (row["edited"]["mask_high_fraction"] or 0) >= 0.1 for row in edited),
            "original_score_fp": sum(row["original"]["score"] >= HIGH for row in edited),
            "original_localized_fp": sum(
                row["original"]["score"] >= HIGH
                and (row["original"]["iou"] or 0) >= LOCALIZATION_IOU for row in edited),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault("TRUFOR_RESULTS", str(OUT / "trufor_raw"))
    (OUT / "trufor_maps").mkdir(parents=True, exist_ok=True)
    (OUT / "trufor_raw").mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    if len(pairs) != 10:
        raise SystemExit(f"expected 10 pairs, got {len(pairs)}")

    images = {}
    for pair in pairs:
        for variant in ("original", "edited"):
            image_id = f"{pair['pair_id']}__{variant}"
            images[image_id] = ImageUtils.load_image(
                str(BATCH / pair[f"{variant}_path"]))
    trufor = TruForExpert(device=args.device, cache=True,
                          results_dir=str(OUT / "trufor_raw"))
    reason = trufor.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor unavailable: {reason}")
    results = trufor.analyze_many(images, progress=True)

    rows = []
    for pair in pairs:
        mask = cv2.imread(str(BATCH / pair["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise SystemExit(f"mask unreadable: {pair['mask_path']}")
        variants = {}
        for variant in ("original", "edited"):
            image_id = f"{pair['pair_id']}__{variant}"
            key = trufor.cache_key(images[image_id])
            arrays = dict(np.load(trufor.result_path(key), allow_pickle=True))
            variants[variant] = _region_metrics(
                float(results[image_id].raw_metric), arrays, mask)
            trufor._last = arrays
            artifacts = trufor.render_artifacts(images[image_id])
            for name, image in artifacts.items():
                cv2.imwrite(str(OUT / "trufor_maps" /
                                f"{pair['pair_id']}__{variant}__{name}.png"), image)
            # render_region_artifacts expects [ymin, xmin, ymax, xmax], while
            # the manifest stores the mask bbox as [x0, y0, x1, y1].
            x0, y0, x1, y1 = pair["mask_bbox_xyxy"]
            crop_artifacts, _ = trufor.render_region_artifacts(
                images[image_id], region_pixels=[y0, x0, y1, x1])
            for name, image in crop_artifacts.items():
                cv2.imwrite(str(OUT / "trufor_maps" /
                                f"{pair['pair_id']}__{variant}__region_{name}.png"), image)
        edited = variants["edited"]
        original = variants["original"]
        large = (edited["mask_fraction"] or 0) >= LARGE_EDIT_FRACTION
        row = {
            "pair_id": pair["pair_id"],
            "source_id": pair["source_id"],
            "generator": pair["generator"],
            "edit_type": pair["edit_type"],
            "edit_method": pair["edit_method"],
            "mask_bbox_xyxy": pair["mask_bbox_xyxy"],
            "large_edit": large,
            "score_delta": round(edited["score"] - original["score"], 6),
            "mask_mean_delta": round(
                (edited["mask_mean"] or 0) - (original["mask_mean"] or 0), 6),
            "mask_p95_delta": round(
                (edited["mask_p95"] or 0) - (original["mask_p95"] or 0), 6),
            "edited_detected": edited["score"] >= HIGH,
            "edited_localized": (edited["iou"] or 0) >= LOCALIZATION_IOU,
            "edited_detected_and_localized": (
                edited["score"] >= HIGH
                and (edited["iou"] or 0) >= LOCALIZATION_IOU),
            "original_score_fp": original["score"] >= HIGH,
            "original_localized_fp": (
                original["score"] >= HIGH
                and (original["iou"] or 0) >= LOCALIZATION_IOU),
            "original": original,
            "edited": edited,
        }
        if row["edited_detected_and_localized"]:
            row["trufor_class"] = "detected_localized"
        elif row["edited_detected"]:
            row["trufor_class"] = "detected_not_localized"
        elif row["edited_localized"]:
            row["trufor_class"] = "localized_not_detected"
        else:
            row["trufor_class"] = "not_detected"
        rows.append(row)

    aggregate = _agg(rows)
    payload = {
        "schema": "trufor_local_edit_analysis/1",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "device": args.device,
        "high_response_threshold": HIGH,
        "localization_iou_threshold": LOCALIZATION_IOU,
        "large_edit_mask_fraction_threshold": LARGE_EDIT_FRACTION,
        "definitions": {
            "edited_detected": "edited score >= 0.5",
            "edited_localized": "IoU(high-response map, edit mask) >= 0.1",
            "original_score_fp": "original score >= 0.5 on an unedited image",
            "original_localized_fp": "original score >= 0.5 and IoU(original high-response map, paired mask) >= 0.1",
        },
        "aggregate": aggregate,
        "timing": trufor.timing_summary(),
        "pairs": rows,
    }
    (OUT / "trufor_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    fields = [
        "pair_id", "generator", "edit_type", "large_edit", "trufor_class",
        "original_score", "edited_score", "score_delta",
        "original_mask_mean", "edited_mask_mean", "mask_mean_delta",
        "original_mask_p95", "edited_mask_p95", "mask_p95_delta",
        "original_mask_high_fraction", "edited_mask_high_fraction",
        "original_iou", "edited_iou", "edited_precision", "edited_recall",
        "edited_detected", "edited_localized", "edited_detected_and_localized",
        "original_score_fp", "original_localized_fp",
        "edited_peak_in_mask", "original_peak_in_mask",
        "mask_pixels", "mask_fraction",
    ]
    with (OUT / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "pair_id": row["pair_id"], "generator": row["generator"],
                "edit_type": row["edit_type"], "large_edit": row["large_edit"],
                "trufor_class": row["trufor_class"],
                "original_score": row["original"]["score"],
                "edited_score": row["edited"]["score"],
                "score_delta": row["score_delta"],
                "original_mask_mean": row["original"]["mask_mean"],
                "edited_mask_mean": row["edited"]["mask_mean"],
                "mask_mean_delta": row["mask_mean_delta"],
                "original_mask_p95": row["original"]["mask_p95"],
                "edited_mask_p95": row["edited"]["mask_p95"],
                "mask_p95_delta": row["mask_p95_delta"],
                "original_mask_high_fraction": row["original"]["mask_high_fraction"],
                "edited_mask_high_fraction": row["edited"]["mask_high_fraction"],
                "original_iou": row["original"]["iou"],
                "edited_iou": row["edited"]["iou"],
                "edited_precision": row["edited"]["precision"],
                "edited_recall": row["edited"]["recall"],
                "edited_detected": row["edited_detected"],
                "edited_localized": row["edited_localized"],
                "edited_detected_and_localized": row["edited_detected_and_localized"],
                "original_score_fp": row["original_score_fp"],
                "original_localized_fp": row["original_localized_fp"],
                "edited_peak_in_mask": row["edited"]["peak_in_mask"],
                "original_peak_in_mask": row["original"]["peak_in_mask"],
                "mask_pixels": row["edited"]["mask_pixels"],
                "mask_fraction": row["edited"]["mask_fraction"],
            })

    lines = [
        "# 10 对 GenImage 整图生成/局部编辑 — 独立 TruFor",
        "",
        f"阈值：score >= {HIGH} 记图像级检出；IoU >= {LOCALIZATION_IOU} 记定位命中；"
        f"mask 面积占比 >= {LARGE_EDIT_FRACTION} 记大范围编辑（p05 单列）。"
        "原图误报指未编辑原图的 score 或同 mask 区域响应被判为局部篡改。",
        "",
        "## 聚合",
        "",
        "| 指标 | 全部 10 对 | 不含 p05（9 对） |",
        "|---|---:|---:|",
        f"| 编辑版图像级检出（score>=0.5） | {aggregate['edited_image_detected']}/10 | {aggregate['excluding_p05']['edited_image_detected']}/9 |",
        f"| 编辑版定位命中（IoU>=0.1） | {aggregate['edited_localized']}/10 | {aggregate['excluding_p05']['edited_localized']}/9 |",
        f"| 编辑版 mask 高响应占比>=0.1 | {aggregate['edited_mask_high_fraction_ge_0.1']}/10 | {aggregate['excluding_p05']['edited_mask_high_fraction_ge_0.1']}/9 |",
        f"| 编辑版检出且定位 | {aggregate['edited_detected_and_localized']}/10 | {aggregate['excluding_p05']['edited_detected_and_localized']}/9 |",
        f"| 编辑版 mask 均值高于原图 | {aggregate['edited_mask_mean_above_original']}/10 | — |",
        f"| 原图 score 误报 | {aggregate['original_score_fp']}/10 | {aggregate['excluding_p05']['original_score_fp']}/9 |",
        f"| 原图同区域定位误报 | {aggregate['original_localized_fp']}/10 | {aggregate['excluding_p05']['original_localized_fp']}/9 |",
        "",
        "## 逐对",
        "",
        "| pair | 类型 | 大编辑 | 原 score | 编 score | Δscore | 原 mask mean | 编 mask mean | 编 IoU | 编检出/定位 | 原 FP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['pair_id']}` | {row['edit_type']} | "
            f"{'Y' if row['large_edit'] else ''} | "
            f"{row['original']['score']:.3f} | {row['edited']['score']:.3f} | "
            f"{row['score_delta']:+.3f} | {row['original']['mask_mean']:.3f} | "
            f"{row['edited']['mask_mean']:.3f} | {row['edited']['iou']:.3f} | "
            f"{'Y' if row['edited_detected'] else 'N'}/"
            f"{'Y' if row['edited_localized'] else 'N'} | "
            f"{'Y' if row['original_score_fp'] else 'N'} |")
    lines += [
        "",
        "## p05 单列",
        "",
        f"p05 编辑 mask 覆盖 {rows[4]['edited']['mask_fraction']:.1%} 的图像面积；"
        f"原图 score {rows[4]['original']['score']:.3f}，编辑版 score "
        f"{rows[4]['edited']['score']:.3f}，mask mean "
        f"{rows[4]['original']['mask_mean']:.3f}→{rows[4]['edited']['mask_mean']:.3f}，"
        f"IoU {rows[4]['edited']['iou']:.3f}。大范围编辑的分数/响应不能与小块编辑直接平均。",
        "",
        "## 产物",
        "",
        "- `per_pair.csv`：逐对 score、mask 响应、IoU、检出/误报标志。",
        "- `trufor_analysis.json`：完整指标与聚合。",
        "- `trufor_maps/*.png`：每张图的 anomaly/confidence 图和同 mask 区域三图。",
        "- 本轮只跑独立 TruFor；Agent 结果在 `analysis/agent/` 另行汇总。",
    ]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "timing": payload["timing"],
                      "report": str(OUT / "report.md")}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
