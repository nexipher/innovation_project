"""Run independent TruFor on the five verified EditBench local-edit pairs.

Both sides of each pair are encoded from their source PNGs with the same
JPEG q95 parameters used by the GenImage pair evaluation.  TruFor is run on
the 10 JPEGs; the mask is used only to score localisation, never as a model
input.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from experts.trufor import TruForExpert  # noqa: E402
from utils.image_utils import ImageUtils  # noqa: E402
from analyze_trufor_local_edit_pairs import (  # noqa: E402
    HIGH, LOCALIZATION_IOU, _region_metrics,
)

BATCH = ROOT / "sft_data/editbench_verification_20261128"
VERIFY = BATCH / "manifest.json"
SOURCES = BATCH / "sources.csv"
RAW = Path("/tmp/editbench_samples")
OUT = BATCH / "analysis/trufor"
JPEG_PARAMS = {"quality": 95, "subsampling": 2, "optimize": False,
               "progressive": False}


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        path.write_bytes(response.read())


def raw_path(stem: str, prefix: str) -> Path:
    return RAW / f"{prefix}__{stem}.png"


def ensure_raw(sample: dict, urls: dict[str, dict]) -> tuple[Path, Path, Path]:
    stem = sample["file_stem"]
    ref = raw_path(stem, "reference_generated")
    out = raw_path(stem, "im_generated_mask-rich_1")
    mask = raw_path(stem, "mask_generated")
    row = urls.get(sample["sample_id"], {})
    for path, key in ((ref, "reference_url"), (out, "actual_editor_output_url"),
                      (mask, "mask_url")):
        if not path.exists():
            if not row.get(key):
                raise FileNotFoundError(f"missing raw file and URL: {path}")
            _download(row[key], path)
    return ref, out, mask


def save_q95(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="JPEG", **JPEG_PARAMS)


def _aggregate(rows: list[dict]) -> dict:
    return {
        "pairs": len(rows),
        "edited_detected": sum(row["edited"]["score"] >= HIGH for row in rows),
        "edited_localized": sum(
            (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "edited_detected_and_localized": sum(
            row["edited"]["score"] >= HIGH
            and (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "edited_mask_high_ge_0_1": sum(
            (row["edited"]["mask_high_fraction"] or 0) >= 0.1 for row in rows),
        "original_score_fp": sum(row["original"]["score"] >= HIGH for row in rows),
        "original_localized_fp": sum(
            row["original"]["score"] >= HIGH
            and (row["original"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "original_map_iou_ge_0_1": sum(
            (row["original"]["iou"] or 0) >= LOCALIZATION_IOU for row in rows),
        "original_map_high_ge_0_1": sum(
            (row["original"]["mask_high_fraction"] or 0) >= 0.1 for row in rows),
        "original_map_covers": [
            row["pair_id"] for row in rows
            if (row["original"]["iou"] or 0) >= LOCALIZATION_IOU
            or (row["original"]["mask_high_fraction"] or 0) >= 0.1
        ],
        "low_score_heatmap_covers": [
            row["pair_id"] for row in rows
            if row["edited"]["score"] < HIGH
            and ((row["edited"]["iou"] or 0) >= LOCALIZATION_IOU
                 or (row["edited"]["mask_high_fraction"] or 0) >= 0.1)
        ],
    }


def _preview(path: Path, row: dict, ref: np.ndarray, edited: np.ndarray,
             mask: np.ndarray, anomaly: np.ndarray) -> None:
    from PIL import ImageDraw, ImageFont
    size = 384
    gap = 10
    top = 66
    bottom = 92
    canvas = Image.new("RGB", (4 * size + 3 * gap, top + size + bottom), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    draw.text((8, 6), f"{row['pair_id']}  ·  row {row['annotation_row']}  ·  {row['file_stem']}",
              fill="black", font=bold)
    draw.text((8, 30), f"TruFor: original {row['original']['score']:.3f} → edited "
                       f"{row['edited']['score']:.3f}  ·  edited IoU {row['edited']['iou']:.3f}  ·  "
                       f"mask high {row['edited']['mask_high_fraction']:.3f}",
              fill=(40, 40, 40), font=font)
    overlay = edited.copy()
    overlay[mask] = (overlay[mask] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    panels = [
        (ref, "Reference (pre-edit)"),
        (edited, "Actual editor output"),
        (overlay, "Output + mask overlay"),
        (anomaly, "TruFor anomaly map"),
    ]
    for i, (arr, label) in enumerate(panels):
        x = i * (size + gap)
        im = Image.fromarray(np.asarray(arr, dtype=np.uint8)).resize(
            (size, size), Image.Resampling.LANCZOS)
        canvas.paste(im, (x, top))
        draw.text((x + 2, top - 18), label, fill="black", font=small)
    draw.text((8, top + size + 8),
              f"mask {row['edited']['mask_fraction']:.3f}  ·  outside mean "
              f"{row['verification']['outside_mean_abs']:.2f}  ·  changed/mask IoU "
              f"{row['verification']['changed_vs_mask_iou']:.3f}",
              fill=(60, 60, 60), font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    out = Path(args.out)
    inputs = out / "inputs"
    maps = out / "trufor_maps"
    raw_results = out / "trufor_raw"
    for d in (inputs, maps, raw_results, out / "previews"):
        d.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRUFOR_RESULTS", str(raw_results))

    verification = json.loads(VERIFY.read_text(encoding="utf-8"))
    urls = {row["sample_id"]: row for row in csv.DictReader(SOURCES.open(encoding="utf-8"))}
    samples = verification["samples"]
    if len(samples) != 5:
        raise SystemExit(f"expected 5 verified samples, got {len(samples)}")

    inputs_by_id = {}
    rows = []
    for sample in samples:
        pair_id = sample["sample_id"]
        ref_png, edit_png, mask_png = ensure_raw(sample, urls)
        ref = np.array(Image.open(ref_png).convert("RGB"))
        edited = np.array(Image.open(edit_png).convert("RGB"))
        mask = np.array(Image.open(mask_png).convert("L")) > 0
        if ref.shape != edited.shape or ref.shape[:2] != mask.shape:
            raise ValueError(f"shape mismatch: {pair_id}")
        orig_jpg = inputs / f"{pair_id}__original.jpg"
        edit_jpg = inputs / f"{pair_id}__edited.jpg"
        mask_png_out = inputs / f"{pair_id}__mask.png"
        save_q95(ref, orig_jpg)
        save_q95(edited, edit_jpg)
        Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_png_out)
        with Image.open(orig_jpg) as a, Image.open(edit_jpg) as b:
            if a.quantization != b.quantization or a.size != b.size:
                raise ValueError(f"JPEG mismatch: {pair_id}")
            width, height = a.size
        inputs_by_id[f"{pair_id}__original"] = ImageUtils.load_image(str(orig_jpg))
        inputs_by_id[f"{pair_id}__edited"] = ImageUtils.load_image(str(edit_jpg))
        rows.append({
            "pair_id": pair_id,
            "annotation_row": sample["annotation_row"],
            "file_stem": sample["file_stem"],
            "attribute": sample["attribute"],
            "object_type": sample["object_type"],
            "scene_type": sample["scene_type"],
            "prompt_mask_rich": sample["prompts"]["mask-rich"],
            "original_jpeg": str(orig_jpg.relative_to(BATCH)),
            "edited_jpeg": str(edit_jpg.relative_to(BATCH)),
            "mask_png": str(mask_png_out.relative_to(BATCH)),
            "width": width, "height": height,
            "verification": sample["metrics"],
        })

    trufor = TruForExpert(device=args.device, cache=True,
                          results_dir=str(raw_results))
    reason = trufor.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor unavailable: {reason}")
    results = trufor.analyze_many(inputs_by_id, progress=True)

    for row in rows:
        pair_id = row["pair_id"]
        mask = np.array(Image.open(BATCH / row["mask_png"]).convert("L")) > 0
        mask_metric = mask.astype(np.uint8) * 255
        for variant in ("original", "edited"):
            image_id = f"{pair_id}__{variant}"
            key = trufor.cache_key(inputs_by_id[image_id])
            arrays = dict(np.load(trufor.result_path(key), allow_pickle=True))
            row[variant] = _region_metrics(
                float(results[image_id].raw_metric), arrays, mask_metric)
            trufor._last = arrays
            artifacts = trufor.render_artifacts(inputs_by_id[image_id])
            for name, image in artifacts.items():
                cv2.imwrite(str(maps / f"{pair_id}__{variant}__{name}.png"), image)
            if variant == "edited":
                # clean JET maps for the representative strips
                anomaly = np.asarray(arrays["map"], dtype=np.float64)
                anomaly = cv2.resize(anomaly, (row["width"], row["height"]),
                                     interpolation=cv2.INTER_LINEAR)
                anomaly_bgr = cv2.applyColorMap(
                    np.clip(anomaly * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite(str(maps / f"{pair_id}__edited__anomaly_clean.png"),
                            anomaly_bgr)
                Image.fromarray(anomaly_bgr[:, :, ::-1]).save(
                    out / "previews" / f"{pair_id}__anomaly.png")
        row["edited_detected"] = row["edited"]["score"] >= HIGH
        row["edited_localized"] = (row["edited"]["iou"] or 0) >= LOCALIZATION_IOU
        row["edited_detected_and_localized"] = (
            row["edited_detected"] and row["edited_localized"])
        row["original_score_fp"] = row["original"]["score"] >= HIGH
        row["original_localized_fp"] = (
            row["original_score_fp"]
            and (row["original"]["iou"] or 0) >= LOCALIZATION_IOU)
        row["low_score_heatmap_covers"] = (
            row["edited"]["score"] < HIGH
            and (row["edited_localized"]
                 or (row["edited"]["mask_high_fraction"] or 0) >= 0.1))
        row["score_delta"] = round(row["edited"]["score"] - row["original"]["score"], 6)
        row["mask_mean_delta"] = round(
            (row["edited"]["mask_mean"] or 0) - (row["original"]["mask_mean"] or 0), 6)

        # Representative strip from the original PNGs, not JPEG, for clarity.
        ref = np.array(Image.open(raw_path(row["file_stem"], "reference_generated")).convert("RGB"))
        edited = np.array(Image.open(raw_path(row["file_stem"], "im_generated_mask-rich_1")).convert("RGB"))
        key = trufor.cache_key(inputs_by_id[f"{pair_id}__edited"])
        e_arr = dict(np.load(trufor.result_path(key), allow_pickle=True))
        amap = cv2.resize(np.asarray(e_arr["map"], dtype=np.float64),
                          (row["width"], row["height"]), interpolation=cv2.INTER_LINEAR)
        anomaly_rgb = cv2.applyColorMap(
            np.clip(amap * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
        _preview(out / "previews" / f"{pair_id}.jpg", row, ref, edited, mask, anomaly_rgb)

    aggregate = _aggregate(rows)
    payload = {
        "schema": "trufor_editbench_analysis/1",
        "verification_manifest": str(VERIFY.relative_to(ROOT)),
        "sources_csv": str(SOURCES.relative_to(ROOT)),
        "raw_dir": str(RAW),
        "device": args.device,
        "encoder": {"library": f"Pillow {Image.__version__}", "params": JPEG_PARAMS,
                    "note": "reference and actual editor output each encoded from PNG"},
        "threshold": HIGH,
        "localization_iou_threshold": LOCALIZATION_IOU,
        "aggregate": aggregate,
        "timing": trufor.timing_summary(),
        "pairs": rows,
    }
    (out / "trufor_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fields = ["pair_id", "annotation_row", "file_stem", "attribute", "object_type",
              "scene_type", "original_score", "edited_score", "score_delta",
              "original_mask_mean", "edited_mask_mean", "mask_mean_delta",
              "original_mask_high_fraction", "edited_mask_high_fraction",
              "original_iou", "edited_iou", "edited_precision", "edited_recall",
              "edited_detected", "edited_localized", "edited_detected_and_localized",
              "original_score_fp", "original_localized_fp",
              "low_score_heatmap_covers", "edited_peak_in_mask", "mask_fraction"]
    with (out / "per_pair.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "pair_id": row["pair_id"], "annotation_row": row["annotation_row"],
                "file_stem": row["file_stem"], "attribute": row["attribute"],
                "object_type": row["object_type"], "scene_type": row["scene_type"],
                "original_score": row["original"]["score"],
                "edited_score": row["edited"]["score"],
                "score_delta": row["score_delta"],
                "original_mask_mean": row["original"]["mask_mean"],
                "edited_mask_mean": row["edited"]["mask_mean"],
                "mask_mean_delta": row["mask_mean_delta"],
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
                "low_score_heatmap_covers": row["low_score_heatmap_covers"],
                "edited_peak_in_mask": row["edited"]["peak_in_mask"],
                "mask_fraction": row["edited"]["mask_fraction"],
            })

    image_fields = ["pair_id", "variant", "original_score", "edited_score",
                    "score_delta", "mask_mean", "mask_p95", "mask_high_fraction",
                    "iou", "precision", "recall", "peak_in_mask", "mask_fraction"]
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            for variant in ("original", "edited"):
                m = row[variant]
                writer.writerow({
                    "pair_id": row["pair_id"], "variant": variant,
                    "original_score": row["original"]["score"],
                    "edited_score": row["edited"]["score"],
                    "score_delta": row["score_delta"],
                    "mask_mean": m["mask_mean"], "mask_p95": m["mask_p95"],
                    "mask_high_fraction": m["mask_high_fraction"],
                    "iou": m["iou"], "precision": m["precision"],
                    "recall": m["recall"], "peak_in_mask": m["peak_in_mask"],
                    "mask_fraction": m["mask_fraction"],
                })

    lines = [
        "# EditBench 5 对 — 独立 TruFor",
        "",
        "5 对、10 张图：reference 与 `im_generated_mask-rich_1` 各自从 PNG 转成 "
        "JPEG q95（quality=95, subsampling=2），TruFor 只读 JPEG；mask 只用于评估定位。",
        f"阈值 score >= {HIGH} 记检出；IoU >= {LOCALIZATION_IOU} 记定位命中。",
        "",
        "## 聚合",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| 编辑图检出（score>=0.5） | {aggregate['edited_detected']}/5 |",
        f"| 编辑图定位命中（IoU>=0.1） | {aggregate['edited_localized']}/5 |",
        f"| 编辑图检出且定位 | {aggregate['edited_detected_and_localized']}/5 |",
        f"| 编辑图 mask 高响应>=0.1 | {aggregate['edited_mask_high_ge_0_1']}/5 |",
        f"| 原图 score 误报 | {aggregate['original_score_fp']}/5 |",
        f"| 原图 score+定位误报 | {aggregate['original_localized_fp']}/5 |",
        f"| 原图 mask IoU>=0.1（未计 score） | {aggregate['original_map_iou_ge_0_1']}/5 |",
        f"| 原图 mask 高响应>=0.1（未计 score） | {aggregate['original_map_high_ge_0_1']}/5 |",
        "",
        "## 逐对",
        "",
        "| pair | 原 score | 编 score | Δscore | 原 mask mean | 编 mask mean | 编 IoU | 编高响应 | 检出/定位 | 低分但热图覆盖 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['pair_id']}` | {row['original']['score']:.3f} | "
            f"{row['edited']['score']:.3f} | {row['score_delta']:+.3f} | "
            f"{row['original']['mask_mean']:.3f} | {row['edited']['mask_mean']:.3f} | "
            f"{row['edited']['iou']:.3f} | {row['edited']['mask_high_fraction']:.3f} | "
            f"{'Y' if row['edited_detected'] else 'N'}/"
            f"{'Y' if row['edited_localized'] else 'N'} | "
            f"{'Y' if row['low_score_heatmap_covers'] else 'N'} |")
    low = aggregate["low_score_heatmap_covers"]
    lines += [
        "",
        "## 图级分数低但热图覆盖编辑区",
        "",
        (", ".join(f"`{x}`" for x in low) if low else "无。"),
        "",
        "逐例：",
        "",
        "- `eb_gen_029_pig`：score 0.115（<0.5），IoU 0.187、mask 高响应 0.191；原图同区域高响应 0.0004，属于编辑后新出现的局部响应。",
        "- `eb_gen_064_triangle`：score 0.469（刚低于 0.5），IoU 0.293、高响应 0.293；原图 0.072，主要是编辑后响应。",
        "- `eb_gen_114_three`：score 0.336，IoU 0.420、高响应 0.583；但原图同区域已有高响应 0.788、IoU 0.537，编辑后反而下降，不能把热图覆盖当作本次编辑证据。",
        "",
        "其他：`eb_gen_025_dog` score 0.473（接近阈值）但 IoU 0、mask 高响应 0，属于 TruFor 未定位；"
        "`eb_gen_015_sawfish` score 0.601 检出，但原图 score 0.947 是图级误报（原图热图并未覆盖 mask）。",
        "",
        "## 代表热图",
        "",
        "`previews/<pair_id>.jpg` 为 reference / actual output / mask overlay / "
        "TruFor anomaly map 四栏；完整 5 对预览见 `previews/`。",
        "",
        "## 产物",
        "",
        "- `per_pair.csv`：5 行逐对指标；`per_image.csv`：10 行逐图指标。",
        "- `trufor_analysis.json`：完整指标、阈值与编码参数。",
        "- `trufor_maps/`：每张图的 anomaly/confidence 图；`previews/`：代表图。",
        "- 输入 JPEG 在 `inputs/`；原始 PNG 只在 `/tmp/editbench_samples`。",
        "",
        "## 边界",
        "",
        "本轮只跑独立 TruFor；未跑 Agent，未安装 PIM，未改规则/训练。"
        "EditBench mask 外不是逐位 bit-exact，详见上一轮核验报告。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cards = []
    for row in rows:
        cards.append(f"""<article>
<h2>{row['pair_id']}</h2>
<p class="meta">row {row['annotation_row']} · {row['attribute']}/{row['object_type']}/{row['scene_type']} ·
original {row['original']['score']:.3f} → edited {row['edited']['score']:.3f} ·
edited IoU {row['edited']['iou']:.3f} · mask high {row['edited']['mask_high_fraction']:.3f}</p>
<figure><img loading="lazy" src="../previews/{row['pair_id']}.jpg" alt="{row['pair_id']}"></figure>
</article>""")
    html_text = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EditBench 5 对 — TruFor 代表热图</title>
<style>
body { font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 1500px; margin: 24px auto; padding: 0 20px; }
h1 { margin-bottom: 4px; }
.meta { color: #666; }
article { border-top: 1px solid #bbb; padding: 16px 0; }
figure { margin: 8px 0 0; }
figure img { width: 100%; height: auto; border: 1px solid #ddd; }
</style></head><body>
<h1>EditBench 5 对 — TruFor 代表热图</h1>
<p class="meta">reference / actual editor output / mask overlay / TruFor anomaly map；
低分但热图覆盖：eb_gen_029_pig、eb_gen_064_triangle、eb_gen_114_three。
eb_gen_114_three 的原图同区域已是高响应，不能单独作为本次编辑证据。</p>
""" + "".join(cards) + """
</body></html>
"""
    (out / "representatives").mkdir(parents=True, exist_ok=True)
    (out / "representatives" / "index.html").write_text(html_text, encoding="utf-8")
    print(json.dumps({"aggregate": aggregate, "timing": payload["timing"],
                      "report": str(out / "report.md")}, ensure_ascii=False,
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
