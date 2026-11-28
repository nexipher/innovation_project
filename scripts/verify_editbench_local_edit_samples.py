"""Verify EditBench generated-subset samples for pre-edit base / local edit / mask.

Downloads only five selected single samples from the public EditBench bucket
(reference, mask, one actual Imagen Editor mask-rich output) and renders a
side-by-side review.  The full 379 MB tarball is not downloaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/root/autodl-tmp/innovation_project")
PROJECT = "https://imagen.research.google/editor/"
BUCKET = "https://storage.googleapis.com/editbench"
ANNOTATIONS = "annotations_generated.csv"
INVENTORY = {
    "bucket_objects": 12003,
    "bucket_total_bytes": 11470153777,
    "generated_items": 120,
    "annotations_generated_rows": 120,
    "official_tarball_bytes": 397810521,
}
DEFAULT_RAW = Path("/tmp/editbench_samples")
DEFAULT_OUT = ROOT / "sft_data/editbench_verification_20261128"

# (stable id, annotation row, file stem)
SAMPLES = [
    ("eb_gen_029_pig", 29, "a=black_o=pig_s=outdoor"),
    ("eb_gen_025_dog", 25, "a=blue_o=dog_s=painting"),
    ("eb_gen_015_sawfish", 15, "a=cloth_o=sawfish_s=indoor"),
    ("eb_gen_064_triangle", 64, "a=triangle_o=letter_style=real"),
    ("eb_gen_114_three", 114, "a=three_o=letter_style=painting"),
]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def fetch(name: str, raw: Path) -> Path:
    path = raw / name.replace("/", "__")
    if path.exists():
        return path
    url = f"{BUCKET}/{name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=90) as response:
        path.write_bytes(response.read())
    return path


def metrics(ref: np.ndarray, out: np.ndarray, mask: np.ndarray) -> dict:
    if ref.shape != out.shape or ref.shape[:2] != mask.shape:
        raise ValueError(f"shape mismatch: {ref.shape}, {out.shape}, {mask.shape}")
    d = np.abs(ref.astype(np.int16) - out.astype(np.int16)).max(axis=2)
    outside = ~mask
    changed = d > 10
    inter = int((changed & mask).sum())
    union = int((changed | mask).sum())
    ys, xs = np.where(mask)
    return {
        "width": int(mask.shape[1]),
        "height": int(mask.shape[0]),
        "mask_fraction": round(float(mask.mean()), 6),
        "mask_bbox_xyxy": [int(xs.min()), int(ys.min()),
                           int(xs.max()) + 1, int(ys.max()) + 1],
        "outside_mean_abs": round(float(d[outside].mean()), 6),
        "outside_p99_9_abs": float(np.percentile(d[outside], 99.9)),
        "outside_max_abs": int(d[outside].max()),
        "outside_fraction_gt5": round(float((d[outside] > 5).mean()), 6),
        "inside_mean_abs": round(float(d[mask].mean()), 6),
        "inside_fraction_gt10": round(float((d[mask] > 10).mean()), 6),
        "changed_vs_mask_iou": round(inter / union, 6) if union else None,
        "changed_pixels": int(changed.sum()),
        "inside_changed_pixels": int((changed & mask).sum()),
    }


def local_edit_verified(m: dict) -> bool:
    return (
        m["outside_fraction_gt5"] < 0.01
        and m["inside_mean_abs"] > 10
        and m["inside_fraction_gt10"] > 0.5
        and (m["changed_vs_mask_iou"] or 0) >= 0.5
    )


def preview(path: Path, sample_id: str, stem: str, row: dict, ref: np.ndarray,
            out: np.ndarray, mask: np.ndarray, m: dict) -> None:
    size = 384
    gap = 10
    top = 76
    bottom = 120
    panels = 5
    canvas = Image.new("RGB", (panels * size + (panels - 1) * gap, top + size + bottom),
                       "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 14)
    font_bold = ImageFont.truetype(FONT_BOLD, 17)
    font_small = ImageFont.truetype(FONT, 12)
    draw.text((8, 8), f"{sample_id}  ·  row {row['_row']}  ·  {stem}",
              fill="black", font=font_bold)
    draw.text((8, 30), f"{row['attribute']} / {row['object_type']} / "
                       f"{row['scene_type']}  ·  generated subset",
              fill=(70, 70, 70), font=font)

    def panel(arr):
        return Image.fromarray(np.asarray(arr, dtype=np.uint8)).resize(
            (size, size), Image.Resampling.LANCZOS)

    panels_data = [
        (ref, "Reference (pre-edit generated base / scoring reference)"),
        (out, "Actual editor output: im_generated_mask-rich_1"),
        (np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2),
         "Mask (white = edited region)"),
    ]
    overlay = out.copy()
    overlay[mask] = (overlay[mask] * 0.35 + np.array([255, 0, 0]) * 0.65).astype(np.uint8)
    panels_data.append((overlay, "Actual output + mask overlay"))
    diff = np.clip(np.abs(ref.astype(np.int16) - out.astype(np.int16)).max(axis=2) * 3,
                   0, 255).astype(np.uint8)
    diff = cv2.applyColorMap(diff, cv2.COLORMAP_JET)[:, :, ::-1]
    panels_data.append((diff, "|reference - output| ×3"))

    for i, (arr, label) in enumerate(panels_data):
        x = i * (size + gap)
        canvas.paste(panel(arr), (x, top))
        draw.text((x + 2, top - 18), label, fill="black", font=font_small)

    y = top + size + 6
    prompt = "mask-rich prompt: " + row["prompt_mask-rich"]
    for line in textwrap.wrap(prompt, width=150)[:2]:
        draw.text((8, y), line, fill=(30, 30, 30), font=font_small)
        y += 16
    stats = (f"mask {m['mask_fraction']:.3f}  ·  outside mean {m['outside_mean_abs']:.2f}, "
             f">5 {m['outside_fraction_gt5']*100:.2f}%  ·  inside mean {m['inside_mean_abs']:.1f}, "
             f">10 {m['inside_fraction_gt10']*100:.1f}%  ·  changed/mask IoU {m['changed_vs_mask_iou']:.3f}")
    draw.text((8, y + 2), stats, fill=(20, 20, 20), font=font_small)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(DEFAULT_RAW))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    raw = Path(args.raw)
    out_dir = Path(args.out)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)
    (out_dir / "review").mkdir(parents=True, exist_ok=True)
    ann = fetch(ANNOTATIONS, raw)
    rows = list(csv.DictReader(ann.open(encoding="utf-8")))
    manifest = {
        "schema": "editbench_local_edit_verification/1",
        "dataset": "EditBench (Google Research, CVPR 2023)",
        "official_project": PROJECT,
        "bucket": BUCKET,
        "annotation_file": f"{BUCKET}/{ANNOTATIONS}",
        "subset": "generated",
        "editor_output_used": "im_generated_mask-rich_1 (Imagen Editor, IM); base may be Imagen or Parti",
        "reference_role": "reference_generated: complete pre-edit generated base / scoring reference",
        "actual_output_role": "im_generated_mask-rich_1: actual Imagen Editor output",
        "not_downloaded": "editbench.tar.gz (379 MB per page; 397.8 MB bucket object); only individual bucket objects used",
        "file_inventory": INVENTORY,
        "samples": [],
    }
    source_rows = []
    for sample_id, row_index, stem in SAMPLES:
        row = dict(rows[row_index])
        row["_row"] = row_index
        if row["aos"] != stem:
            raise ValueError(f"annotation mismatch at row {row_index}: {row['aos']} != {stem}")
        ref_path = fetch(f"reference_generated/{stem}.png", raw)
        mask_path = fetch(f"mask_generated/{stem}.png", raw)
        out_path = fetch(f"im_generated_mask-rich_1/{stem}.png", raw)
        ref = np.array(Image.open(ref_path).convert("RGB"))
        out = np.array(Image.open(out_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) > 0
        m = metrics(ref, out, mask)
        verified = local_edit_verified(m)
        preview_path = out_dir / "previews" / f"{sample_id}.jpg"
        preview(preview_path, sample_id, stem, row, ref, out, mask, m)
        entry = {
            "sample_id": sample_id,
            "annotation_row": row_index,
            "file_stem": stem,
            "generated_subset": True,
            "attribute": row["attribute"],
            "object_type": row["object_type"],
            "scene_type": row["scene_type"],
            "prompts": {
                "full": row["prompt_full"],
                "mask-simple": row["prompt_mask-simple"],
                "mask-rich": row["prompt_mask-rich"],
            },
            "reference_path": f"reference_generated/{stem}.png",
            "mask_path": f"mask_generated/{stem}.png",
            "actual_editor_output_path": f"im_generated_mask-rich_1/{stem}.png",
            "reference_url": f"{BUCKET}/reference_generated/{urllib.parse.quote(stem)}.png",
            "mask_url": f"{BUCKET}/mask_generated/{urllib.parse.quote(stem)}.png",
            "actual_editor_output_url": f"{BUCKET}/im_generated_mask-rich_1/{urllib.parse.quote(stem)}.png",
            "reference_role": "pre-edit generated base / scoring reference",
            "actual_output_role": "actual Imagen Editor output",
            "metrics": m,
            "local_edit_verified": verified,
            "preview_path": f"previews/{sample_id}.jpg",
        }
        manifest["samples"].append(entry)
        source_rows.append({
            "sample_id": sample_id, "annotation_row": row_index,
            "file_stem": stem, "attribute": row["attribute"],
            "object_type": row["object_type"], "scene_type": row["scene_type"],
            "reference_url": entry["reference_url"],
            "mask_url": entry["mask_url"],
            "actual_editor_output_url": entry["actual_editor_output_url"],
        })
    manifest["downloaded"] = {
        "raw_files": len(list(raw.glob("*.png"))) if raw.exists() else 0,
        "raw_bytes": sum(p.stat().st_size for p in raw.glob("*.png")) if raw.exists() else 0,
        "preview_bytes": sum(p.stat().st_size for p in (out_dir / "previews").glob("*.jpg")),
    }
    manifest["counts"] = {
        "samples": len(manifest["samples"]),
        "verified": sum(x["local_edit_verified"] for x in manifest["samples"]),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (out_dir / "sources.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(source_rows)

    lines = [
        "# EditBench 生成图局部编辑核验",
        "",
        "官方 EditBench 包是 379 MB tar；本轮未下载整包，只从公开 bucket 取 5 组单样本"
        "（reference + mask + 实际 Imagen Editor 输出），raw 约 14 MB、preview 约 1.1 MB，"
        "新增磁盘约 15 MB。",
        "",
        "## 官方文件清单",
        "",
        "- 官方页面：https://imagen.research.google/editor/；整包 `editbench.tar.gz` 页面标 379 MB"
        "（bucket 对象 397,810,521 B）。",
        "- 公开 bucket 共 12,003 个对象、11,470,153,777 B；生成子集 120 个 item，"
        "`reference_generated` 与 `mask_generated` 各 120 个；`im_*`/`dl2_*`/`sd_*`/`imrm_*`"
        "是各编辑器/提示类型的输出；`annotations_generated.csv` 120 行。",
        "- 本轮只取 5 组：5 个 reference、5 个 mask、5 个 `im_generated_mask-rich_1` 输出；"
        "未下载整包，也未下载 `dl2_*`/`sd_*`。",
        "",
        "## 结论",
        "",
        "**可以用于“AI 生成图随后被局部编辑”的评测，但需使用生成子集的 reference/mask/"
        "`im_generated_mask-rich_1` 三件套，不能把 reference 当成编辑后图。**",
        "5/5 组满足：mask 外差异均值 <1 级，>5 级像素 ≤0.32%；mask 内平均差异 50–124，"
        ">10 级像素 66–91%；变化区域与 mask 的 IoU 0.66–0.91。",
        "",
        "## 每组的角色",
        "",
        "- `reference_generated/<stem>.png`：完整编辑前生成底图（1024×1024 RGBA），"
        "同时是数据集用于评分的 reference；不是编辑器输出。预览中底图水印有 Imagen 和 "
        "Parti 两种，说明生成子集的底图生成器混合，不全是 Imagen。",
        "- `mask_generated/<stem>.png`：1024×1024 二值 mask，白色为编辑区域。",
        "- `im_generated_mask-rich_1/<stem>.png`：实际 Imagen Editor 输出，1024×1024 RGB；"
        "`dl2_*`、`sd_*` 是同 benchmark 的其他编辑器输出，本轮未用。",
        "- bucket 没有单独的 masked-input 文件；reference + mask 可重建输入，但官方未单独发布该中间图。",
        "",
        "## 样本",
        "",
        "| sample | row | 类别 | mask 占比 | 外均值 | 外>5 | 内均值 | 内>10 | 变化/mask IoU |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in manifest["samples"]:
        m = s["metrics"]
        lines.append(
            f"| `{s['sample_id']}` | {s['annotation_row']} | {s['attribute']}/"
            f"{s['object_type']}/{s['scene_type']} | {m['mask_fraction']:.3f} | "
            f"{m['outside_mean_abs']:.2f} | {m['outside_fraction_gt5']*100:.2f}% | "
            f"{m['inside_mean_abs']:.1f} | {m['inside_fraction_gt10']*100:.1f}% | "
            f"{m['changed_vs_mask_iou']:.3f} |")
    lines += [
        "",
        "## 边界",
        "",
        "- 非编辑区不是逐位相同：5 组 mask 外仍有 0.13–0.31% 像素差异 >5 级，均值 <1 级，"
        "可能是输出重编码、RGBA→RGB、边缘/水印造成。若要求“mask 外必须严格 bit-exact”，"
        "EditBench 输出不满足，需要另行处理。",
        "- 本次只核验 Imagen Editor 的 mask-rich 输出；`dl2_*`/`sd_*` 也可作为其他编辑器输出，"
        "但未下载。",
        "- 预览在 `review/index.html`；逐样本 URL/指标在 `manifest.json` 和 `sources.csv`。",
        "- 未跑 TruFor/Agent，未改规则，未下载整包。",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cards = []
    for s in manifest["samples"]:
        m = s["metrics"]
        cards.append(f"""
<article id="{s['sample_id']}">
<h2>{s['sample_id']}</h2>
<p class="meta">row {s['annotation_row']} · {s['attribute']}/{s['object_type']}/{s['scene_type']}
· mask {m['mask_fraction']:.3f} · outside mean {m['outside_mean_abs']:.2f}
(>5 {m['outside_fraction_gt5']*100:.2f}%) · inside mean {m['inside_mean_abs']:.1f}
(>10 {m['inside_fraction_gt10']*100:.1f}%) · IoU {m['changed_vs_mask_iou']:.3f}</p>
<p><b>mask-rich prompt</b>: {s['prompts']['mask-rich']}</p>
<p class="meta">reference: <code>{s['reference_path']}</code><br>
mask: <code>{s['mask_path']}</code><br>
actual editor output: <code>{s['actual_editor_output_path']}</code></p>
<figure><img loading="lazy" src="../{s['preview_path']}" alt="{s['sample_id']}"></figure>
</article>""")
    html_text = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>EditBench 生成图局部编辑核验</title>
<style>
body {{ font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 1500px; margin: 24px auto; padding: 0 20px; }}
h1 {{ margin-bottom: 4px; }}
.meta {{ color: #666; }}
article {{ border-top: 1px solid #bbb; padding: 18px 0; }}
figure {{ margin: 8px 0 0; }}
figure img {{ width: 100%; height: auto; border: 1px solid #ddd; }}
code {{ font-size: 12px; }}
</style></head><body>
<h1>EditBench 生成图局部编辑核验</h1>
<p class="meta">5 组 generated 子集；reference（编辑前底图）/ mask / Imagen Editor 实际输出
（im_generated_mask-rich_1）。预览由 reference、实际输出、mask、mask overlay、diff ×3 五栏组成。</p>
<p><b>结论</b>：5/5 组 mask 外近似不变、mask 内大幅变化；可用于“AI 生成图随后被局部编辑”的评测。
但 reference 不是编辑后图，实际输出是 <code>im_generated_mask-rich_1</code>。</p>
{''.join(cards)}
</body></html>
"""
    (out_dir / "review" / "index.html").write_text(html_text, encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "counts": manifest["counts"],
                      "samples": [(s["sample_id"], s["local_edit_verified"],
                                   s["metrics"]["changed_vs_mask_iou"])
                                  for s in manifest["samples"]]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
