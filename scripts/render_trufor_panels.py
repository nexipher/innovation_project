"""Render original / GT mask / map / conf panels for the 16-image TruFor smoke.

The npz files are matched by the image-byte SHA-256 recorded in each
``meta.json``; no model is loaded and no inference is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from scripts.trufor_autosplice_smoke import _load_bgr, _mask  # noqa: E402

DEFAULT_REPORT = os.path.join(
    PROJECT_ROOT, "sft_data/g8_localization/trufor_autosplice_smoke_20261102_cpu.json")
DEFAULT_RAW = os.path.join(
    PROJECT_ROOT, "sft_data/g8_localization/trufor_raw")
DEFAULT_OUT = os.path.join(
    PROJECT_ROOT, "sft_data/g8_localization/panels_20261102")
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _hash_map(raw_dir):
    found = {}
    for meta_path in Path(raw_dir).glob("*/meta.json"):
        meta = json.load(open(meta_path, encoding="utf-8"))
        npz = meta_path.parent / "result.npz"
        if npz.exists():
            found[meta["image_sha256"]] = str(npz)
    return found


def _heat(array, shape, contour=None):
    array = np.asarray(array, dtype=np.float32)
    array = np.squeeze(array)
    height, width = shape
    if array.shape != (height, width):
        array = cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)
    array = np.clip(array, 0.0, 1.0)
    bgr = cv2.applyColorMap(np.uint8(np.round(array * 255)), cv2.COLORMAP_INFERNO)
    if contour is not None:
        cv2.drawContours(bgr, contour, -1, (0, 255, 0), 1)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _mask_panel(mask, shape):
    height, width = shape
    if mask is None:
        canvas = Image.new("RGB", (width, height), (28, 28, 28))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), "No GT mask\n(authentic)", fill=(230, 230, 230),
                  font=_font(18))
        return canvas
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    rgb = np.repeat((mask > 0)[:, :, None] * 255, 3, axis=2).astype(np.uint8)
    return Image.fromarray(rgb)


def _row_panel(original, mask_panel, map_panel, conf_panel, title, subtitle,
               width=300):
    images = [original, mask_panel, map_panel, conf_panel]
    labels = ["original", "GT mask", "map (green=GT)", "conf"]
    resized = []
    for image in images:
        height = max(1, round(image.height * width / image.width))
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    height = max(image.height for image in resized)
    gutter, top, bottom = 8, 58, 52
    canvas = Image.new("RGB", (4 * width + 3 * gutter, top + height + bottom),
                       (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill=(20, 20, 20), font=_font(15))
    draw.text((8, 26), subtitle, fill=(80, 80, 80), font=_font(12))
    for index, (image, label) in enumerate(zip(resized, labels)):
        x = index * (width + gutter)
        canvas.paste(image, (x, top))
        draw.text((x, top + height + 4), label, fill=(20, 20, 20), font=_font(13))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()
    report = json.load(open(args.report, encoding="utf-8"))
    hashes = _hash_map(args.raw_dir)
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for index, check in enumerate(report["checks"], 1):
        image_path = os.path.join(PROJECT_ROOT, check["path"])
        bgr = _load_bgr(image_path)
        digest = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
        npz_path = hashes.get(digest)
        if not npz_path:
            raise SystemExit(f"no raw npz for {check['variant_id']} ({digest})")
        arrays = dict(np.load(npz_path, allow_pickle=True))
        original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        shape = (original.height, original.width)
        mask = _mask(os.path.join(PROJECT_ROOT, check["mask_path"])) \
            if check.get("mask_path") and check["label"] == "Fake" else None
        contour = None
        if mask is not None:
            resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            contour, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        map_panel = _heat(arrays["map"], shape, contour)
        conf_panel = _heat(arrays["conf"], shape, contour)
        if check["label"] == "Fake":
            subtitle = (f"score={check['score']:.3f}  IoU={check['iou']:.3f}  "
                        f"F1={check['f1']:.3f}  recall={check['recall']:.3f}  "
                        f"high={check['map_high_fraction']:.3f}")
        else:
            subtitle = (f"score={check['score']:.3f}  real high-area="
                        f"{check['map_high_fraction']:.3f}  "
                        f"no GT mask")
        panel = _row_panel(original, _mask_panel(mask, shape), map_panel,
                           conf_panel, f"{index:02d}  {check['variant_id']}",
                           subtitle)
        filename = f"{index:02d}__{check['variant_id']}.png"
        panel.save(os.path.join(args.out, filename))
        rows.append({
            "order": index,
            "variant_id": check["variant_id"],
            "source_id": check["source_id"],
            "label": check["label"],
            "score": check["score"],
            "iou": check["iou"],
            "f1": check["f1"],
            "recall": check["recall"],
            "map_high_fraction": check["map_high_fraction"],
            "panel": filename,
            "raw_npz": os.path.relpath(npz_path, PROJECT_ROOT),
        })

    # Contact sheet: thumbnails of the four columns; readable at a glance.
    thumb_width, gutter, title_height = 220, 6, 42
    thumbs = []
    for row in rows:
        image = Image.open(os.path.join(args.out, row["panel"]))
        height = max(1, round(image.height * thumb_width / image.width))
        thumbs.append(image.resize((thumb_width, height), Image.Resampling.LANCZOS))
    row_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (4 * thumb_width + 3 * gutter,
                              len(thumbs) * (row_height + title_height)),
                      (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for index, (row, image) in enumerate(zip(rows, thumbs)):
        y = index * (row_height + title_height)
        draw.text((4, y + 4),
                  f"{row['order']:02d} {row['variant_id']}  "
                  f"label={row['label']} score={row['score']:.3f}",
                  fill=(20, 20, 20), font=_font(12))
        sheet.paste(image, (0, y + title_height))
    sheet.save(os.path.join(args.out, "contact_sheet.png"))

    manifest = {
        "schema": "trufor_panel_manifest/1",
        "report": os.path.relpath(args.report, PROJECT_ROOT),
        "raw_dir": os.path.relpath(args.raw_dir, PROJECT_ROOT),
        "count": len(rows),
        "rows": rows,
        "contact_sheet": "contact_sheet.png",
        "note": "Panels are diagnostic; no threshold or direction is changed.",
    }
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w",
                             encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({"out": args.out, "count": len(rows),
                      "contact_sheet": os.path.join(args.out, "contact_sheet.png")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
