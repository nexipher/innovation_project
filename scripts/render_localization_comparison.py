"""Render TruFor vs IML-ViT panels for the same 16-image smoke batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from scripts.render_trufor_panels import (  # noqa: E402
    _font, _hash_map, _heat, _mask_panel,
)
from scripts.trufor_autosplice_smoke import _load_bgr, _mask  # noqa: E402

TRUFOR_REPORT = ROOT / "sft_data/g8_localization/trufor_autosplice_smoke_20261102_cpu.json"
IML_REPORT = ROOT / "sft_data/g8_localization/iml_vit_autosplice_smoke16_20261105/report.json"
TRUFOR_RAW = ROOT / "sft_data/g8_localization/trufor_raw"
IML_RAW = ROOT / "sft_data/g8_localization/iml_vit_autosplice_smoke16_20261105/raw"
DEFAULT_OUT = ROOT / "sft_data/g8_localization/localization_compare16_20261105"


def _row(original, mask_panel, trufor_panel, iml_panel, title, subtitle, width=280):
    images = [original, mask_panel, trufor_panel, iml_panel]
    labels = ["original", "GT mask", "TruFor map", "IML-ViT map"]
    resized = []
    for image in images:
        height = max(1, round(image.height * width / image.width))
        resized.append(image.resize((width, height), Image.Resampling.LANCZOS))
    height = max(image.height for image in resized)
    gutter, top, bottom = 7, 58, 50
    canvas = Image.new("RGB", (4 * width + 3 * gutter, top + height + bottom),
                       (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), title, fill=(20, 20, 20), font=_font(14))
    draw.text((6, 25), subtitle, fill=(70, 70, 70), font=_font(11))
    for index, (image, label) in enumerate(zip(resized, labels)):
        x = index * (width + gutter)
        canvas.paste(image, (x, top))
        draw.text((x, top + height + 4), label, fill=(20, 20, 20), font=_font(12))
    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trufor = json.load(open(TRUFOR_REPORT, encoding="utf-8"))
    iml = json.load(open(IML_REPORT, encoding="utf-8"))
    iml_checks = {row["variant_id"]: row for row in iml["checks"]}
    tf_hashes = _hash_map(TRUFOR_RAW)
    rows = []
    for index, check in enumerate(trufor["checks"], 1):
        bgr = _load_bgr(ROOT / check["path"])
        digest = hashlib.sha256(np.ascontiguousarray(bgr).tobytes()).hexdigest()
        tf_npz = tf_hashes[digest]
        tf_arrays = dict(np.load(tf_npz, allow_pickle=True))
        iml_arrays = dict(np.load(IML_RAW / f"{check['variant_id']}.npz",
                                  allow_pickle=True))
        original = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        shape = (original.height, original.width)
        mask = _mask(ROOT / check["mask_path"]) if check["label"] == "Fake" else None
        contour = None
        if mask is not None:
            resized = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            contour, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        tf_panel = _heat(tf_arrays["map"], shape, contour)
        iml_map = iml_arrays["map_uint8"].astype(np.float32) / 255.0
        iml_panel = _heat(iml_map, shape, contour)
        iml_check = iml_checks[check["variant_id"]]
        if check["label"] == "Fake":
            subtitle = (f"TruFor IoU {check['iou']:.3f}/F1 {check['f1']:.3f}  "
                        f"IML IoU {iml_check['iou']:.3f}/F1 {iml_check['f1']:.3f}")
        else:
            subtitle = (f"Real: TruFor high {check['map_high_fraction']:.3f}  "
                        f"IML high {iml_check['map_high_fraction']:.3f}")
        panel = _row(original, _mask_panel(mask, shape), tf_panel, iml_panel,
                     f"{index:02d} {check['variant_id']}", subtitle)
        name = f"{index:02d}__{check['variant_id']}.png"
        panel.save(out / name)
        rows.append({"order": index, "variant_id": check["variant_id"],
                     "label": check["label"], "panel": name,
                     "trufor_iou": check["iou"], "trufor_f1": check["f1"],
                     "iml_iou": iml_check["iou"], "iml_f1": iml_check["f1"],
                     "trufor_high": check["map_high_fraction"],
                     "iml_high": iml_check["map_high_fraction"]})
    # contact sheet
    thumb_width, gutter, title_height = 210, 5, 38
    thumbs = []
    for row in rows:
        image = Image.open(out / row["panel"])
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
                  f"{row['order']:02d} {row['variant_id']}  {row['label']}",
                  fill=(20, 20, 20), font=_font(11))
        sheet.paste(image, (0, y + title_height))
    sheet.save(out / "contact_sheet.png")
    json.dump({"schema": "localization_compare/1", "rows": rows,
               "trufor_report": str(TRUFOR_REPORT.relative_to(ROOT)),
               "iml_report": str(IML_REPORT.relative_to(ROOT)),
               "note": "Same 16 inputs; maps are shown with the GT contour."},
              open(out / "manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(json.dumps({"out": str(out), "rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
