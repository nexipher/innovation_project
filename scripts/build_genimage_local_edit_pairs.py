"""Build GenImage whole-generated / locally-edited review pairs.

Each pair starts from one pure GenImage q95 derivative (whole truth
AI-generated, local truth Not tampered), applies one local add/delete/replace
edit, and records the actual changed-pixel mask.  Both sides are encoded with
the same JPEG q95 parameters.  This script only builds data, manifest and the
review page; it does not run TruFor or the Agent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "dataset/GenImage_q95/eval_dev_manifest_20261102.json"
DEFAULT_OUT = ROOT / "sft_data/genimage_local_edits_20261128"

RECENT_MANIFESTS = [
    "sft_data/lora_small_20261128/run_manifest.json",
    "sft_data/lora_small_20261128/validation_manifest_independent_20261128.json",
    "sft_data/lora_smoke_20261128/run_manifest.json",
    "sft_data/sft_candidates_20261127/records.json",
    "sft_data/g8_fixed_dual_20261127/manifest.json",
    "sft_data/g8_dev_eval_20261127_part1/manifest.json",
    "sft_data/g8_dev_eval_20261127_part2/manifest.json",
    "sft_data/independent_expert_20261127/report.json",
]
SOURCE_RE = re.compile(r"genimage__([A-Za-z0-9]+)__([^\"'\\\s,\]}]+?)__q95")
JPEG_PARAMS = {"quality": 95, "subsampling": 2, "optimize": False,
               "progressive": False}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def used_recent_sources() -> tuple[set[str], list[str]]:
    used: set[str] = set()
    present = []
    for rel in RECENT_MANIFESTS:
        path = ROOT / rel
        if not path.exists():
            continue
        present.append(rel)
        text = path.read_text(encoding="utf-8", errors="ignore")
        used.update(f"{group}/{name}" for group, name in SOURCE_RE.findall(text))
    return used, present


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def feather_rect(height: int, width: int, feather: int) -> np.ndarray:
    alpha = np.ones((height, width), np.float32)
    f = max(1, int(feather))
    ramp = np.linspace(0, 1, f + 1, dtype=np.float32)[1:]
    alpha[:f] *= ramp[:, None]
    alpha[-f:] *= ramp[::-1, None]
    alpha[:, :f] *= ramp[None, :]
    alpha[:, -f:] *= ramp[::-1][None, :]
    return alpha


def paste_box(img: np.ndarray, src: tuple[int, int, int, int],
              dst: tuple[int, int, int, int], feather: int = 3) -> np.ndarray:
    x0, y0, x1, y1 = src
    dx0, dy0, dx1, dy1 = dst
    patch = img[y0:y1, x0:x1].copy()
    tw, th = dx1 - dx0, dy1 - dy0
    if patch.shape[1] != tw or patch.shape[0] != th:
        patch = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
    alpha = feather_rect(th, tw, feather)[..., None]
    out = img.copy()
    roi = out[dy0:dy1, dx0:dx1].astype(np.float32)
    out[dy0:dy1, dx0:dx1] = np.clip(
        patch.astype(np.float32) * alpha + roi * (1 - alpha), 0, 255
    ).astype(np.uint8)
    return out


def fill_rect_background(img: np.ndarray, rect: tuple[int, int, int, int],
                         feather: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = rect
    ring = np.zeros(img.shape[:2], np.uint8)
    ring[y0:y1, x0:x1] = 255
    ring = cv2.dilate(ring, np.ones((9, 9), np.uint8), iterations=1)
    ring[y0:y1, x0:x1] = 0
    color = np.median(img[ring > 0], axis=0).astype(np.float32)
    alpha = feather_rect(y1 - y0, x1 - x0, feather)[..., None]
    out = img.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(
        color[None, None, :] * alpha + roi * (1 - alpha), 0, 255
    ).astype(np.uint8)
    return out


def poly_mask(shape: tuple[int, int], points: list[tuple[int, int]],
              dilate: int = 3) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.array(points, np.int32)], 255)
    if dilate:
        mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8), iterations=1)
    return mask


def poly_fill(img: np.ndarray, mask: np.ndarray, degree: int = 2) -> np.ndarray:
    """Fill a mask with a global low-order polynomial background fit."""
    h, w = img.shape[:2]
    ys, xs = np.where(mask == 0)
    if len(xs) > 60000:
        pick = np.random.RandomState(0).choice(len(xs), 60000, replace=False)
        xs, ys = xs[pick], ys[pick]
    xn, yn = xs / w * 2 - 1, ys / h * 2 - 1
    cols = [np.ones(len(xn))]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            cols.append(xn ** (d - k) * yn ** k)
    design = np.stack(cols, 1)
    yy, xx = np.mgrid[0:h, 0:w]
    xn2, yn2 = xx.ravel() / w * 2 - 1, yy.ravel() / h * 2 - 1
    cols2 = [np.ones(len(xn2))]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            cols2.append(xn2 ** (d - k) * yn2 ** k)
    pred_design = np.stack(cols2, 1)
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.5)
    out = img.astype(np.float32).copy()
    for channel in range(3):
        coef = np.linalg.lstsq(
            design, img[ys, xs, channel].astype(np.float32), rcond=None)[0]
        pred = (pred_design @ coef).reshape(h, w)
        out[:, :, channel] = pred * soft + out[:, :, channel] * (1 - soft)
    return np.clip(out, 0, 255).astype(np.uint8)


def cutout_paste(img: np.ndarray, src: tuple[int, int, int, int],
                 dst: tuple[int, int, int, int], obj_mask: np.ndarray,
                 shadow: tuple[int, int] = (7, 9), blur: float = 6.0,
                 darkness: float = 0.35) -> np.ndarray:
    x0, y0, x1, y1 = src
    dx0, dy0, dx1, dy1 = dst
    patch = img[y0:y1, x0:x1].astype(np.float32)
    alpha = obj_mask[y0:y1, x0:x1].astype(np.float32) / 255
    alpha = np.clip(cv2.GaussianBlur(alpha, (0, 0), 1.2), 0, 1)
    ph, pw = alpha.shape
    pad = 60
    h, w = img.shape[:2]
    canvas = np.pad(img.astype(np.float32), ((pad, pad), (pad, pad), (0, 0)),
                    mode="edge")
    tx, ty = dx0 + pad, dy0 + pad
    shadow_map = np.zeros((ph + 2 * pad, pw + 2 * pad), np.float32)
    shadow_map[pad:pad + ph, pad:pad + pw] = alpha
    shadow_map = cv2.warpAffine(
        shadow_map, np.float32([[1, 0, shadow[0]], [0, 1, shadow[1]]]),
        (pw + 2 * pad, ph + 2 * pad))
    shadow_map = cv2.GaussianBlur(shadow_map, (0, 0), blur)
    canvas[ty - pad:ty + ph + pad, tx - pad:tx + pw + pad] *= (
        1 - darkness * shadow_map[..., None])
    roi = canvas[ty:ty + ph, tx:tx + pw]
    roi[:] = patch * alpha[..., None] + roi * (1 - alpha[..., None])
    return np.clip(canvas[pad:pad + h, pad:pad + w], 0, 255).astype(np.uint8)


def add_box_label(img: np.ndarray) -> np.ndarray:
    scale = 4
    lw, lh = 130 * scale, 58 * scale
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.rounded_rectangle([1, 1, lw - 2, lh - 2], radius=4 * scale,
                           fill=(250, 250, 248, 255),
                           outline=(160, 160, 155, 255), width=scale)
    font_bold = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16 * scale)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8 * scale)
    draw.text((8 * scale, 5 * scale), "FRAGILE", font=font_bold,
              fill=(20, 20, 20, 255))
    draw.text((9 * scale, 26 * scale), "HANDLE WITH CARE", font=font,
              fill=(60, 60, 60, 255))
    x, y = 8 * scale, 40 * scale
    rng = np.random.RandomState(7)
    while x < lw - 10 * scale:
        bw = rng.randint(1, 4) * scale
        draw.rectangle([x, y, min(x + bw, lw - 10 * scale), y + 10 * scale],
                       fill=(10, 10, 10, 255))
        x += bw + rng.randint(1, 4) * scale
    layer = layer.resize((130, 58), Image.LANCZOS).rotate(
        -1.5, expand=True, resample=Image.BICUBIC)
    alpha = np.array(layer.split()[-1], np.float32) / 255
    rgb = np.array(layer.convert("RGB"), np.float32)
    shadow = cv2.GaussianBlur(alpha, (0, 0), 2.0) * 0.35
    x0, y0 = 64, 92
    h, w = alpha.shape
    out = img.astype(np.float32).copy()
    roi = out[y0:y0 + h, x0:x0 + w]
    roi *= 1 - shadow[..., None]
    roi[:] = roi * (1 - alpha[..., None]) + rgb * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_price_tag(img: np.ndarray) -> np.ndarray:
    scale = 4
    tw, th = 46 * scale, 72 * scale
    layer = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.line([(tw // 2, 0), (tw // 2, 18 * scale)], fill=(60, 60, 60, 255),
              width=2 * scale)
    draw.rounded_rectangle([2 * scale, 16 * scale, tw - 2 * scale,
                            th - 2 * scale], radius=5 * scale,
                           fill=(245, 240, 225, 255),
                           outline=(150, 140, 120, 255), width=scale)
    font_bold = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 8 * scale)
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 6 * scale)
    draw.text((7 * scale, 24 * scale), "SIZE", font=font, fill=(70, 60, 50, 255))
    draw.text((7 * scale, 34 * scale), "M", font=font_bold, fill=(30, 30, 30, 255))
    draw.text((7 * scale, 48 * scale), "$ 39", font=font, fill=(70, 60, 50, 255))
    layer = layer.resize((46, 72), Image.LANCZOS)
    alpha = np.array(layer.split()[-1], np.float32) / 255
    rgb = np.array(layer.convert("RGB"), np.float32)
    shadow = cv2.GaussianBlur(alpha, (0, 0), 1.5) * 0.4
    x0, y0 = 382, 52
    h, w = alpha.shape
    out = img.astype(np.float32).copy()
    roi = out[y0:y0 + h, x0:x0 + w]
    roi *= 1 - shadow[..., None]
    roi[:] = roi * (1 - alpha[..., None]) + rgb * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def add_sphere(img: np.ndarray) -> np.ndarray:
    src = (315, 115, 410, 212)
    x0, y0, x1, y1 = src
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    roi = hsv[y0:y1, x0:x1]
    mask = ((roi[:, :, 1] < 120) & (roi[:, :, 2] > 90)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == idx).astype(np.uint8) * 255
    full = np.zeros(img.shape[:2], np.uint8)
    full[y0:y1, x0:x1] = mask
    return cutout_paste(img, src, (20, 400, 115, 497), full)


def make_specs() -> list[dict]:
    def add_556(img: np.ndarray) -> np.ndarray:
        return paste_box(img, (100, 85, 160, 160), (178, 85, 238, 160),
                         feather=2)

    def delete_stick(img: np.ndarray) -> np.ndarray:
        mask = poly_mask(img.shape[:2], [(0, 45), (128, 42), (128, 90), (0, 93)],
                         dilate=3)
        return poly_fill(img, mask, degree=2)

    def delete_922(img: np.ndarray) -> np.ndarray:
        return fill_rect_background(img, (160, 105, 246, 160), feather=2)

    def delete_830(img: np.ndarray) -> np.ndarray:
        return fill_rect_background(img, (75, 78, 230, 175), feather=2)

    def replace_notice(img: np.ndarray) -> np.ndarray:
        return paste_box(img, (50, 100, 150, 200), (153, 125, 253, 225),
                         feather=3)

    def replace_screens(img: np.ndarray) -> np.ndarray:
        return paste_box(img, (75, 55, 210, 195), (0, 55, 65, 195), feather=3)

    def replace_vqdm_text(img: np.ndarray) -> np.ndarray:
        return paste_box(img, (10, 100, 90, 145), (165, 105, 245, 150),
                         feather=2)

    return [
        dict(pair_id="p01_vqdm_box_label_add",
             source_id="VQDM/VQDM_1000_200_08_844_vqdm_00143",
             edit_type="add", edit_method="synthetic_shipping_label",
             description="在纸箱正面添加白色 FRAGILE 物流标签", editor=add_box_label),
        dict(pair_id="p02_wukong_tag_add",
             source_id="Wukong/841_wukong_image61",
             edit_type="add", edit_method="synthetic_price_tag",
             description="在衣架右侧挂上 SIZE M 价格吊牌", editor=add_price_tag),
        dict(pair_id="p03_sd14_sphere_add",
             source_id="SD14/722_sdv4_00039",
             edit_type="add", edit_method="same_image_object_cutout",
             description="把左下空白木纹处补入一颗同图白色球体（含阴影）",
             editor=add_sphere),
        dict(pair_id="p04_glide_556_text_add",
             source_id="Glide/GLIDE_1000_200_05_556_glide_00127",
             edit_type="add", edit_method="copy_existing_text_block",
             description="在网页右侧空白处复制一段正文文本块", editor=add_556),
        dict(pair_id="p05_biggan_stick_delete",
             source_id="BigGAN/623_biggan_00127",
             edit_type="delete", edit_method="background_fill",
             description="删除画面中央的细木棍", editor=delete_stick),
        dict(pair_id="p06_glide_922_paragraph_delete",
             source_id="Glide/GLIDE_1000_200_09_922_glide_00035",
             edit_type="delete", edit_method="background_fill",
             description="删除右栏一段正文", editor=delete_922),
        dict(pair_id="p07_glide_830_caption_delete",
             source_id="Glide/GLIDE_1000_200_08_830_glide_00035",
             edit_type="delete", edit_method="background_fill",
             description="删除页面中部的说明文字块", editor=delete_830),
        dict(pair_id="p08_adm_notice_replace",
             source_id="ADM/916_adm_85",
             edit_type="replace", edit_method="copy_existing_notice",
             description="用白色通知单覆盖右下橙色警示标签", editor=replace_notice),
        dict(pair_id="p09_glide_screens_replace",
             source_id="Glide/GLIDE_1000_200_04_466_glide_00039",
             edit_type="replace", edit_method="copy_existing_screen",
             description="把左侧手机屏幕替换为中间屏幕的画面", editor=replace_screens),
        dict(pair_id="p10_vqdm_document_replace",
             source_id="VQDM/VQDM_1000_200_09_916_vqdm_00035",
             edit_type="replace", edit_method="copy_existing_text_block",
             description="用左侧文本块替换右侧文本块", editor=replace_vqdm_text),
    ]


def save_jpeg(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array).save(path, format="JPEG", **JPEG_PARAMS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    out_dir = Path(args.out)
    pairs_dir = out_dir / "pairs"
    review_dir = out_dir / "review"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    index = {row["source_id"]: row for row in source_manifest["inputs"]}
    recent_used, recent_present = used_recent_sources()

    pairs = []
    selection_rows = []
    for spec in make_specs():
        sid = spec["source_id"]
        if sid not in index:
            raise KeyError(f"source not in {args.manifest}: {sid}")
        meta = index[sid]
        if meta["group"] == "Real":
            raise ValueError(f"not a generated source: {sid}")
        if meta["new_split"] != "dev":
            raise ValueError(f"source split is not dev: {sid} ({meta['new_split']})")
        if not meta.get("no_known_use"):
            raise ValueError(f"source has known use: {sid}")
        if sid in recent_used:
            raise ValueError(f"source used by recent manifests: {sid}")

        source_path = ROOT / meta["path"]
        original = load_rgb(source_path)
        edited = spec["editor"](original.copy())
        diff = np.any(edited != original, axis=2)
        if not diff.any():
            raise ValueError(f"edit produced no changed pixels: {spec['pair_id']}")
        mask = (diff.astype(np.uint8) * 255)
        ys, xs = np.where(mask > 0)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                int(ys.max()) + 1]
        outside = ~diff
        non_edit_changed = int(np.any(edited[outside] != original[outside]))

        pair_dir = pairs_dir / spec["pair_id"]
        pair_dir.mkdir(parents=True, exist_ok=True)
        original_path = pair_dir / "original.jpg"
        edited_path = pair_dir / "edited.jpg"
        mask_path = pair_dir / "mask.png"
        save_jpeg(original, original_path)
        save_jpeg(edited, edited_path)
        Image.fromarray(mask).save(mask_path)

        with Image.open(original_path) as probe:
            orig_quant = probe.quantization
            orig_size = probe.size
        with Image.open(edited_path) as probe:
            edit_quant = probe.quantization
            edit_size = probe.size
        if orig_quant != edit_quant:
            raise ValueError(f"JPEG quantization tables differ: {spec['pair_id']}")
        if orig_size != edit_size:
            raise ValueError(f"JPEG sizes differ: {spec['pair_id']}")
        if non_edit_changed:
            raise ValueError(f"non-edit pixels changed: {spec['pair_id']}")

        pairs.append({
            "pair_id": spec["pair_id"],
            "source_id": sid,
            "generator": meta["group"],
            "source_path": meta["path"],
            "source_sha256": meta.get("source_sha256"),
            "width": meta["width"],
            "height": meta["height"],
            "edit_type": spec["edit_type"],
            "edit_method": spec["edit_method"],
            "edit_description": spec["description"],
            "original_path": f"pairs/{spec['pair_id']}/original.jpg",
            "edited_path": f"pairs/{spec['pair_id']}/edited.jpg",
            "mask_path": f"pairs/{spec['pair_id']}/mask.png",
            "mask_bbox_xyxy": bbox,
            "mask_pixels": int(diff.sum()),
            "whole_image_truth": "AI-generated",
            "local_truth_original": "Not tampered",
            "local_truth_edited": "Tampered",
            "original_local_truth_basis": (
                "GenImage generator-only q95 derivative; no local edit step"),
            "edited_local_truth_basis": (
                "local edit applied by this batch; changed-pixel mask recorded"),
            "original_sha256": _sha256(original_path),
            "edited_sha256": _sha256(edited_path),
            "mask_sha256": _sha256(mask_path),
            "non_edit_pixels_changed_before_encode": non_edit_changed,
            "jpeg_params": JPEG_PARAMS,
            "jpeg_quantization_tables_identical": True,
        })
        selection_rows.append({
            "pair_id": spec["pair_id"], "source_id": sid,
            "generator": meta["group"], "split": meta["new_split"],
            "no_known_use": meta.get("no_known_use"),
            "in_recent_manifests": False,
            "source_path": meta["path"],
        })

    counts = {
        "pairs": len(pairs),
        "by_edit_type": {
            key: sum(1 for pair in pairs if pair["edit_type"] == key)
            for key in ("add", "delete", "replace")
        },
        "by_generator": {
            key: sum(1 for pair in pairs if pair["generator"] == key)
            for key in sorted({pair["generator"] for pair in pairs})
        },
    }
    payload = {
        "schema": "genimage_local_edit_pairs/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": (
            "Whole image is AI-generated; original local truth is Not tampered; "
            "edited version has one local add/delete/replace edit with a mask."),
        "source_manifest": str(Path(args.manifest).relative_to(ROOT)),
        "selection": {
            "split": "dev",
            "no_known_use_required": True,
            "test_split_sealed": True,
            "recent_manifests": recent_present,
            "recent_used_source_count": len(recent_used),
            "note": "all selected sources are dev, no_known_use, and absent "
                    "from recent training/validation/SFT manifests",
        },
        "encoder": {
            "library": f"Pillow {Image.__version__}",
            "params": JPEG_PARAMS,
            "note": "both original and edited are encoded from the same "
                    "decoded source with identical parameters",
        },
        "truth": {
            "whole_image_generation": "AI-generated",
            "local_manipulation": {"original": "Not tampered",
                                   "edited": "Tampered"},
        },
        "counts": counts,
        "pairs": pairs,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")

    with (out_dir / "selection_audit.csv").open("w", newline="",
                                                encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(selection_rows)

    with (out_dir / "encode_audit.csv").open("w", newline="",
                                             encoding="utf-8") as handle:
        fields = ["pair_id", "width", "height", "mask_pixels",
                  "mask_bbox_xyxy", "jpeg_quality", "jpeg_subsampling",
                  "quant_tables_identical", "non_edit_pixels_changed_before_encode"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow({
                "pair_id": pair["pair_id"], "width": pair["width"],
                "height": pair["height"], "mask_pixels": pair["mask_pixels"],
                "mask_bbox_xyxy": " ".join(map(str, pair["mask_bbox_xyxy"])),
                "jpeg_quality": JPEG_PARAMS["quality"],
                "jpeg_subsampling": JPEG_PARAMS["subsampling"],
                "quant_tables_identical": True,
                "non_edit_pixels_changed_before_encode": 0,
            })

    write_review_page(out_dir, pairs, counts)
    write_readme(out_dir, counts, recent_present)
    print(json.dumps({"out": str(out_dir), "counts": counts}, ensure_ascii=False,
                     indent=2))
    return 0


def write_review_page(out_dir: Path, pairs: list[dict], counts: dict) -> None:
    type_label = {"add": "添加", "delete": "删除", "replace": "替换"}
    items = []
    nav = []
    for pair in pairs:
        nav.append(f'<li><a href="#{pair["pair_id"]}">{pair["pair_id"]}</a>'
                   f' — {pair["generator"]} / {type_label[pair["edit_type"]]}</li>')
        items.append(f"""
<article id="{pair['pair_id']}">
  <h2>{pair['pair_id']}</h2>
  <p class="meta"><b>{pair['generator']}</b> · {type_label[pair['edit_type']]} ·
  {pair['edit_description']}<br>
  源：<code>{pair['source_id']}</code> · mask 像素 {pair['mask_pixels']} ·
  bbox {pair['mask_bbox_xyxy']}</p>
  <div class="grid">
    <figure><img loading="lazy" src="../{pair['original_path']}" alt="original">
      <figcaption>原图 · 整图 AI-generated / 局部 Not tampered</figcaption></figure>
    <figure><img loading="lazy" src="../{pair['edited_path']}" alt="edited">
      <figcaption>编辑版 · 整图 AI-generated / 局部 Tampered</figcaption></figure>
    <figure><img loading="lazy" class="mask" src="../{pair['mask_path']}" alt="mask">
      <figcaption>mask · 白色为编辑区域</figcaption></figure>
  </div>
</article>""")
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GenImage 整图生成 / 局部编辑配对审核</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
       max-width: 1500px; margin: 24px auto; padding: 0 20px; }}
h1 {{ margin-bottom: 4px; }}
.sub {{ color: #666; margin-top: 0; }}
nav ul {{ columns: 2; padding-left: 20px; }}
article {{ border-top: 1px solid #bbb; padding: 18px 0 8px; }}
.meta {{ color: #555; }}
.grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
figure {{ margin: 0; }}
figure img {{ width: 100%; height: auto; max-height: 72vh; object-fit: contain;
             background: #111; border-radius: 4px; }}
figure.mask img {{ image-rendering: pixelated; }}
figcaption {{ font-size: 13px; color: #666; margin-top: 4px; }}
code {{ font-size: 12px; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }}
  nav ul {{ columns: 1; }} }}
</style></head><body>
<h1>GenImage 整图生成 / 局部编辑配对审核</h1>
<p class="sub">共 {counts['pairs']} 对：添加 {counts['by_edit_type']['add']}、
删除 {counts['by_edit_type']['delete']}、替换 {counts['by_edit_type']['replace']}。
两图整图真值均为 AI-generated；原图局部 Not tampered，编辑版局部 Tampered。
mask 为编码前实际变化像素。</p>
<p class="sub">本轮只交付数据、manifest 和审核页；尚未运行 TruFor / Agent。
编辑是否自然、mask 是否对齐，请按对检查。</p>
<nav><ul>{''.join(nav)}</ul></nav>
{''.join(items)}
</body></html>
"""
    (out_dir / "review" / "index.html").write_text(html, encoding="utf-8")


def write_readme(out_dir: Path, counts: dict, recent_present: list[str]) -> None:
    text = f"""# GenImage 整图生成 / 局部编辑配对（2026-11-28）

本轮交付 {counts['pairs']} 对数据、配对 manifest 和审核页；未运行 TruFor / Agent。

## 目录

- `manifest.json`：逐对记录源图、生成器、编辑类型、两图与 mask 路径、真值、编码参数。
- `pairs/<pair_id>/original.jpg`：原图（整图 AI-generated，局部 Not tampered）。
- `pairs/<pair_id>/edited.jpg`：局部编辑版（整图 AI-generated，局部 Tampered）。
- `pairs/<pair_id>/mask.png`：编码前实际变化像素，白色为编辑区域。
- `review/index.html`：原图 / 编辑版 / mask 并排审核页。
- `selection_audit.csv`：源图 split、no_known_use、近期使用审计。
- `encode_audit.csv`：尺寸、mask 像素、JPEG 参数、非编辑区变化审计。

## 数量

- 添加 {counts['by_edit_type']['add']}、删除 {counts['by_edit_type']['delete']}、
  替换 {counts['by_edit_type']['replace']}。
- 生成器分布：{json.dumps(counts['by_generator'], ensure_ascii=False)}。

## 选源与真值

- 全部来自 `dataset/GenImage_q95` 的 dev split，且 `no_known_use=true`。
- 已排除近期训练、验证、SFT 和固定对照 manifests 使用的源；test split 未使用。
- 原图是 GenImage generator-only q95 派生，无局部编辑记录；编辑版由本批次施加一次
  局部添加/删除/替换，mask 由编码前逐像素差异生成。
- 两图整图真值均为 `AI-generated`；局部真值分别为 `Not tampered` / `Tampered`。

## 编码

- Pillow `quality=95, subsampling=2, optimize=False, progressive=False`。
- 原图和编辑版都从同一份解码源图出发，用相同参数编码。
- 已校验两图 JPEG quantization tables 一致；mask 外像素在编码前变化为 0。

## 边界

- 只做数据、manifest、审核页；没有跑 TruFor、Agent、Qwen，也没有改专家或阈值。
- 编辑自然度和 mask 对齐需人工复核；未通过的对可以丢弃或重做。
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
