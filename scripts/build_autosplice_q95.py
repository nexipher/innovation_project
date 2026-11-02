"""
Build AutoSplice q95 inputs plus an explicit source -> variant -> mask manifest.

Source layout (dataset/AutoSplice/):
  Authentic/<base>.jpg
  Forged_JPEG100/<base>_<variant>.jpg
  Mask/<base>_mask.png
  Caption/<base>.json

Only Authentic and Forged_JPEG100 are used.  The q95 encoder is the same
identity as the MNW and GenImage inputs: Pillow, quality 95, 4:2:0, no
optimize/progressive, no resize.  Masks are kept lossless and never resized.

The pairing manifest is deliberately explicit: one base may have 1-3 forged
variants but only one mask/caption, so the manifest marks the mask as
shared-by-base rather than pretending each variant has its own mask.  A
deterministic audit samples bases and checks that the pixel difference between
Authentic and Forged_JPEG100 is much larger inside the mask than outside.

  python scripts/build_autosplice_q95.py --dry-run
  python scripts/build_autosplice_q95.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, features  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402

BASE = os.path.join(PROJECT_ROOT, "dataset", "AutoSplice")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "AutoSplice_q95")
AUTH_OUT = os.path.join(OUTPUT_ROOT, "Authentic")
FORGED_OUT = os.path.join(OUTPUT_ROOT, "Forged_JPEG100")
PAIRING_MANIFEST = os.path.join(OUTPUT_ROOT, "pairing_manifest.json")
INPUTS_MANIFEST = os.path.join(OUTPUT_ROOT, "inputs_manifest.json")
JPEG_PARAMS = {"quality": 95, "subsampling": 2,
               "optimize": False, "progressive": False}
AUDIT_SEED = 20261102


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encoder_identity() -> dict:
    return {"library": f"Pillow {Image.__version__}",
            "jpeg_backend": features.version("jpg") or "unknown",
            "params": dict(JPEG_PARAMS)}


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT)


def _encode(source: str, target: str, dry_run: bool) -> dict:
    with Image.open(source) as image:
        width, height = image.size
        mode = image.mode
        rgb = image.convert("RGB")
        if rgb.size != (width, height):
            raise RuntimeError(f"尺寸变化：{source}")
        if not dry_run:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            rgb.save(target, format="JPEG", **JPEG_PARAMS)
    result = {"path": _rel(target), "width": width, "height": height,
              "source_mode": mode}
    if not dry_run:
        with Image.open(target) as check:
            if check.size != (width, height):
                raise RuntimeError(f"派生尺寸变化：{target}")
        result.update({"sha256": sha256_of(target),
                       "bytes": os.path.getsize(target)})
    return result


def _mask_info(path: str) -> dict:
    import numpy as np

    with Image.open(path) as image:
        mode = image.mode
        array = np.asarray(image.convert("L"))
    nonzero = array > 0
    ys, xs = np.nonzero(nonzero)
    bbox = ([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
            if len(xs) else None)
    return {"path": _rel(path), "sha256": sha256_of(path),
            "bytes": os.path.getsize(path), "width": int(array.shape[1]),
            "height": int(array.shape[0]), "mode": mode,
            "values": sorted({int(v) for v in np.unique(array)}),
            "area_fraction": float(nonzero.mean()), "bbox": bbox}


def _audit(base_rows: list, sample_per_stratum: int = 10) -> dict:
    """Sample bases by variant count; compare diff inside/outside the mask."""
    import numpy as np

    base_rows = [row for row in base_rows if row["forged"] and row["mask"]]
    strata = defaultdict(list)
    for row in base_rows:
        strata[len(row["forged"])].append(row["source_id"])
    rng = random.Random(AUDIT_SEED)
    sampled = []
    for count in sorted(strata):
        ids = sorted(strata[count])
        rng.shuffle(ids)
        sampled.extend(ids[:sample_per_stratum])
    sampled = sorted(set(sampled))

    records = []
    for row in base_rows:
        if row["source_id"] not in sampled:
            continue
        with Image.open(os.path.join(PROJECT_ROOT, row["authentic"]["path"])) as im:
            authentic = np.asarray(im.convert("RGB"), dtype=np.int16)
        with Image.open(os.path.join(PROJECT_ROOT, row["mask"]["path"])) as im:
            mask = np.asarray(im.convert("L")) > 0
        for forged in row["forged"]:
            with Image.open(os.path.join(PROJECT_ROOT, forged["source_path"])) as im:
                fake = np.asarray(im.convert("RGB"), dtype=np.int16)
            diff = np.abs(fake - authentic).mean(axis=2)
            inside = float(diff[mask].mean()) if mask.any() else None
            outside = float(diff[~mask].mean()) if (~mask).any() else None
            records.append({
                "source_id": row["source_id"], "variant": forged["variant"],
                "forged_path": forged["source_path"],
                "mask_path": row["mask"]["path"],
                "width": row["authentic"]["width"],
                "height": row["authentic"]["height"],
                "mask_area_fraction": row["mask"]["area_fraction"],
                "diff_inside_mean": round(inside, 3) if inside is not None else None,
                "diff_outside_mean": round(outside, 3) if outside is not None else None,
                "inside_over_outside": (round(inside / outside, 3)
                                        if inside is not None and outside else None),
                "inside_gt_outside": (inside > outside
                                      if inside is not None and outside is not None else None),
            })
    return {
        "seed": AUDIT_SEED,
        "sample_per_stratum": sample_per_stratum,
        "sampled_bases": sampled,
        "sampled_base_count": len(sampled),
        "sampled_variant_count": len(records),
        "all_sampled_inside_gt_outside": all(r["inside_gt_outside"] for r in records),
        "records": records,
        "note": ("Mask is one per base and is shared by all forged variants of that "
                 "base; the audit checks the sampled variants against the same mask."),
    }


def build(dry_run: bool = False) -> dict:
    base = Path(BASE)
    authentic = {p.stem: p for p in (base / "Authentic").glob("*.jpg")}
    forged = defaultdict(list)
    for path in (base / "Forged_JPEG100").glob("*.jpg"):
        forged[path.stem.split("_")[0]].append(path)
    masks = {p.stem.replace("_mask", ""): p
             for p in (base / "Mask").glob("*.png")}
    captions = {p.stem: p for p in (base / "Caption").glob("*.json")}

    problems = []
    if set(forged) - set(authentic):
        problems.append(f"forged bases without authentic: {len(set(forged)-set(authentic))}")
    if set(masks) - set(authentic):
        problems.append(f"mask bases without authentic: {len(set(masks)-set(authentic))}")
    if set(captions) - set(authentic):
        problems.append(f"caption bases without authentic: {len(set(captions)-set(authentic))}")
    if problems:
        raise SystemExit("AutoSplice 配对失败：\n  " + "\n  ".join(problems))

    encoder = encoder_identity()
    source_rows = []
    inputs = []
    size_mismatches = []
    for source_id in sorted(authentic):
        auth = authentic[source_id]
        auth_q95 = _encode(str(auth), os.path.join(AUTH_OUT, source_id + ".jpg"), dry_run)
        mask_info = _mask_info(str(masks[source_id])) if source_id in masks else None
        caption = None
        if source_id in captions:
            caption = {"path": _rel(str(captions[source_id])),
                       "sha256": sha256_of(str(captions[source_id])),
                       "bytes": os.path.getsize(captions[source_id])}
        variants = []
        for path in sorted(forged.get(source_id, [])):
            suffix = path.stem[len(source_id) + 1:]
            out = _encode(str(path), os.path.join(FORGED_OUT, path.stem + ".jpg"), dry_run)
            if mask_info and (out["width"], out["height"]) != (mask_info["width"], mask_info["height"]):
                size_mismatches.append({"source_id": source_id, "variant": suffix,
                                        "image": out["path"], "mask": mask_info["path"]})
            variants.append({
                "variant": suffix, "source_path": _rel(str(path)),
                "source_sha256": sha256_of(str(path)),
                "source_bytes": os.path.getsize(path),
                "width": out["width"], "height": out["height"],
                "source_mode": out["source_mode"],
                "q95_path": out["path"], "q95_sha256": out.get("sha256"),
                "q95_bytes": out.get("bytes"),
                "mask_shared_by_base": mask_info["path"] if mask_info else None,
            })
        source_rows.append({
            "source_id": source_id,
            "authentic": {"path": _rel(str(auth)), "sha256": sha256_of(str(auth)),
                          "bytes": os.path.getsize(auth),
                          "width": auth_q95["width"], "height": auth_q95["height"],
                          "source_mode": auth_q95["source_mode"],
                          "q95_path": auth_q95["path"],
                          "q95_sha256": auth_q95.get("sha256"),
                          "q95_bytes": auth_q95.get("bytes")},
            "mask": mask_info,
            "caption": caption,
            "forged": variants,
            "variant_count": len(variants),
        })
        # Inputs: authentic real image
        inputs.append({
            "variant_id": f"autosplice__{source_id}__authentic__q95",
            "condition": "jpeg_q95", "path": auth_q95["path"],
            "sha256": auth_q95.get("sha256"), "bytes": auth_q95.get("bytes"),
            "source_sha256": sha256_of(str(auth)), "source_path": _rel(str(auth)),
            "group": "Authentic", "label": "Real", "generator": "AutoSplice",
            "source_id": source_id, "treatment": "jpeg_q95", "container": "jpg",
            "width": auth_q95["width"], "height": auth_q95["height"],
            "encoder": encoder, "split": "unassigned",
            "mask_path": mask_info["path"] if mask_info else None,
            "mask_sha256": mask_info["sha256"] if mask_info else None,
            "caption_path": caption["path"] if caption else None,
        })
        for forged_row in variants:
            inputs.append({
                "variant_id": f"autosplice__{source_id}__{forged_row['variant']}__q95",
                "condition": "jpeg_q95", "path": forged_row["q95_path"],
                "sha256": forged_row.get("q95_sha256"), "bytes": forged_row.get("q95_bytes"),
                "source_sha256": forged_row["source_sha256"],
                "source_path": forged_row["source_path"],
                "group": "Forged_JPEG100", "label": "Fake", "generator": "AutoSplice",
                "source_id": source_id, "variant": forged_row["variant"],
                "treatment": "jpeg_q95", "container": "jpg",
                "width": forged_row["width"], "height": forged_row["height"],
                "encoder": encoder, "split": "unassigned",
                "mask_path": mask_info["path"] if mask_info else None,
                "mask_sha256": mask_info["sha256"] if mask_info else None,
                "caption_path": caption["path"] if caption else None,
            })

    if size_mismatches:
        problems.append(f"mask/image size mismatches: {len(size_mismatches)}")
    if problems:
        raise SystemExit("q95/配对构建失败：\n  " + "\n  ".join(problems))

    audit = _audit(source_rows)
    payload = {
        "schema": "autosplice_q95/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_root": "dataset/AutoSplice",
        "output_root": "dataset/AutoSplice_q95",
        "encoder": encoder,
        "rules": {
            "only_authentic_and_jpeg100": True,
            "no_resize": True,
            "mask_lossless": True,
            "originals_untouched": True,
            "q85_q70_not_built": True,
            "no_threshold_fitting": True,
        },
        "counts": {
            "authentic": len(authentic),
            "forged": sum(len(v) for v in forged.values()),
            "bases_with_forged": len(forged),
            "authentic_only": len(set(authentic) - set(forged)),
            "mask": len(masks),
            "caption": len(captions),
            "inputs": len(inputs),
            "forged_by_variant_count": dict(Counter(len(v) for v in forged.values())),
        },
        "pairing": {
            "source": "Authentic/<base>.jpg",
            "operation": "Forged_JPEG100/<base>_<variant>.jpg",
            "mask": "Mask/<base>_mask.png",
            "caption": "Caption/<base>.json",
            "mask_shared_by_base": True,
            "note": ("one mask/caption per base; 1-3 forged variants share the base "
                     "mask. The manifest records this instead of inventing per-variant masks."),
        },
        "audit": audit,
        "sources": source_rows,
        "authentic_only": [s["source_id"] for s in source_rows if not s["forged"]],
        "inputs": inputs,
    }
    if not dry_run:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        with open(PAIRING_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump({k: payload[k] for k in
                       ("schema", "created_at", "source_root", "output_root",
                        "encoder", "rules", "counts", "pairing", "audit",
                        "sources", "authentic_only")},
                      handle, ensure_ascii=False, indent=2)
        with open(INPUTS_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump({"schema": "autosplice_inputs/1",
                       "created_at": payload["created_at"],
                       "encoder": encoder,
                       "counts": {"total": len(inputs),
                                  "real": sum(i["label"] == "Real" for i in inputs),
                                  "fake": sum(i["label"] == "Fake" for i in inputs)},
                       "inputs": inputs}, handle, ensure_ascii=False, indent=2)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    payload = build(dry_run=args.dry_run)
    print(json.dumps(payload["counts"], ensure_ascii=False))
    print(json.dumps({"audit_bases": payload["audit"]["sampled_base_count"],
                      "audit_variants": payload["audit"]["sampled_variant_count"],
                      "all_inside_gt_outside": payload["audit"]["all_sampled_inside_gt_outside"]},
                     ensure_ascii=False))
    if not args.dry_run:
        print(f"pairing: {_rel(PAIRING_MANIFEST)}")
        print(f"inputs: {_rel(INPUTS_MANIFEST)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
