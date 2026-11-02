"""
Build the fixed GenImage JPEG-q95 evaluation inputs (CPU only; plan §10.1).

Sources:
  * dataset/GenImage_Test/<generator>/*.png  (8 generators, 1000 each)
  * dataset/Real/*.jpg                       (1000 shared real images)

Output:
  * dataset/GenImage_q95/<generator>/<stem>.jpg
  * dataset/Real_q95/<stem>.jpg
  * dataset/GenImage_q95/manifest.json

The encoder is the same identity as the existing MNW q95 inputs: Pillow,
quality 95, 4:2:0 subsampling, no optimize/progressive, no resize.  Original
files are never modified.  The manifest records both source and derived
hashes; images are ignored by Git, the manifest is tracked.

  python scripts/build_genimage_q95.py --dry-run
  python scripts/build_genimage_q95.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, features  # noqa: E402

from config import PROJECT_ROOT  # noqa: E402

GENIMAGE_ROOT = os.path.join(PROJECT_ROOT, "dataset", "GenImage_Test")
REAL_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Real")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "GenImage_q95")
REAL_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Real_q95")
MANIFEST = os.path.join(OUTPUT_ROOT, "manifest.json")
JPEG_PARAMS = {"quality": 95, "subsampling": 2,
               "optimize": False, "progressive": False}


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


def _source_rows():
    rows = []
    for group_dir in sorted(Path(GENIMAGE_ROOT).iterdir()):
        if not group_dir.is_dir():
            continue
        for source in sorted(group_dir.iterdir()):
            if source.is_file() and source.suffix.lower() in (".png", ".jpg", ".jpeg"):
                rows.append({"source": str(source), "group": group_dir.name,
                             "label": "Fake"})
    for source in sorted(Path(REAL_ROOT).iterdir()):
        if source.is_file() and source.suffix.lower() in (".png", ".jpg", ".jpeg"):
            rows.append({"source": str(source), "group": "Real", "label": "Real"})
    return rows


def build(dry_run: bool = False) -> dict:
    rows = _source_rows()
    if not rows:
        raise SystemExit("没有找到 GenImage/Real 源文件")

    # Source hashes first, so duplicate content can be marked consistently.
    source_hashes = {}
    for row in rows:
        source_hashes[row["source"]] = sha256_of(row["source"])
    by_hash = {}
    for row in rows:
        by_hash.setdefault(source_hashes[row["source"]], []).append(row["source"])

    encoder = encoder_identity()
    entries = []
    problems = []
    for row in rows:
        source = row["source"]
        source_hash = source_hashes[source]
        with Image.open(source) as image:
            width, height = image.size
            mode_before = image.mode
            rgb = image.convert("RGB")
            if rgb.size != (width, height):
                problems.append(f"尺寸变化：{source}")
                continue
            if row["group"] == "Real":
                target = os.path.join(REAL_OUTPUT_ROOT,
                                      os.path.splitext(os.path.basename(source))[0] + ".jpg")
            else:
                target = os.path.join(OUTPUT_ROOT, row["group"],
                                      os.path.splitext(os.path.basename(source))[0] + ".jpg")
            if not dry_run:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                rgb.save(target, format="JPEG", **JPEG_PARAMS)
        if dry_run:
            derived_hash = None
            derived_bytes = None
        else:
            if not os.path.exists(target):
                problems.append(f"未生成：{target}")
                continue
            with Image.open(target) as check:
                if check.size != (width, height):
                    problems.append(f"派生尺寸变化：{target}")
            derived_hash = sha256_of(target)
            derived_bytes = os.path.getsize(target)
        entries.append({
            "path": os.path.relpath(target, PROJECT_ROOT),
            "condition": "jpeg_q95", "group": row["group"],
            "label": row["label"], "generator": row["group"],
            "source_path": os.path.relpath(source, PROJECT_ROOT),
            "source_sha256": source_hash,
            "source_bytes": os.path.getsize(source),
            "sha256": derived_hash, "bytes": derived_bytes,
            "width": width, "height": height,
            "source_mode": mode_before,
            "mode_converted_from": None if mode_before == "RGB" else mode_before,
            "source_id": f"{row['group']}/{os.path.splitext(os.path.basename(source))[0]}",
            "split": "unassigned", "treatment": "jpeg_q95", "container": "jpg",
            "duplicate_group": (
                by_hash[source_hash][0] if len(by_hash[source_hash]) > 1 else None),
        })

    if problems:
        raise SystemExit("q95 构建失败：\n  " + "\n  ".join(problems))
    if dry_run:
        return {"entries": entries, "written": False,
                "counts": {"total": len(entries), "fake": sum(e["label"] == "Fake" for e in entries),
                           "real": sum(e["label"] == "Real" for e in entries)}}

    payload = {
        "schema": "genimage_q95_inputs/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_roots": {"genimage": "dataset/GenImage_Test", "real": "dataset/Real"},
        "output_roots": {"genimage": "dataset/GenImage_q95", "real": "dataset/Real_q95"},
        "encoder": encoder,
        "rules": {
            "no_resize": True,
            "originals_untouched": True,
            "same_encoder_for_real_and_fake": True,
            "q85_q70_not_built": True,
            "no_threshold_fitting": True,
            "note": ("GenImage fakes have different native resolutions (128–1024); "
                     "the shared Real pool is 500px-level. No one-to-one source pairing "
                     "is claimed; each generator group uses the same 1000 Real images."),
        },
        "pairing": {"real_pool": "dataset/Real", "one_to_one": False,
                    "note": "每组假图与共享 Real 池配对；Real 在八组间复用，组间不独立。"},
        "duplicates": {h: paths for h, paths in by_hash.items() if len(paths) > 1},
        "counts": {"total": len(entries),
                   "fake": sum(e["label"] == "Fake" for e in entries),
                   "real": sum(e["label"] == "Real" for e in entries),
                   "by_group": {g: sum(e["group"] == g for e in entries)
                                for g in sorted({e["group"] for e in entries})}},
        "inputs": entries,
    }
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return {"entries": entries, "written": True, "counts": payload["counts"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = build(dry_run=args.dry_run)
    print(json.dumps(result["counts"], ensure_ascii=False))
    if result["written"]:
        print(f"manifest: {os.path.relpath(MANIFEST, PROJECT_ROOT)}")
    else:
        print("（dry-run）未写文件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
