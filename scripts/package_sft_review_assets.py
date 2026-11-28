"""Package every image shown on the 24 candidate pages under review_assets/.

The run environment has the q95 inputs, source images, TruFor panels and PROBE
crops in gitignored dataset/trace directories.  A reviewer who clones the
repository cannot see them.  This script copies those exact files (no
re-measurement) into a per-sample directory and rewrites the already-rendered
page image links to batch-relative `review_assets/...` paths.

It does not change records, traces, tool outputs or verdicts.  The original
runtime paths stay in records/traces for provenance; only the HTML view points
at the packaged copies.  Run from the repository root.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"

ROLE_BY_NAME = (
    ("model_original_crop", "trufor_original_crop.png"),
    ("model_anomaly_map", "trufor_anomaly_map.png"),
    ("model_confidence_map", "trufor_confidence_map.png"),
    ("region_", "probe_region.png"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_one(src_rel: str, dest: Path, name: str) -> dict:
    src = ROOT / src_rel
    if not src.is_file():
        raise SystemExit(f"missing source asset: {src}")
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    shutil.copy2(src, target)
    return {
        "original": src_rel,
        "packaged": str(target.relative_to(ROOT)),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def panel_name(path: str) -> str | None:
    base = os.path.basename(path)
    for needle, name in ROLE_BY_NAME:
        if needle in base:
            return name
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    records = json.loads((out / "records.json").read_text(encoding="utf-8"))
    assets_root = out / "review_assets"
    if assets_root.exists():
        shutil.rmtree(assets_root)
    assets_root.mkdir(parents=True)

    manifest = []
    for record in records:
        sid = record["sample_id"]
        dest = assets_root / sid
        entries = {}
        suffix = Path(record["image_path"]).suffix.lower() or ".jpg"
        if suffix == ".jpeg":
            suffix = ".jpg"
        entries["input"] = copy_one(record["image_path"], dest, f"input{suffix}")
        src_suffix = Path(record["source_image"]).suffix.lower() or ".jpg"
        if src_suffix == ".jpeg":
            src_suffix = ".jpg"
        entries["source"] = copy_one(record["source_image"], dest,
                                     f"source{src_suffix}")

        trace = json.loads((ROOT / record["trace_path"]).read_text(encoding="utf-8"))
        panels = {}
        for turn in trace.get("conversations") or []:
            for path in turn.get("image_paths") or []:
                name = panel_name(path)
                if name and name not in panels:
                    panels[name] = copy_one(path, dest, name)
        for required in ("trufor_original_crop.png", "trufor_anomaly_map.png",
                         "trufor_confidence_map.png", "probe_region.png"):
            if required not in panels:
                raise SystemExit(f"{sid}: missing {required} in trace image_paths")
        entries["trufor_original_crop"] = panels["trufor_original_crop.png"]
        entries["trufor_anomaly_map"] = panels["trufor_anomaly_map.png"]
        entries["trufor_confidence_map"] = panels["trufor_confidence_map.png"]
        entries["probe_region"] = panels["probe_region.png"]

        overlays = []
        for index, overlay in enumerate(record.get("review_overlays") or [], 1):
            name = f"overlay_{index}_{os.path.basename(overlay)}"
            overlays.append(copy_one(overlay, dest, name))
        entries["overlays"] = overlays
        manifest.append({"sample_id": sid, "assets": entries})

    # Rewrite the rendered pages.  Both the src value (page-relative) and the
    # provenance text (project-relative) are replaced with the packaged path.
    replacements = []
    for item in manifest:
        sid = item["sample_id"]
        for entry in item["assets"].values():
            if isinstance(entry, list):
                entries = entry
            else:
                entries = [entry]
            for asset in entries:
                old = asset["original"]
                packaged = asset["packaged"]
                rel = os.path.relpath(ROOT / packaged, out).replace(os.sep, "/")
                variants = {old, os.path.relpath(ROOT / old, out).replace(os.sep, "/")}
                for variant in variants:
                    replacements.append((variant, rel))
    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    pages = sorted(out.glob("[0-9][0-9][0-9]__*.html"))
    if len(pages) != 24:
        raise SystemExit(f"expected 24 pages, got {len(pages)}")
    for page in pages:
        text = page.read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        page.write_text(text, encoding="utf-8")
        srcs = re.findall(r'src="([^"]+)"', text)
        missing = [src for src in srcs if not (out / src).resolve().is_file()]
        if missing:
            raise SystemExit(f"{page.name}: missing packaged images {missing[:5]}")
        wrong = [src for src in srcs if not src.startswith("review_assets/")]
        if wrong:
            raise SystemExit(f"{page.name}: non-packaged src {wrong[:5]}")
        if len(srcs) < 6:
            raise SystemExit(f"{page.name}: only {len(srcs)} images")
        for required in ("trufor_original_crop", "trufor_anomaly_map",
                         "trufor_confidence_map", "probe_region"):
            if not any(required in src for src in srcs):
                raise SystemExit(f"{page.name}: missing {required}")

    (assets_root / "ASSET_MANIFEST.json").write_text(
        json.dumps({"schema": "sft_review_assets/1", "samples": manifest},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total_bytes = sum(asset["bytes"] for item in manifest
                      for entry in item["assets"].values()
                      for asset in (entry if isinstance(entry, list) else [entry]))
    lines = [
        "# 审核图片包",
        "",
        f"- 24 条案例，逐条目录；共 {sum(len(item['assets']) for item in manifest)} 类图片。",
        f"- 总字节：{total_bytes}（约 {total_bytes / 1e6:.1f} MB）。",
        "- 文件按 `original` 路径从运行环境复制，未重新生成、未重新测量。",
        "- 页面里的原图、源图、TruFor 原图裁剪/异常图/可靠性图、PROBE 裁剪、预测框叠加",
        "  均指向本目录；原始 trace 仍保留运行时路径。",
        "- 映射与 sha256 见 `ASSET_MANIFEST.json`。",
    ]
    (assets_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(assets_root),
        "samples": len(manifest),
        "pages": len(pages),
        "total_mb": round(total_bytes / 1e6, 2),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
