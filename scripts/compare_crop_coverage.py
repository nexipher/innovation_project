"""Compare delivered TruFor crop coverage before/after the 2026-11-28 fix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
SPLIT = ROOT / "sft_data/sft_candidates_20261127/split.json"


def _crop(record: dict):
    path = ROOT / record["trace_path"]
    if not path.exists():
        return None, None
    trace = json.loads(path.read_text(encoding="utf-8"))
    for turn in trace.get("conversations") or []:
        if turn.get("from") != "user":
            continue
        value = turn.get("value") or ""
        if not value.lstrip().startswith("{"):
            continue
        try:
            token = json.loads(value)
        except ValueError:
            continue
        if token.get("expert") != "trufor_expert":
            continue
        layout = token.get("model_image_layout") or token.get("artifact_layout") or {}
        original = layout.get("original_crop") or {}
        return (original.get("region_normalized_1000"),
                original.get("region_source") or original.get("inspection_region_source"))
    return None, None


def _npz_for(analysis_row: dict):
    score = analysis_row["trufor_score"]
    for path in (SMALL / "analysis/trufor_raw").glob("*/result.npz"):
        arrays = np.load(path, allow_pickle=True)
        if "score" not in arrays:
            continue
        if abs(float(np.asarray(arrays["score"]).reshape(-1)[0]) - score) < 1e-5:
            return {key: arrays[key] for key in arrays.files}
    return None


def _mask(source_id: str):
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entry = split["autosplice"]["bases"][str(source_id)]
    path = Path(entry["mask"]["path"])
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _coverage(mask, crop, shape):
    if mask is None or not crop:
        return None
    h, w = shape
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    ymin, xmin, ymax, xmax = [float(v) for v in crop]
    y0, y1 = int(round(ymin / 1000 * h)), int(round(ymax / 1000 * h))
    x0, x1 = int(round(xmin / 1000 * w)), int(round(xmax / 1000 * w))
    box = np.zeros_like(mask)
    box[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return round(int(np.logical_and(mask, box).sum()) / int(mask.sum()), 6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", default=str(SMALL / "validation_fixed"))
    parser.add_argument("--out", default=str(SMALL / "analysis/crop_coverage_fix.md"))
    args = parser.parse_args()
    fixed = Path(args.fixed)
    analysis = json.loads((SMALL / "analysis/analysis.json").read_text(
        encoding="utf-8"))["samples"]
    before = {arm: {row["sample_id"]: row for row in json.loads(
        (SMALL / f"validation/{arm}/records.json").read_text(encoding="utf-8"))}
        for arm in ("base", "lora")}
    after = {arm: {row["sample_id"]: row for row in json.loads(
        (fixed / arm / "records.json").read_text(encoding="utf-8"))}
        for arm in ("base", "lora")}
    rows = []
    for sample in analysis:
        if sample["group"] != "AutoSplice_Forged":
            continue
        sid = sample["sample_id"]
        arrays = _npz_for(sample)
        if arrays is None:
            raise SystemExit(f"no npz for {sid}")
        mask = _mask(sample["source_id"])
        shape = np.asarray(arrays["map"]).shape
        row = {"sample_id": sid, "source_id": sample["source_id"]}
        for arm in ("base", "lora"):
            for side, source in (("before", before), ("after", after)):
                crop, source_name = _crop(source[arm][sid])
                row[f"{arm}_{side}_crop"] = crop
                row[f"{arm}_{side}_source"] = source_name
                row[f"{arm}_{side}_coverage"] = _coverage(mask, crop, shape)
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# TruFor 三图裁剪对编辑 mask 的覆盖率（修复前后）",
        "",
        "| sample | base before | base after | LoRA before | LoRA after |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(arm):
            cov = row[f"{arm}_after_coverage"]
            src = row[f"{arm}_after_source"]
            before_cov = row[f"{arm}_before_coverage"]
            return f"{before_cov} → {cov} ({src})"
        lines.append(
            f"| `{row['sample_id']}` | {row['base_before_coverage']} | "
            f"{row['base_after_coverage']} | {row['lora_before_coverage']} | "
            f"{row['lora_after_coverage']} |")
    lines += [
        "",
        "覆盖率 = 交付三图内容区域覆盖编辑 mask 的比例；before 是 anomaly-peak 裁剪，"
        "after 是 high-response 包围盒裁剪。未改 TruFor 权重或阈值。",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "out": str(out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
