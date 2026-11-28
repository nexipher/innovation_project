"""Independent TruFor analysis of the 15 AutoSplice validation images.

Runs TruFor with the runtime weights/preprocessing on 10 forged + 5 authentic
validation images, measures score/map-vs-mask overlap, and checks whether the
delivered TruFor three-panel crop covers the mask.  It also joins the base/LoRA
validation records to classify each arm:
  not_called, weak_or_mislocalized, panels_miss_region,
  usable_but_agent_missed, detected.
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

SMALL = ROOT / "sft_data/lora_small_20261128"
MANIFEST = SMALL / "validation_manifest.json"
SPLIT = ROOT / "sft_data/sft_candidates_20261127/split.json"
OUT = SMALL / "analysis"


def _mask(path: str):
    m = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit(f"mask unreadable: {path}")
    return m


def _metrics(map_array, conf_array, mask, crop):
    anomaly = np.asarray(map_array, dtype=np.float64)
    h, w = anomaly.shape
    high = anomaly >= 0.5
    peak_y, peak_x = np.unravel_index(int(np.argmax(anomaly)), anomaly.shape)
    out = {
        "anomaly_peak": round(float(anomaly.max()), 6),
        "high_fraction": round(float(high.mean()), 6),
        "peak_norm": [int(round(peak_y / max(1, h - 1) * 1000)),
                      int(round(peak_x / max(1, w - 1) * 1000))],
    }
    if conf_array is not None:
        conf = np.asarray(conf_array, dtype=np.float64)
        out["conf_mean"] = round(float(conf.mean()), 6)
        out["low_conf_fraction"] = round(float((conf < 0.5).mean()), 6)
    if mask is None:
        return out
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    inter = int(np.logical_and(resized, high).sum())
    union = int(np.logical_or(resized, high).sum())
    out.update({
        "mask_pixels": int(resized.sum()),
        "high_pixels": int(high.sum()),
        "iou": round(inter / union, 6) if union else None,
        "precision": round(inter / int(high.sum()), 6) if high.sum() else None,
        "recall": round(inter / int(resized.sum()), 6) if resized.sum() else None,
        "peak_in_mask": bool(resized[peak_y, peak_x]),
    })
    return out


def _crop_coverage(mask, crop, shape):
    if mask is None or not crop:
        return None
    h, w = shape
    resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    if not resized.any():
        return None
    ymin, xmin, ymax, xmax = [float(v) for v in crop]
    y0, y1 = int(round(ymin / 1000 * h)), int(round(ymax / 1000 * h))
    x0, x1 = int(round(xmin / 1000 * w)), int(round(xmax / 1000 * w))
    box = np.zeros_like(resized)
    box[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    return round(int(np.logical_and(resized, box).sum()) / int(resized.sum()), 6)


def _agent(records, sid, arm):
    row = records.get(sid)
    if not row:
        return {"present": False}
    token = next((e for e in row.get("evidence") or []
                  if e.get("source") == "trufor_expert"), None)
    return {
        "present": True,
        "trufor_called": token is not None,
        "trufor_strength": token.get("strength") if token else None,
        "local": row.get("local_manipulation"),
        "overall": row.get("final_verdict"),
        "calls": row.get("expert_calls"),
    }


def _crop_from_trace(trace, arm):
    # Re-read the delivered token from the trace; the record's evidence compact
    # view intentionally omits artifact_layout.
    path = ROOT / trace["trace_path"]
    t = json.loads(path.read_text(encoding="utf-8"))
    for turn in t.get("conversations") or []:
        value = turn.get("value") or ""
        if not value.lstrip().startswith("{"):
            continue
        try:
            token = json.loads(value)
        except ValueError:
            continue
        if token.get("expert") != "trufor_expert":
            continue
        layout = token.get("artifact_layout") or {}
        crop = (layout.get("original_crop") or {}).get("region_normalized_1000")
        return crop
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "trufor_maps").mkdir(exist_ok=True)
    (OUT / "trufor_raw").mkdir(exist_ok=True)
    os.environ.setdefault("TRUFOR_RESULTS", str(OUT / "trufor_raw"))

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    samples = [row for row in manifest["samples"]
               if row["group"] in ("AutoSplice_Forged", "AutoSplice_Authentic")]
    if len(samples) != 15:
        raise SystemExit(f"expected 15 AutoSplice samples, got {len(samples)}")
    images = {row["sample_id"]: ImageUtils.load_image(str(ROOT / row["path"]))
              for row in samples}
    trufor = TruForExpert(device=args.device, cache=False,
                          results_dir=str(OUT / "trufor_raw"))
    reason = trufor.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor unavailable: {reason}")
    results = trufor.analyze_many(images, progress=True)

    base = {row["sample_id"]: row for row in json.loads(
        (SMALL / "validation/base/records.json").read_text(encoding="utf-8"))}
    lora = {row["sample_id"]: row for row in json.loads(
        (SMALL / "validation/lora/records.json").read_text(encoding="utf-8"))}
    rows = []
    for sample in samples:
        sid = sample["sample_id"]
        key = trufor.cache_key(images[sid])
        arrays = dict(np.load(trufor.result_path(key), allow_pickle=True))
        mask = None
        if sample["group"] == "AutoSplice_Forged":
            entry = split["autosplice"]["bases"][str(sample["source_id"])]
            mask = _mask(entry["mask"]["path"])
        crop_base = _crop_from_trace(base[sid], "base") if sid in base else None
        crop_lora = _crop_from_trace(lora[sid], "lora") if sid in lora else None
        metrics = _metrics(arrays["map"], arrays.get("conf"), mask, None)
        map_shape = np.asarray(arrays["map"]).shape
        # Save rendered panels for review.
        trufor._last = arrays
        artifacts = trufor.render_artifacts(images[sid])
        for name, image in artifacts.items():
            cv2.imwrite(str(OUT / "trufor_maps" / f"{sid}__{name}.png"), image)
        row = {
            "sample_id": sid,
            "source_id": sample["source_id"],
            "group": sample["group"],
            "ground_truth": sample["ground_truth"],
            "path": sample["path"],
            "trufor_score": round(float(results[sid].raw_metric), 6),
            **metrics,
            "base_crop_bbox": crop_base,
            "lora_crop_bbox": crop_lora,
            "base_crop_mask_coverage": _crop_coverage(mask, crop_base, map_shape),
            "lora_crop_mask_coverage": _crop_coverage(mask, crop_lora, map_shape),
            "base": _agent(base, sid, "base"),
            "lora": _agent(lora, sid, "lora"),
        }
        rows.append(row)

    def classify(row, arm):
        a = row[arm]
        if not a.get("present"):
            return "missing_record"
        if not a.get("trufor_called"):
            return "not_called"
        if row["group"] == "AutoSplice_Authentic":
            return "false_positive" if a.get("local") == "Tampered" else "true_negative"
        score = row["trufor_score"]
        iou = row.get("iou")
        cov = row.get(f"{arm}_crop_mask_coverage")
        if score < 0.5 or (iou is not None and iou < 0.1) or row.get("high_pixels", 0) == 0:
            return "weak_or_mislocalized"
        if cov is not None and cov < 0.5:
            return "panels_miss_region"
        if a.get("local") == "Tampered":
            return "detected"
        return "usable_but_agent_missed"
    for row in rows:
        row["base_class"] = classify(row, "base")
        row["lora_class"] = classify(row, "lora")

    fields = ["sample_id", "source_id", "group", "ground_truth", "trufor_score",
              "anomaly_peak", "high_fraction", "conf_mean", "low_conf_fraction",
              "iou", "precision", "recall", "peak_in_mask",
              "base_crop_bbox", "lora_crop_bbox",
              "base_crop_mask_coverage", "lora_crop_mask_coverage",
              "base_class", "lora_class",
              "base_trufor_strength", "base_local", "base_overall",
              "lora_trufor_strength", "lora_local", "lora_overall"]
    with (OUT / "per_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for arm in ("base", "lora"):
                for key in ("trufor_strength", "local", "overall"):
                    out[f"{arm}_{key}"] = (row[arm] or {}).get(key)
            writer.writerow({key: json.dumps(out.get(key), ensure_ascii=False)
                             if isinstance(out.get(key), (list, dict)) else out.get(key)
                             for key in fields})
    (OUT / "analysis.json").write_text(
        json.dumps({"samples": rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    lines = [
        "# 15 张 AutoSplice 验证图 — TruFor 独立分析",
        "",
        "判读：score、异常图/mask 的 IoU/recall、三图裁剪对 mask 的覆盖率，以及",
        "base/LoRA 的实际 local 标签。分类：not_called / weak_or_mislocalized /",
        "panels_miss_region / usable_but_agent_missed / detected；真图为",
        "true_negative / false_positive。",
        "",
        "| sample | group | score | IoU | recall | base crop | LoRA crop | base class | lora class |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sample_id']}` | {row['group']} | {row['trufor_score']} | "
            f"{row.get('iou')} | {row.get('recall')} | "
            f"{row.get('base_crop_mask_coverage')} | {row.get('lora_crop_mask_coverage')} | "
            f"{row['base_class']} | {row['lora_class']} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "report": str(OUT / "report.md"),
                      "base_classes": {k: sum(1 for r in rows if r['base_class'] == k)
                                       for k in sorted({r['base_class'] for r in rows})},
                      "lora_classes": {k: sum(1 for r in rows if r['lora_class'] == k)
                                       for k in sorted({r['lora_class'] for r in rows})}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
