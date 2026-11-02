"""
Small localization smoke for TruFor on AutoSplice q95 (plan §10.5).

This is a *plumbing / geometry / false-positive* check, not an expert-selection
result.  It samples dev-partition images, runs the official TruFor test.py via
the existing adapter, and records for each image:

  * score, map and conf shapes, finite checks;
  * for forged images: pixel IoU/F1/precision/recall against the GT mask,
    mean map inside/outside the mask, predicted and GT areas;
  * for authentic images: score, high-response area fraction (false-positive
    area), map statistics;
  * batch wall time, per-image time and card peak memory.

The mask is shared by all forged variants of a base (see the AutoSplice pairing
manifest); this script uses the same mask for each sampled variant and says so.

  TRUFOR_HOME=/root/autodl-tmp/third_party/TruFor \
  TRUFOR_PYTHON=/root/miniconda3/bin/python \
  TRUFOR_RESULTS=sft_data/g8_localization/trufor_raw \
  FORENSIC_ALLOW_GPU=1 python scripts/trufor_autosplice_smoke.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from PIL import Image

from config import PROJECT_ROOT  # noqa: E402
from experts.trufor import HIGH_RESPONSE_THRESHOLD, TruForExpert  # noqa: E402

INPUTS = os.path.join(PROJECT_ROOT, "dataset", "AutoSplice_q95",
                      "inputs_manifest.json")
OUT_DIR = os.path.join(PROJECT_ROOT, "sft_data", "g8_localization")
SEED = 20261102


def _read(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _environment() -> dict:
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        devices = int(torch.cuda.device_count())
    except Exception:
        cuda, devices = False, 0
    return {"cgroup_max": _read("/sys/fs/cgroup/memory.max"),
            "cgroup_high": _read("/sys/fs/cgroup/memory.high"),
            "cgroup_current": _read("/sys/fs/cgroup/memory.current"),
            "cgroup_events": _read("/sys/fs/cgroup/memory.events"),
            "torch_cuda_available": cuda,
            "torch_cuda_device_count": devices}


def _load_bgr(path: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])


def _mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def _resize(array: np.ndarray, shape) -> np.ndarray:
    height, width = int(shape[0]), int(shape[1])
    if array.shape[:2] == (height, width):
        return array
    return cv2.resize(array, (width, height), interpolation=cv2.INTER_LINEAR)


def _select(rows: list) -> tuple[list, list]:
    """8 forged from distinct/related bases + 8 authentic, deterministic."""
    forged = [r for r in rows if r["label"] == "Fake" and r.get("mask_path")]
    authentic = [r for r in rows if r["label"] == "Real"]
    by_base = {}
    for row in forged:
        by_base.setdefault(row["source_id"], []).append(row)
    bases_by_count = {}
    for base, variants in by_base.items():
        bases_by_count.setdefault(len(variants), []).append(base)
    rng = random.Random(SEED)
    chosen = []
    # One base with all 3 variants plus bases from the 1/2/3 strata.
    for count, take in ((3, 1), (2, 3), (1, 4)):
        bases = sorted(bases_by_count.get(count, []))
        rng.shuffle(bases)
        for base in bases[:take]:
            chosen.extend(sorted(by_base[base], key=lambda r: r["variant"])[:1]
                          if count != 3 else
                          sorted(by_base[base], key=lambda r: r["variant"]))
    chosen = chosen[:8]
    real_with_forged = [r for r in authentic if r["source_id"] in by_base]
    real_only = [r for r in authentic if r["source_id"] not in by_base]
    rng.shuffle(real_with_forged)
    rng.shuffle(real_only)
    reals = real_with_forged[:4] + real_only[:4]
    return chosen, reals


def _metrics(score: float, map_array: np.ndarray, conf_array, mask, label: str) -> dict:
    map_array = np.asarray(map_array, dtype=np.float64)
    map_array = np.squeeze(map_array)
    if map_array.ndim != 2:
        raise RuntimeError(f"map shape is not 2-D: {map_array.shape}")
    if mask is None:
        mask = np.zeros_like(map_array, dtype=bool)
    else:
        mask = _resize(mask.astype(np.uint8), map_array.shape) > 0
    pred = map_array >= HIGH_RESPONSE_THRESHOLD
    tp = int((pred & mask).sum())
    fp = int((pred & ~mask).sum())
    fn = int((~pred & mask).sum())
    union = tp + fp + fn
    out = {
        "score": float(score),
        "map_shape": list(map_array.shape),
        "map_min": float(map_array.min()),
        "map_max": float(map_array.max()),
        "map_mean": float(map_array.mean()),
        "map_high_fraction": float(pred.mean()),
        "map_finite": bool(np.isfinite(map_array).all()),
        "mask_area_fraction": float(mask.mean()),
        "predicted_area_fraction": float(pred.mean()),
        "iou": float(tp / union) if union else None,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "map_inside_mean": (float(map_array[mask].mean()) if mask.any() else None),
        "map_outside_mean": (float(map_array[~mask].mean()) if (~mask).any() else None),
        "label": label,
    }
    if label != "Fake":
        # Real images have no GT mask; pixel IoU/F1 are undefined, not zero.
        for key in ("iou", "f1", "precision", "recall", "map_inside_mean"):
            out[key] = None
    if conf_array is not None:
        conf = np.squeeze(np.asarray(conf_array, dtype=np.float64))
        if conf.shape != map_array.shape:
            conf = _resize(conf, map_array.shape)
        out.update({"conf_shape": list(conf.shape),
                    "conf_mean": float(conf.mean()),
                    "conf_low_fraction": float((conf < 0.5).mean()),
                    "conf_finite": bool(np.isfinite(conf).all())})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1（仅小批定位）")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault("TRUFOR_RESULTS",
                          os.path.join(PROJECT_ROOT, "sft_data", "g8_localization",
                                       "trufor_raw"))
    payload = json.load(open(INPUTS, encoding="utf-8"))
    rows = [r for r in payload["inputs"] if r.get("split") == "dev"]
    forged, reals = _select(rows)
    selected = forged + reals
    if len(forged) != 8 or len(reals) != 8:
        raise SystemExit(f"sample selection failed: forged={len(forged)} real={len(reals)}")

    expert = TruForExpert(device=args.device)
    reason = expert.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor 不可用：{reason}")

    images = {row["variant_id"]: _load_bgr(os.path.join(PROJECT_ROOT, row["path"]))
              for row in selected}
    started = time.perf_counter()
    try:
        results = expert.analyze_many(images, progress=True)
    except Exception as exc:
        report = {"schema": "trufor_autosplice_smoke/1",
                  "created_at": datetime.now().isoformat(timespec="seconds"),
                  "ok": False, "error": f"{type(exc).__name__}: {exc}",
                  "device": args.device, "environment": _environment(),
                  "weights": expert.provenance().as_dict(),
                  "sample_ids": [r["variant_id"] for r in selected]}
        out = args.out or os.path.join(OUT_DIR, "trufor_autosplice_smoke_20261102.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(report, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False))
        return 1
    wall = time.perf_counter() - started

    checks = []
    for row in selected:
        image = images[row["variant_id"]]
        key = expert.cache_key(image)
        arrays = dict(np.load(expert.result_path(key), allow_pickle=True))
        mask = _mask(os.path.join(PROJECT_ROOT, row["mask_path"])) if row.get("mask_path") else None
        metrics = _metrics(float(results[row["variant_id"]].raw_metric),
                           arrays["map"], arrays.get("conf"), mask, row["label"])
        metrics.update({
            "variant_id": row["variant_id"], "source_id": row["source_id"],
            "path": row["path"], "mask_path": row.get("mask_path"),
            "group": row["group"],
        })
        checks.append(metrics)

    forged_checks = [c for c in checks if c["label"] == "Fake"]
    real_checks = [c for c in checks if c["label"] == "Real"]
    report = {
        "schema": "trufor_autosplice_smoke/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ok": all(c["map_finite"] and c.get("conf_finite", True) for c in checks),
        "device": args.device,
        "weights": expert.provenance().as_dict(),
        "inputs_manifest": INPUTS,
        "seed": SEED,
        "sample": {
            "forged": len(forged_checks), "authentic": len(real_checks),
            "note": "dev partition; one mask per base shared by variants",
        },
        "timing": {"batch_wall_s": round(wall, 2),
                   "per_image_s": round(wall / len(checks), 3),
                   "adapter_timing": expert.timing_summary()},
        "maps": {"high_response_threshold": HIGH_RESPONSE_THRESHOLD,
                 "score_meaning": "TruFor detection head sigmoid; higher = more manipulated",
                 "map_meaning": "per-pixel P(forged) response; not a calibrated probability"},
        "checks": checks,
        "aggregate": {
            "forged_iou_mean": float(np.mean([c["iou"] for c in forged_checks
                                              if c["iou"] is not None])),
            "forged_f1_mean": float(np.mean([c["f1"] for c in forged_checks
                                             if c["f1"] is not None])),
            "forged_score_mean": float(np.mean([c["score"] for c in forged_checks])),
            "authentic_score_mean": float(np.mean([c["score"] for c in real_checks])),
            "authentic_high_fraction_mean": float(np.mean(
                [c["map_high_fraction"] for c in real_checks])),
            "authentic_false_positive_images": sum(
                1 for c in real_checks if c["map_high_fraction"] > 0.01),
        },
    }
    out = args.out or os.path.join(OUT_DIR, "trufor_autosplice_smoke_20261102.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps({k: report[k] for k in ("ok", "sample", "timing", "aggregate")},
                     ensure_ascii=False, indent=2))
    print(f"report: {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
