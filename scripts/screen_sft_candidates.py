"""Independent PROBE/TruFor screening for the SFT candidate shortlist.

No Qwen and no Agent loop: this only measures the shortlist so the final 24
can be chosen for discussion value.  Truth and masks are recorded for the
review sheet; they are never passed to the candidate writer.

Outputs:
  screening/per_sample.csv
  screening/summary.json
  screening/trufor_raw/<cache_key>/result.npz   (local cache, gitignored)
  screening/trufor_maps/<sample_id>__*.png      (review panels, gitignored)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from experts.probe_dino import ProbeDinoExpert  # noqa: E402
from experts.trufor import (  # noqa: E402
    HIGH_RESPONSE_THRESHOLD, TruForExpert,
)
from utils.image_utils import ImageUtils  # noqa: E402

DEFAULT_SHORTLIST = ROOT / "sft_data/sft_candidates_20261127/shortlist.json"
DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127/screening"
PROBE_THRESHOLD = 0.5
TRUFOR_SCORE_THRESHOLD = 0.5


def _load_mask(path: str):
    mask = cv2.imread(str(ROOT / path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise SystemExit(f"mask unreadable: {path}")
    return mask


def _map_metrics(arrays: dict, mask_path: str | None) -> dict:
    anomaly = np.asarray(arrays["map"], dtype=np.float64)
    confidence = arrays.get("conf")
    high = anomaly >= HIGH_RESPONSE_THRESHOLD
    peak_y, peak_x = np.unravel_index(int(np.argmax(anomaly)), anomaly.shape)
    height, width = anomaly.shape[:2]
    out = {
        "anomaly_peak": round(float(anomaly.max()), 6),
        "anomaly_peak_norm": [int(round(peak_y / max(1, height - 1) * 1000)),
                              int(round(peak_x / max(1, width - 1) * 1000))],
        "high_response_fraction": round(float(high.mean()), 6),
        "anomaly_mean": round(float(anomaly.mean()), 6),
    }
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float64)
        out["confidence_mean"] = round(float(confidence.mean()), 6)
        out["low_confidence_fraction"] = round(float((confidence < 0.5).mean()), 6)
    if mask_path:
        mask = _load_mask(mask_path)
        resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        truth = resized > 0
        pred = high
        inter = int(np.logical_and(truth, pred).sum())
        union = int(np.logical_or(truth, pred).sum())
        out["iou"] = round(inter / union, 6) if union else None
        out["precision"] = round(inter / int(pred.sum()), 6) if pred.sum() else None
        out["recall"] = round(inter / int(truth.sum()), 6) if truth.sum() else None
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shortlist", default=str(DEFAULT_SHORTLIST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")

    os.environ.setdefault(
        "PROBE_WEIGHTS",
        "/root/autodl-tmp/third_party/probe/DINOv2_best_model_step_34999.pth")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "trufor_raw"
    map_dir = out / "trufor_maps"
    raw_dir.mkdir(exist_ok=True)
    map_dir.mkdir(exist_ok=True)
    os.environ.setdefault("TRUFOR_RESULTS", str(raw_dir))

    payload = json.loads(Path(args.shortlist).read_text(encoding="utf-8"))
    samples = payload["samples"]
    if not samples:
        raise SystemExit("empty shortlist")
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    images, failures = {}, []
    for sample in samples:
        sid = sample["variant_id"]
        try:
            images[sid] = ImageUtils.load_image(str(ROOT / sample["path"]))
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sid, "stage": "load",
                             "error": f"{type(exc).__name__}: {exc}"})
    if failures:
        (out / "run_error.json").write_text(json.dumps(
            {"failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(f"image load failed: {failures[:3]}")

    probe = ProbeDinoExpert(device=args.device)
    reason = probe.unavailable_reason()
    if reason:
        raise SystemExit(f"PROBE unavailable: {reason}")
    probe_scores = {}
    for index, sample in enumerate(samples, 1):
        sid = sample["variant_id"]
        try:
            result = probe.analyze(images[sid])
            probe_scores[sid] = float(getattr(result, "raw_metric", result.strength))
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_id": sid, "stage": "probe",
                             "error": f"{type(exc).__name__}: {exc}"})
            break
        if index % 25 == 0:
            print(f"PROBE {index}/{len(samples)}", flush=True)
    del probe
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    if failures:
        (out / "run_error.json").write_text(json.dumps(
            {"failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(f"PROBE failed: {failures[:3]}")

    trufor = TruForExpert(device=args.device, cache=False,
                          results_dir=str(raw_dir))
    reason = trufor.unavailable_reason()
    if reason:
        raise SystemExit(f"TruFor unavailable: {reason}")
    trufor_scores, arrays_by_id = {}, {}
    for start in range(0, len(samples), args.batch_size):
        chunk_samples = samples[start:start + args.batch_size]
        chunk = {s["variant_id"]: images[s["variant_id"]] for s in chunk_samples}
        try:
            results = trufor.analyze_many(chunk, progress=True)
        except Exception as exc:  # noqa: BLE001
            failures.append({"sample_ids": list(chunk), "stage": "trufor",
                             "error": f"{type(exc).__name__}: {exc}"})
            break
        for sample in chunk_samples:
            sid = sample["variant_id"]
            result = results.get(sid)
            if result is None:
                failures.append({"sample_id": sid, "stage": "trufor",
                                 "error": "missing_result"})
                continue
            key = trufor.cache_key(images[sid])
            arrays = dict(np.load(trufor.result_path(key), allow_pickle=True))
            trufor_scores[sid] = float(result.raw_metric)
            arrays_by_id[sid] = arrays
        print(f"TruFor {min(start + args.batch_size, len(samples))}/{len(samples)}",
              flush=True)
    if failures:
        (out / "run_error.json").write_text(json.dumps(
            {"failures": failures}, ensure_ascii=False, indent=2))
        raise SystemExit(f"TruFor failed: {failures[:3]}")

    rows = []
    for sample in samples:
        sid = sample["variant_id"]
        arrays = arrays_by_id[sid]
        metrics = _map_metrics(arrays, sample.get("mask_path"))
        trufor._last = arrays
        artifacts = trufor.render_artifacts(images[sid])
        panel_paths = {}
        for name, image in artifacts.items():
            path = map_dir / f"{sid}__{name}.png"
            cv2.imwrite(str(path), image)
            panel_paths[name] = str(path.relative_to(ROOT))
        rows.append({
            "kind": sample["kind"],
            "sample_id": sid,
            "source_id": sample["source_id"],
            "group": sample["group"],
            "split": sample["split"],
            "path": sample["path"],
            "source_path": sample.get("source_path"),
            "mask_path": sample.get("mask_path"),
            "ground_truth": sample.get("ground_truth"),
            "whole_truth": sample["task_truth"]["whole_image_generation"],
            "local_truth": sample["task_truth"]["local_manipulation"],
            "probe_score": round(probe_scores[sid], 6),
            "probe_pred_ai": probe_scores[sid] >= PROBE_THRESHOLD,
            "trufor_score": round(trufor_scores[sid], 6),
            "trufor_pred_tampered": trufor_scores[sid] >= TRUFOR_SCORE_THRESHOLD,
            **metrics,
            "panels": json.dumps(panel_paths, ensure_ascii=False),
        })

    csv_path = out / "per_sample.csv"
    fields = list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "sft_candidate_screening/1",
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "shortlist": str(Path(args.shortlist)),
        "samples": len(rows),
        "probe_threshold": PROBE_THRESHOLD,
        "trufor_score_threshold": TRUFOR_SCORE_THRESHOLD,
        "trufor_map_threshold": HIGH_RESPONSE_THRESHOLD,
        "by_kind": {},
        "probe_failures": 0,
        "trufor_failures": 0,
        "per_sample_csv": str(csv_path.relative_to(ROOT)),
        "note": "truth and mask metrics are review-only; not shown to the candidate writer",
    }
    for kind in sorted({row["kind"] for row in rows}):
        subset = [row for row in rows if row["kind"] == kind]
        summary["by_kind"][kind] = {
            "n": len(subset),
            "probe_pred_ai": sum(1 for row in subset if row["probe_pred_ai"]),
            "trufor_pred_tampered": sum(1 for row in subset
                                        if row["trufor_pred_tampered"]),
            "probe_score_mean": round(float(np.mean([row["probe_score"]
                                                     for row in subset])), 6),
            "trufor_score_mean": round(float(np.mean([row["trufor_score"]
                                                      for row in subset])), 6),
        }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
