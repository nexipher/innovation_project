"""IML-ViT localization evaluation on the fixed AutoSplice q95 inputs.

Uses the official checkout at the pinned commit, the uploaded CAT-Net/TruFor-
protocol checkpoint, and the official ``pad`` transform (zero-padding to
1024x1024, ImageNet normalization, top-left crop).  The model has **no image-
level score**; this script only reports map-based localization metrics and
never invents a score.

Example:
  NO_ALBUMENTATIONS_UPDATE=1 FORENSIC_ALLOW_GPU=1 \
  python scripts/iml_vit_autosplice_eval.py \
    --inputs sft_data/g8_localization/autosplice_dev_subset_20261102.json \
    --out sft_data/g8_localization/iml_vit_autosplice_dev_20261102
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import resource
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

REPO = Path("/root/autodl-tmp/third_party/IML-ViT/"
            "IML-ViT-07dd2be0f4ea27a5c97c9fa5ffbe236733833eac")
CHECKPOINT = Path("/root/autodl-tmp/third_party/IML-ViT/weights/"
                  "iml-vit_checkpoint_trufor_20231104.pth")
DEFAULT_INPUTS = ROOT / "sft_data/g8_localization/autosplice_dev_subset_20261102.json"
DEFAULT_OUT = ROOT / "sft_data/g8_localization/iml_vit_autosplice_dev_20261102"
HIGH_THRESHOLD = 0.5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dist(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values), "mean": round(float(a.mean()), 6),
        "std": round(float(a.std(ddof=0)), 6), "min": round(float(a.min()), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
        "max": round(float(a.max()), 6),
    }


def _map_metrics(prob, mask, label):
    prob = np.asarray(prob, dtype=np.float64)
    if prob.ndim != 2:
        raise ValueError(f"map is not 2-D: {prob.shape}")
    pred = prob >= HIGH_THRESHOLD
    mask = np.asarray(mask, dtype=bool) if mask is not None else None
    out = {
        "map_shape": list(prob.shape),
        "map_min": float(prob.min()), "map_max": float(prob.max()),
        "map_mean": float(prob.mean()), "map_high_fraction": float(pred.mean()),
        "map_finite": bool(np.isfinite(prob).all()),
        "score": None, "score_available": False,
        "map_inside_mean": None, "map_outside_mean": float(prob.mean()),
        "iou": None, "f1": None, "precision": None, "recall": None,
        "mask_area_fraction": 0.0, "predicted_area_fraction": float(pred.mean()),
    }
    if label == "Fake":
        if mask is None:
            raise ValueError("forged row without mask")
        if mask.shape != prob.shape:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (prob.shape[1], prob.shape[0]),
                             interpolation=cv2.INTER_NEAREST) > 0
        tp = int((pred & mask).sum())
        fp = int((pred & ~mask).sum())
        fn = int((~pred & mask).sum())
        union = tp + fp + fn
        out.update({
            "mask_area_fraction": float(mask.mean()),
            "map_inside_mean": (float(prob[mask].mean()) if mask.any() else None),
            "map_outside_mean": (float(prob[~mask].mean()) if (~mask).any() else None),
            "iou": float(tp / union) if union else None,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
            "precision": float(tp / (tp + fp)) if (tp + fp) else None,
            "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        })
    return out


def _source_aggregate(checks):
    by_source = {}
    for check in checks:
        by_source.setdefault(check["source_id"], []).append(check)
    sources = []
    for sid, rows in sorted(by_source.items()):
        real = [row for row in rows if row["label"] == "Real"]
        fake = [row for row in rows if row["label"] == "Fake"]
        if len(real) > 1:
            raise ValueError(f"{sid}: expected at most one authentic row, got {len(real)}")
        authentic = None
        if real:
            authentic = {
                "variant_id": real[0]["variant_id"],
                "map_high_fraction": real[0]["map_high_fraction"],
                "map_mean": real[0]["map_mean"],
                "score": None,
            }
        source = {
            "source_id": sid, "variant_count": len(fake),
            "authentic": authentic,
            "forged": [{
                "variant_id": row["variant_id"], "iou": row["iou"],
                "f1": row["f1"], "precision": row["precision"],
                "recall": row["recall"], "map_high_fraction": row["map_high_fraction"],
                "map_inside_mean": row["map_inside_mean"],
                "map_outside_mean": row["map_outside_mean"],
            } for row in sorted(fake, key=lambda row: row["variant_id"])],
        }
        if fake:
            source["forged_summary"] = {
                "iou_mean": _mean(row["iou"] for row in fake),
                "iou_max": max(row["iou"] for row in fake) if all(
                    row["iou"] is not None for row in fake) else None,
                "f1_mean": _mean(row["f1"] for row in fake),
                "precision_mean": _mean(row["precision"] for row in fake),
                "recall_mean": _mean(row["recall"] for row in fake),
            }
        sources.append(source)
    forged_sources = [source for source in sources if source["variant_count"]]
    real_sources = [source for source in sources if source["authentic"] is not None]
    return {
        "sources": sources,
        "n_sources": len(sources),
        "n_forged_sources": len(forged_sources),
        "n_real_sources": len(real_sources),
        "forged_variant_iou": _dist([row["iou"] for row in checks
                                     if row["label"] == "Fake"]),
        "forged_variant_f1": _dist([row["f1"] for row in checks
                                    if row["label"] == "Fake"]),
        "forged_variant_recall": _dist([row["recall"] for row in checks
                                        if row["label"] == "Fake"]),
        "forged_variant_precision": _dist([row["precision"] for row in checks
                                           if row["label"] == "Fake"]),
        "forged_source_iou_mean": _dist(
            [source["forged_summary"]["iou_mean"] for source in forged_sources]),
        "forged_source_f1_mean": _dist(
            [source["forged_summary"]["f1_mean"] for source in forged_sources]),
        "real_high_fraction": _dist(
            [source["authentic"]["map_high_fraction"] for source in real_sources]),
        "real_false_positive_sources": [
            source for source in real_sources
            if source["authentic"]["map_high_fraction"] > 0.01
        ],
        "score_available": False,
        "note": ("IML-ViT has no image-level score/conf. Same-source variants "
                 "are repeated measurements and are aggregated by base."),
    }


def _mean(values):
    values = [value for value in values if value is not None]
    return round(statistics.mean(values), 6) if values else None


def run(inputs_path: Path, out_dir: Path, device: str, batch_size: int):
    inputs_path = inputs_path.resolve()
    out_dir = out_dir.resolve()
    payload = json.load(open(inputs_path, encoding="utf-8"))
    rows = payload["inputs"]
    if device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise SystemExit("CUDA 不可用")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    sys.path.insert(0, str(REPO))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    from iml_vit_model import iml_vit_model
    # The project also has a top-level `utils` package; import the official
    # transform by file path so it cannot be shadowed by the project package.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "iml_transforms_official", REPO / "utils/iml_transforms.py")
    transform_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform_module)
    get_albu_transforms = transform_module.get_albu_transforms

    torch_device = torch.device(device)
    started = time.perf_counter()
    load_started = time.perf_counter()
    model = iml_vit_model()
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    model_state = model.state_dict()
    missing = [key for key in model_state if key not in state]
    unexpected = [key for key in state if key not in model_state]
    mismatch = [key for key in model_state if key in state
                and tuple(model_state[key].shape) != tuple(state[key].shape)]
    if missing or unexpected or mismatch:
        raise SystemExit(f"checkpoint coverage failed: missing={missing[:3]} "
                         f"unexpected={unexpected[:3]} mismatch={mismatch[:3]}")
    model.load_state_dict(state, strict=True)
    model.to(torch_device).eval()
    load_s = round(time.perf_counter() - load_started, 3)
    transform = get_albu_transforms(type_="pad", outputsize=1024)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    checks = []
    failures = []
    keys = [row["variant_id"] for row in rows]
    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        tensors = []
        valid = []
        for row in chunk:
            try:
                image = Image.open(ROOT / row["path"]).convert("RGB")
                array = np.asarray(image)
                tensor = transform(image=array)["image"]
                tensors.append(tensor)
                valid.append((row, image.size))
            except Exception as exc:
                failures.append({"variant_id": row["variant_id"],
                                 "error": f"preprocess: {type(exc).__name__}: {exc}"})
        if not tensors:
            continue
        batch = torch.stack(tensors).to(torch_device)
        dummy = torch.zeros((len(tensors), 1, 1024, 1024), device=torch_device,
                            dtype=batch.dtype)
        try:
            with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
                _, mask_pred, _ = model(batch, dummy, dummy)
            probabilities = mask_pred[:, 0].detach().cpu().numpy()
        except Exception as exc:
            failures.extend({"variant_id": row["variant_id"],
                             "error": f"forward: {type(exc).__name__}: {exc}"}
                            for row, _ in valid)
            continue
        for (row, size), prob in zip(valid, probabilities):
            width, height = size
            prob = np.asarray(prob[:height, :width], dtype=np.float32)
            if not np.isfinite(prob).all():
                failures.append({"variant_id": row["variant_id"],
                                 "error": "non-finite map"})
                continue
            mask = None
            if row["label"] == "Fake":
                mask = np.asarray(Image.open(ROOT / row["mask_path"]).convert("L")) > 0
            metrics = _map_metrics(prob, mask, row["label"])
            metrics.update({
                "variant_id": row["variant_id"], "source_id": row["source_id"],
                "label": row["label"], "variant": row.get("variant"),
                "path": row["path"], "mask_path": row.get("mask_path"),
                "source_sha256": row.get("source_sha256"),
                "q95_sha256": row.get("sha256"),
                "mask_sha256": row.get("mask_sha256"),
            })
            np.savez_compressed(raw_dir / f"{row['variant_id']}.npz",
                                map_uint8=np.round(prob * 255).astype(np.uint8))
            checks.append(metrics)
    wall = round(time.perf_counter() - started, 2)
    gpu_peak = gpu_reserved = None
    if device.startswith("cuda"):
        gpu_peak = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        gpu_reserved = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
    report = {
        "schema": "iml_vit_autosplice_eval/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "ok": len(failures) == 0 and len(checks) == len(rows),
        "device": device,
        "inputs": str(inputs_path.relative_to(ROOT)),
        "inputs_sha256": sha256(inputs_path),
        "code": {
            "repo": str(REPO), "commit": "07dd2be0f4ea27a5c97c9fa5ffbe236733833eac",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256(CHECKPOINT),
            "checkpoint_bytes": CHECKPOINT.stat().st_size,
            "checkpoint_tensors": len(state),
            "model_tensors": len(model_state),
            "strict_load": True,
        },
        "preprocess": {
            "padding": "PadIfNeeded(min_height=1024,min_width=1024,top_left,0)",
            "normalize": "ImageNet mean/std",
            "crop": "top-left 1024x1024; output cropped to original h,w",
            "mask_for_forward": "dummy zeros; model returns sigmoid map only",
        },
        "timing": {"wall_s": wall, "load_s": load_s,
                   "per_input_s": round(wall / max(1, len(checks)), 3)},
        "resources": {"gpu_peak_allocated_gb": gpu_peak,
                      "gpu_peak_reserved_gb": gpu_reserved,
                      "host_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        "failures": failures,
        "checks": checks,
        "aggregate": _source_aggregate(checks) if checks else None,
    }
    json.dump(report, open(out_dir / "report.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    a = report["aggregate"]
    lines = [
        "# IML-ViT AutoSplice q95 定位评测",
        "",
        f"checkpoint `{CHECKPOINT.name}`（sha256 `{report['code']['checkpoint_sha256'][:16]}…`）；"
        f"device `{device}`；输入 `{report['inputs']}`。",
        "",
        "> IML-ViT **没有图像级 score/conf**；本报告只给 map 定位指标，不补造分数。"
        "mask 为 base 级共享，同源变体按 base 聚合。",
        "",
    ]
    if a:
        lines += [
            "| 项目 | 值 |",
            "| --- | --- |",
            f"| forged variant IoU | mean {a['forged_variant_iou']['mean']:.3f}，"
            f"median {a['forged_variant_iou']['p50']:.3f} |",
            f"| forged variant F1 | mean {a['forged_variant_f1']['mean']:.3f}，"
            f"median {a['forged_variant_f1']['p50']:.3f} |",
            f"| forged source IoU | mean {a['forged_source_iou_mean']['mean']:.3f} |",
            f"| real high-response | mean {a['real_high_fraction']['mean']:.3f}，"
            f">1% sources {len(a['real_false_positive_sources'])}/{a['n_real_sources']} |",
            f"| score | 无（模型不提供） |",
            "",
        ]
    lines += ["完整逐图指标、源级聚合、失败和资源字段见同目录 `report.json`。"]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "checked": len(checks),
                      "failures": len(failures),
                      "iou_mean": a["forged_variant_iou"]["mean"] if a else None,
                      "f1_mean": a["forged_variant_f1"]["mean"] if a else None,
                      "wall_s": wall}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    run(Path(args.inputs), Path(args.out), args.device, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
