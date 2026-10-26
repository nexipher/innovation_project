"""
GPU smoke for one candidate expert, one process at a time (plan §4.17e).

The smoke does **not** judge wiring by "fake must score higher than real".
It verifies the interface numerically on the same image:

  * official preprocessing tensor  ==  adapter preprocessing tensor
  * official forward score         ==  adapter score

and records first-call vs steady-state time, GPU peak, host RSS and cgroup
memory.  SPAI and AIDE must be run as two separate processes:

  FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
      scripts/smoke_candidate.py --expert spai
  FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
      scripts/smoke_candidate.py --expert aide

No Agent prompt, stopping rule or b2 record is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from config import PROJECT_ROOT  # noqa: E402

MANIFEST = os.path.join(PROJECT_ROOT, "dataset", "mnw_div2k", "inputs_manifest.json")


def _pick_images() -> list:
    manifest = json.load(open(MANIFEST, encoding="utf-8"))
    entries = [e for e in manifest["inputs"] if e["condition"] == "original"]
    fake = next(e for e in entries if e["label"] == "Fake")
    real = next(e for e in entries if e["label"] == "Real")
    return [fake, real]


def _load_pil(entry: dict) -> Image.Image:
    return Image.open(os.path.join(PROJECT_ROOT, entry["path"])).convert("RGB")


def _cgroup(name: str):
    path = f"/sys/fs/cgroup/memory.{name}"
    try:
        with open(path, encoding="utf-8") as handle:
            return int(handle.read().strip())
    except OSError:
        return None


def _official_tensor_spai(evaluator, pil):
    """Official SPAI test transform, invoked outside the adapter's method."""
    from experts.spai import _fill_defaults, _install_import_shims

    _install_import_shims()
    from spai.config import _C as DEFAULT
    from spai.data.data_finetune import build_transform

    checkpoint = torch.load(evaluator.weights_path, map_location="cpu",
                            mmap=True, weights_only=False)
    config = checkpoint["config"].clone()
    config.defrost()
    _fill_defaults(config, DEFAULT)
    transform = build_transform(False, config)
    return transform(image=np.asarray(pil.convert("RGB")))["image"]


def _official_tensor_aide(pil):
    """Official TestDataset pipeline, written here rather than calling the adapter."""
    from torchvision import transforms
    from data.dct import DCT_base_Rec_Module

    to_tensor = transforms.ToTensor()
    dct = DCT_base_Rec_Module()
    post = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    tensor = to_tensor(pil.convert("RGB"))
    parts = dct(tensor)
    return torch.stack([post(part) for part in parts] + [post(tensor)], dim=0)


def _official_score(expert: str, model, tensor) -> float:
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(next(model.parameters()).device))
    if expert == "spai":
        return float(torch.sigmoid(logits).reshape(-1)[0].item())
    return float(torch.softmax(logits, dim=1)[:, 1].item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expert", choices=("spai", "aide"), required=True)
    parser.add_argument("--checkpoint", default="progan_train",
                        help="AIDE checkpoint (only progan_train is the smoke default)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if not os.environ.get("FORENSIC_ALLOW_GPU"):
        raise SystemExit("GPU 未授权：先设置 FORENSIC_ALLOW_GPU=1（仅冒烟）")

    report = {
        "schema": "candidate_gpu_smoke/1",
        "expert": args.expert,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "device": args.device,
        "cgroup_memory_start": _cgroup("current"),
        "host_maxrss_start_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    try:
        import experts.spai as spai_module
        import experts.aide as aide_module

        if args.expert == "spai":
            evaluator = spai_module.SPAIEvaluator(device=args.device)
            official_tensor = _official_tensor_spai
        else:
            evaluator = aide_module.AIDEEvaluator(checkpoint=args.checkpoint,
                                                  device=args.device)
            official_tensor = _official_tensor_aide

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        evaluator.load()
        report["load_s"] = round(time.perf_counter() - t0, 3)
        report["gpu_alloc_after_load_gb"] = round(
            torch.cuda.memory_allocated() / 1024 ** 3, 3)
        report["gpu_reserved_after_load_gb"] = round(
            torch.cuda.memory_reserved() / 1024 ** 3, 3)
        report["gpu_peak_after_load_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 3)

        images = _pick_images()
        report["images"] = [{"path": e["path"], "sha256": e["sha256"],
                             "label": e["label"]} for e in images]
        report["checks"] = []
        adapter_scores = []
        first = None
        for index, entry in enumerate(images):
            pil = _load_pil(entry)
            if first is None:
                t = time.perf_counter()
                adapter_score = evaluator.score(pil)
                first = time.perf_counter() - t
            else:
                t = time.perf_counter()
                adapter_score = evaluator.score(pil)
                adapter_scores.append(time.perf_counter() - t)

            tensor = official_tensor(evaluator, pil) if args.expert == "spai" \
                else official_tensor(pil)
            adapter_tensor = evaluator.preprocess(pil)
            tensor_equal = bool(torch.equal(tensor, adapter_tensor))
            tensor_max_diff = float((tensor - adapter_tensor).abs().max().item())
            official = _official_score(args.expert, evaluator._model, tensor)
            report["checks"].append({
                "image": entry["path"], "label": entry["label"],
                "tensor_equal": tensor_equal,
                "tensor_max_abs_diff": tensor_max_diff,
                "official_score": official,
                "adapter_score": adapter_score,
                "score_abs_diff": abs(official - adapter_score),
            })

        # steady-state calls (same two images, alternating)
        steady = []
        for index in range(3):
            pil = _load_pil(images[index % len(images)])
            t = time.perf_counter()
            evaluator.score(pil)
            steady.append(time.perf_counter() - t)
        report["first_call_s"] = round(first, 4)
        report["steady_calls_s"] = [round(x, 4) for x in steady]
        report["steady_median_s"] = round(statistics.median(steady), 4)

        report["gpu_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
        report["gpu_reserved_peak_gb"] = round(
            torch.cuda.max_memory_reserved() / 1024 ** 3, 3)
        report["host_maxrss_kb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["cgroup_memory_end"] = _cgroup("current")
        report["cgroup_memory_peak"] = _cgroup("peak")
        report["ok"] = all(
            check["tensor_equal"] and check["score_abs_diff"] <= 1e-6
            for check in report["checks"])
    except Exception as exc:
        import traceback

        report["ok"] = False
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-4000:]
    report["finished_at"] = datetime.now().isoformat(timespec="seconds")

    out = args.out or os.path.join(
        PROJECT_ROOT, "sft_data", "g8_expert_eval",
        f"smoke_{args.expert}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: report.get(k) for k in
                      ("expert", "ok", "load_s", "first_call_s", "steady_median_s",
                       "gpu_peak_gb", "gpu_reserved_peak_gb", "host_maxrss_kb",
                       "cgroup_memory_peak", "error")}, ensure_ascii=False))
    for check in report.get("checks", []):
        print(json.dumps(check, ensure_ascii=False))
    print(f"report: {out}")
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
