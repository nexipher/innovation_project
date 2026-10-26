"""
Official-path verification for one candidate, one process at a time.

Unlike `smoke_candidate.py` (which compares two paths on the same loaded
model), this script calls the **official** data pipeline directly:

  * SPAI: `spai.data.data_finetune.CSVDataset` (official reader + transform)
    and, separately, the official `python -m spai infer` CLI as an
    end-to-end reference.  The adapter is compared on the same images.
  * AIDE: `data.datasets.TestDataset` (official TestDataset: PIL RGB ->
    ToTensor -> DCT -> Resize/Normalize) and the official output formula
    `softmax(logits, dim=1)[:, 1]`; `--checkpoint` selects progan/sd14/GenImage.
  * DRCT: official `data.transform.create_val_transforms(size=224, is_crop=True)`
    and the official `network.models.get_models` factory; output formula
    `1 - softmax(logits, dim=1)[:, 0]`.

For each image it records:
  preprocessed tensor equality / max diff, official vs adapter logits,
  official vs adapter final score, repeated `Expert.analyze()` calls in the
  same process, finite checks, model coverage, and the bound identity
  (weights sha256, code commit, preprocessing, output semantics).

Limitations are recorded explicitly in the JSON: the AIDE model construction
and streaming load reuse the adapter (the checkpoint contains the trunk);
the official `engine_finetune.evaluate` distributed loop is not called, but
the official TestDataset and the official output formula are.  SPAI's
official CLI runs in a separate process; its model is not shared with the
adapter.

  FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
      scripts/verify_candidate_official.py --expert spai
  FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
      scripts/verify_candidate_official.py --expert aide --checkpoint sd14_train
  FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
      scripts/verify_candidate_official.py --expert drct
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from config import PROJECT_ROOT  # noqa: E402
from experts.candidate_common import load_candidate_code, load_candidate_weights  # noqa: E402

SPAI_REPO = "/root/autodl-tmp/third_party/spai/repo"
SPAI_CFG = os.path.join(SPAI_REPO, "configs", "spai.yaml")
SPAI_WEIGHTS = "/root/autodl-tmp/third_party/spai/weights/spai.pth"
AIDE_REPO = "/root/autodl-tmp/third_party/aide/repo"
AIDE_WEIGHTS = "/root/autodl-tmp/third_party/aide/weights/progan_train.pth"
DRCT_REPO = "/root/autodl-tmp/third_party/drct/repo"
DRCT_WEIGHTS = (
    "/root/autodl-tmp/third_party/drct/weights/DRCT-2M/sdv14/"
    "convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"
)

IMAGES = [
    ("fake", os.path.join(PROJECT_ROOT, "dataset/mnw_div2k/Fake/Adobe_Firefly_v4/"
                          "Adobe_Firefly_4_09a4b560f85f4ee482b201522.jpg"), 1),
    ("real", os.path.join(PROJECT_ROOT, "dataset/mnw_div2k/Real/DIV2K_valid_HR/"
                          "0841.png"), 0),
]

WRAPPER = '''\
import sys, types
sys.path.insert(0, {repo!r})
import timm.data.transforms as tt
if not hasattr(tt, "_pil_interp"):
    from PIL import Image
    table = {{"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
              "nearest": Image.NEAREST}}
    tt._pil_interp = lambda method: table[method]
sys.modules.setdefault("clip", types.ModuleType("clip"))
sys.modules.setdefault("neptune", types.ModuleType("neptune"))
from spai.__main__ import cli
cli()
'''


def _cgroup(name: str):
    try:
        with open(f"/sys/fs/cgroup/memory.{name}", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except OSError:
        return None


def _prepare_data(tmp: str):
    data_dir = os.path.join(tmp, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name, path, _label in IMAGES:
        shutil.copyfile(path, os.path.join(data_dir, os.path.basename(path)))
    csv_path = os.path.join(tmp, "two.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "class", "split"])
        for name, path, label in IMAGES:
            writer.writerow([os.path.basename(path), label, "test"])
    return data_dir, csv_path


def _prepare_aide_data(tmp: str):
    root = os.path.join(tmp, "aide_data")
    os.makedirs(os.path.join(root, "0_real"), exist_ok=True)
    os.makedirs(os.path.join(root, "1_fake"), exist_ok=True)
    for name, path, label in IMAGES:
        folder = "1_fake" if label == 1 else "0_real"
        shutil.copyfile(path, os.path.join(root, folder, os.path.basename(path)))
    return root


def _bgr(pil: Image.Image) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(pil.convert("RGB"))[:, :, ::-1])


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _coverage(evaluator) -> dict:
    return evaluator.check_coverage()


def _spai_official_cli(tmp: str, csv_path: str, data_dir: str) -> dict:
    wrapper = os.path.join(tmp, "spai_cli_wrapper.py")
    with open(wrapper, "w", encoding="utf-8") as handle:
        handle.write(WRAPPER.format(repo=SPAI_REPO))
    out_dir = os.path.join(tmp, "spai_out")
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, FORENSIC_ALLOW_GPU="1",
               NO_ALBUMENTATIONS_UPDATE="1")
    command = [sys.executable, wrapper, "infer",
               "--cfg", SPAI_CFG, "--input", csv_path,
               "--input-csv-root-dir", data_dir,
               "--model", SPAI_WEIGHTS, "--output", out_dir,
               "--batch-size", "1", "--tag", "official"]
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=SPAI_REPO, env=env,
                               capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - started
    scores = {}
    output_csv = os.path.join(out_dir, "two.csv")
    if completed.returncode == 0 and os.path.exists(output_csv):
        with open(output_csv, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                scores[row["image"]] = float(row["official"])
    return {"returncode": completed.returncode, "elapsed_s": round(elapsed, 3),
            "scores": scores, "stderr_tail": completed.stderr[-2000:]}


def _spai_official_tensors(csv_path: str, data_dir: str) -> dict:
    from experts.spai import _install_import_shims

    sys.path.insert(0, SPAI_REPO)
    _install_import_shims()
    from spai.config import get_config
    from spai.data.data_finetune import CSVDataset, build_transform

    config = get_config({"cfg": SPAI_CFG, "opts": ()})
    dataset = CSVDataset(csv_path, data_dir, split="test",
                         transform=build_transform(False, config))
    out = {}
    for index in range(len(dataset)):
        tensor, label, _idx = dataset[index]
        name = os.path.basename(dataset.entries[index]["image"])
        out[name] = {"tensor": tensor.squeeze(0), "label": int(label),
                     "shape": list(tensor.shape)}
    return out


def _aide_official_tensors(root: str) -> dict:
    sys.path.insert(0, AIDE_REPO)
    from data.datasets import TestDataset

    dataset = TestDataset(is_train=False,
                          args=SimpleNamespace(eval_data_path=root))
    out = {}
    for index in range(len(dataset)):
        tensor, label = dataset[index]
        name = os.path.basename(dataset.data_list[index]["image_path"])
        out[name] = {"tensor": tensor, "label": int(label),
                     "shape": list(tensor.shape)}
    return out


def _run_spai(tmp: str) -> dict:
    from experts.spai import SPAIExpert

    data_dir, csv_path = _prepare_data(tmp)
    report = {"official_cli": _spai_official_cli(tmp, csv_path, data_dir)}
    official = _spai_official_tensors(csv_path, data_dir)

    torch.cuda.reset_peak_memory_stats()
    expert = SPAIExpert(device="cuda")
    checks = []
    adapter_scores = {}
    for name, path, label in IMAGES:
        pil = Image.open(path).convert("RGB")
        bgr = _bgr(pil)
        t0 = time.perf_counter()
        first = expert.analyze(bgr)
        first_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        second = expert.analyze(bgr)
        second_s = time.perf_counter() - t0
        evaluator = expert._evaluator
        adapter_tensor = evaluator.preprocess(pil)
        adapter_logits = evaluator.raw_logits(pil)
        official_tensor = official[os.path.basename(path)]["tensor"]
        with torch.no_grad():
            official_logits = evaluator._model(
                [official_tensor.unsqueeze(0).to(evaluator.device)],
                evaluator._cfg.MODEL.FEATURE_EXTRACTION_BATCH)
        adapter_score = float(torch.sigmoid(adapter_logits).reshape(-1)[0].item())
        adapter_scores[os.path.basename(path)] = adapter_score
        checks.append({
            "image": os.path.basename(path), "label": label,
            "tensor_equal": bool(torch.equal(adapter_tensor, official_tensor)),
            "tensor_max_abs_diff": float((adapter_tensor - official_tensor).abs().max()),
            "adapter_logit": float(adapter_logits.reshape(-1)[0]),
            "official_logit": float(official_logits.reshape(-1)[0]),
            "logit_abs_diff": float((adapter_logits.reshape(-1)[0]
                                     - official_logits.reshape(-1)[0]).abs()),
            "adapter_score": adapter_score,
            "official_cli_score": report["official_cli"]["scores"].get(
                os.path.basename(path)),
            "cli_score_abs_diff": abs(
                adapter_score - report["official_cli"]["scores"].get(
                    os.path.basename(path), float("nan"))),
            "expert_analyze_first": float(first.raw_metric),
            "expert_analyze_second": float(second.raw_metric),
            "analyze_equal": float(first.raw_metric) == float(second.raw_metric),
            "finite": _finite(first.raw_metric) and _finite(second.raw_metric),
            "first_analyze_s": round(first_s, 4),
            "second_analyze_s": round(second_s, 4),
        })
    report.update({
        "checks": checks,
        "coverage": _coverage(expert._evaluator),
        "identity": expert._evaluator.identity(),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "gpu_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "host_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cgroup_current": _cgroup("current"),
        "cgroup_peak": _cgroup("peak"),
        "reused": ["model construction and streaming checkpoint load (adapter; "
                   "checkpoint/config are the pinned official ones)",
                   "adapter preprocessing call for the tensor comparison"],
        "independent": ["official python -m spai infer CLI (separate process, "
                        "official CSVDataset + model + output CSV)",
                        "official CSVDataset preprocessed tensors",
                        "official model logits on the official tensor"],
    })
    return report


def _run_aide(tmp: str, checkpoint: str = "progan_train") -> dict:
    from experts.aide import AIDEExpert

    root = _prepare_aide_data(tmp)
    official = _aide_official_tensors(root)
    torch.cuda.reset_peak_memory_stats()
    expert = AIDEExpert(checkpoint=checkpoint, device="cuda")
    checks = []
    for name, path, label in IMAGES:
        pil = Image.open(path).convert("RGB")
        bgr = _bgr(pil)
        first = expert.analyze(bgr)
        second = expert.analyze(bgr)
        evaluator = expert._evaluator
        adapter_tensor = evaluator.preprocess(pil)
        adapter_logits = evaluator.raw_logits(pil)
        official_tensor = official[os.path.basename(path)]["tensor"]
        with torch.no_grad():
            official_logits = evaluator._model(
                official_tensor.unsqueeze(0).to(evaluator.device))
        official_score = float(torch.softmax(official_logits, dim=1)[:, 1].item())
        adapter_score = float(torch.softmax(adapter_logits, dim=1)[:, 1].item())
        checks.append({
            "image": os.path.basename(path), "label": label,
            "tensor_equal": bool(torch.equal(adapter_tensor, official_tensor)),
            "tensor_max_abs_diff": float((adapter_tensor - official_tensor).abs().max()),
            "adapter_logits": [float(x) for x in adapter_logits.reshape(-1)],
            "official_logits": [float(x) for x in official_logits.reshape(-1)],
            "logit_abs_diff_max": float(
                (adapter_logits.reshape(-1) - official_logits.reshape(-1)).abs().max()),
            "adapter_score": adapter_score,
            "official_score": official_score,
            "score_abs_diff": abs(adapter_score - official_score),
            "expert_analyze_first": float(first.raw_metric),
            "expert_analyze_second": float(second.raw_metric),
            "analyze_equal": float(first.raw_metric) == float(second.raw_metric),
            "finite": _finite(first.raw_metric) and _finite(second.raw_metric),
        })
    return {
        "checkpoint": checkpoint,
        "checks": checks,
        "coverage": _coverage(expert._evaluator),
        "identity": expert._evaluator.identity(),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "gpu_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "host_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cgroup_current": _cgroup("current"),
        "cgroup_peak": _cgroup("peak"),
        "reused": ["official AIDE_Model construction and streaming checkpoint "
                   "load (adapter; checkpoint contains the ConvNeXt-XXL trunk)",
                   "adapter preprocessing call for the tensor comparison"],
        "independent": ["official data.datasets.TestDataset (PIL RGB -> ToTensor "
                        "-> DCT -> Resize/Normalize)",
                        "official output formula softmax(logits, dim=1)[:, 1] "
                        "on the official tensor",
                        "repeated Expert.analyze() calls in the same process"],
        "limitation": ("engine_finetune.evaluate's distributed all_gather loop "
                       "was not called; the official TestDataset and output "
                       "formula were used directly."),
    }


def _run_drct(tmp: str) -> dict:
    from experts.drct import DRCTExpert, _install_import_shims
    from data.transform import create_val_transforms

    _install_import_shims()
    official_transform = create_val_transforms(size=224, is_crop=True)
    torch.cuda.reset_peak_memory_stats()
    expert = DRCTExpert(device="cuda")
    checks = []
    for name, path, label in IMAGES:
        pil = Image.open(path).convert("RGB")
        bgr = _bgr(pil)
        first = expert.analyze(bgr)
        second = expert.analyze(bgr)
        evaluator = expert._evaluator
        adapter_tensor = evaluator.preprocess(pil)
        official_tensor = official_transform(
            image=np.asarray(pil.convert("RGB")))["image"]
        adapter_logits = evaluator.raw_logits(pil)
        with torch.no_grad():
            official_logits = evaluator._model(
                official_tensor.unsqueeze(0).to(evaluator.device))
        official_score = float(1.0 - torch.softmax(official_logits, dim=1)[:, 0].item())
        adapter_score = float(1.0 - torch.softmax(adapter_logits, dim=1)[:, 0].item())
        checks.append({
            "image": os.path.basename(path), "label": label,
            "tensor_equal": bool(torch.equal(adapter_tensor, official_tensor)),
            "tensor_max_abs_diff": float((adapter_tensor - official_tensor).abs().max()),
            "adapter_logits": [float(x) for x in adapter_logits.reshape(-1)],
            "official_logits": [float(x) for x in official_logits.reshape(-1)],
            "logit_abs_diff_max": float(
                (adapter_logits.reshape(-1) - official_logits.reshape(-1)).abs().max()),
            "adapter_score": adapter_score,
            "official_score": official_score,
            "score_abs_diff": abs(adapter_score - official_score),
            "expert_analyze_first": float(first.raw_metric),
            "expert_analyze_second": float(second.raw_metric),
            "analyze_equal": float(first.raw_metric) == float(second.raw_metric),
            "finite": _finite(first.raw_metric) and _finite(second.raw_metric),
        })
    return {
        "checks": checks,
        "coverage": _coverage(expert._evaluator),
        "identity": expert._evaluator.identity(),
        "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3),
        "gpu_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "host_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cgroup_current": _cgroup("current"),
        "cgroup_peak": _cgroup("peak"),
        "reused": ["official network.models.get_models + streaming checkpoint "
                   "load (adapter; checkpoint is the pinned official one)",
                   "adapter preprocessing call for the tensor comparison"],
        "independent": ["official data.transform.create_val_transforms("
                        "size=224, is_crop=True) tensors",
                        "official output formula 1 - softmax(logits, dim=1)[:, 0]",
                        "repeated DRCTExpert.analyze() calls in one process"],
        "limitation": ("official scripts/train.py --is_test DataLoader loop "
                       "was not invoked; the official transform, official model "
                       "factory and official output formula were used directly."),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expert", choices=("spai", "aide", "drct"), required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="AIDE checkpoint: progan_train (default), sd14_train, GenImage_train")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if not os.environ.get("FORENSIC_ALLOW_GPU"):
        raise SystemExit("GPU 未授权：先设置 FORENSIC_ALLOW_GPU=1（仅核验）")
    checkpoint = args.checkpoint or "progan_train"
    report = {"schema": "candidate_official_check/1", "expert": args.expert,
              "checked_at": datetime.now().isoformat(timespec="seconds"),
              "code": load_candidate_code()["pinned"][args.expert],
              "weights": load_candidate_weights()["candidates"][args.expert]}
    if args.expert == "aide":
        report["checkpoint"] = checkpoint
    with tempfile.TemporaryDirectory(prefix=f"official_{args.expert}_") as tmp:
        if args.expert == "spai":
            report.update(_run_spai(tmp))
        elif args.expert == "aide":
            report.update(_run_aide(tmp, checkpoint=checkpoint))
        else:
            report.update(_run_drct(tmp))
    report["ok"] = (
        report.get("coverage", {}).get("ok") is True
        and all(check.get("tensor_equal") and check.get("finite") and
                check.get("analyze_equal") for check in report["checks"])
        and all(check.get("score_abs_diff", check.get("cli_score_abs_diff", 9)) <= 1e-6
                for check in report["checks"]))
    label = args.expert if args.expert != "aide" else f"aide_{checkpoint}"
    out = args.out or os.path.join(
        PROJECT_ROOT, "sft_data", "g8_expert_eval",
        f"official_check_{label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"expert": args.expert, "ok": report["ok"],
                      "coverage_keys": report["coverage"].get("model_state_keys"),
                      "coverage_missing": report["coverage"].get("missing"),
                      "gpu_peak_gb": report.get("gpu_peak_gb"),
                      "gpu_reserved_peak_gb": report.get("gpu_reserved_peak_gb"),
                      "host_maxrss_kb": report.get("host_maxrss_kb"),
                      "cgroup_current": report.get("cgroup_current")},
                     ensure_ascii=False))
    for check in report["checks"]:
        print(json.dumps(check, ensure_ascii=False))
    print(f"report: {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
