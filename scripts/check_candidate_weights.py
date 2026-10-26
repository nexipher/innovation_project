"""
CPU checks for the SPAI/AIDE/DRCT candidate adapters (plan §4.17).

Run with the isolated candidate environment:

  /root/autodl-tmp/venvs/candidates/bin/python scripts/check_candidate_weights.py --hash

What it verifies, without loading model weights into host RAM:

  * file size against `experts/candidate_weights.json` (and sha256 with --hash);
  * key/shape coverage of the built model against the checkpoint (meta build);
  * the official preprocessing on a synthetic 500×332 image;
  * the AIDE checkpoint already contains the ConvNeXt-XXL trunk.

It does NOT run a forward pass and does NOT touch the GPU.  The GPU smoke
budget is reported separately; a real run needs the project's GPU
authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402
from experts.candidate_common import load_candidate_weights  # noqa: E402


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_files(hashes: bool) -> dict:
    manifest = load_candidate_weights()
    report = {"checked": {}, "problems": []}

    def check(label: str, path: str, expected_bytes: int, expected_sha: str):
        entry = {"path": path, "exists": os.path.isfile(path)}
        if not entry["exists"]:
            report["problems"].append(f"{label}: 文件不存在 {path}")
            report["checked"][label] = entry
            return
        size = os.path.getsize(path)
        entry["bytes"] = size
        entry["bytes_ok"] = size == expected_bytes
        if not entry["bytes_ok"]:
            report["problems"].append(
                f"{label}: 大小不符（期望 {expected_bytes}，实际 {size}）")
        if hashes:
            actual = _sha256(path)
            entry["sha256"] = actual
            entry["sha256_ok"] = actual == expected_sha
            if not entry["sha256_ok"]:
                report["problems"].append(
                    f"{label}: sha256 不符（期望 {expected_sha[:16]}…，"
                    f"实际 {actual[:16]}…）")
        report["checked"][label] = entry

    spai = manifest["candidates"]["spai"]
    check("spai", spai["path"], spai["bytes"], spai["sha256"])
    for name, entry in manifest["candidates"]["aide"]["checkpoints"].items():
        check(f"aide/{name}", entry["path"], entry["bytes"], entry["sha256"])
    drct = manifest["candidates"]["drct"]
    check("drct/archive", drct["archive_path"], drct["archive_bytes"],
          drct["archive_sha256"])
    check("drct/checkpoint", drct["path"], drct["bytes"], drct["sha256"])
    return report


def _synthetic_image():
    from PIL import Image
    import numpy as np

    height, width = 332, 500
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack([
        (xx * 255 // max(width - 1, 1)).astype("uint8"),
        (yy * 255 // max(height - 1, 1)).astype("uint8"),
        ((xx + yy) * 255 // max(width + height - 2, 1)).astype("uint8"),
    ], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _check_spai() -> dict:
    from experts.spai import SPAIEvaluator

    evaluator = SPAIEvaluator(device="cpu")
    try:
        coverage = evaluator.check_coverage()
        preprocess = evaluator.check_preprocess(_synthetic_image())
        return {"ok": True, "coverage": coverage, "preprocess": preprocess}
    except Exception as exc:
        import traceback

        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}


def _check_aide(all_checkpoints: bool) -> dict:
    from experts.aide import CHECKPOINTS, PRIMARY_CHECKPOINT, AIDEEvaluator

    names = list(CHECKPOINTS) if all_checkpoints else [PRIMARY_CHECKPOINT]
    results = {}
    for name in names:
        evaluator = AIDEEvaluator(checkpoint=name, device="cpu")
        try:
            results[name] = {
                "ok": True,
                "coverage": evaluator.check_coverage(),
                "preprocess": evaluator.check_preprocess(_synthetic_image()),
            }
        except Exception as exc:
            import traceback

            results[name] = {"ok": False,
                             "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()[-2000:]}
    return results


def _check_drct() -> dict:
    from experts.drct import DRCTEvaluator

    evaluator = DRCTEvaluator(device="cpu")
    try:
        coverage = evaluator.check_coverage()
        preprocess = evaluator.check_preprocess(_synthetic_image())
        return {"ok": True, "coverage": coverage, "preprocess": preprocess}
    except Exception as exc:
        import traceback

        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:]}


def _check_drct_subprocess() -> dict:
    """Run the DRCT section in a fresh process: AIDE and DRCT both use `data`."""
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory(prefix="drct_cpu_check_") as tmp:
        out = os.path.join(tmp, "report.json")
        completed = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--only", "drct", "--out", out],
            capture_output=True, text=True)
        if os.path.exists(out):
            payload = json.load(open(out, encoding="utf-8"))
            result = payload.get("drct", {})
            result["_subprocess"] = {
                "returncode": completed.returncode,
                "stderr_tail": completed.stderr[-1000:],
            }
            return result
        return {"ok": False,
                "error": f"DRCT subprocess failed (rc={completed.returncode})",
                "_subprocess": {"returncode": completed.returncode,
                                "stderr_tail": completed.stderr[-1000:]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hash", action="store_true",
                        help="also recompute sha256 of all candidate files (slow, multi-GB read)")
    parser.add_argument("--all-aide", action="store_true",
                        help="check key coverage of sd14_train and GenImage_train too")
    parser.add_argument("--only", choices=("all", "spai", "aide", "drct"), default="all",
                        help="AIDE and DRCT both use a top-level `data` package; "
                             "run them with separate --only aide / --only drct "
                             "processes instead of combining them")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import torch
    import timm

    report = {
        "schema": "candidate_cpu_checks/1",
        "checked_at": datetime.now().isoformat(timespec="seconds"),
        "interpreter": sys.executable,
        "environment": {
            "torch": torch.__version__,
            "timm": timm.__version__,
        },
        "hash_verified": bool(args.hash),
        "weights": _check_files(args.hash),
    }
    if args.only in ("all", "spai"):
        report["spai"] = _check_spai()
    if args.only in ("all", "aide"):
        report["aide"] = _check_aide(args.all_aide)
    if args.only == "drct":
        report["drct"] = _check_drct()
    elif args.only == "all":
        report["drct"] = _check_drct_subprocess()
    try:
        import einops
        import open_clip
        report["environment"]["einops"] = einops.__version__
        report["environment"]["open_clip"] = "installed"
    except Exception:
        pass

    out = args.out or os.path.join(
        PROJECT_ROOT, "sft_data", "g8_expert_eval",
        f"cpu_checks_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"interpreter: {sys.executable}")
    print(f"weights: {report['weights']['problems'] or 'size/sha OK'}")
    for name in ("spai", "aide", "drct"):
        if name not in report:
            continue
        section = report[name]
        if name in ("spai", "drct"):
            print(f"{name.upper()}: ok={section['ok']} "
                  f"{section.get('coverage', {}).get('model_state_keys')} keys "
                  f"{section.get('error', '')}")
        else:
            for checkpoint, entry in section.items():
                print(f"AIDE/{checkpoint}: ok={entry['ok']} "
                      f"{entry.get('coverage', {}).get('model_state_keys')} keys "
                      f"backbone={entry.get('coverage', {}).get('backbone_keys')} "
                      f"{entry.get('error', '')}")
    print(f"report: {out}")
    return 0 if not report["weights"]["problems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
