"""
Shared plumbing for the offline candidate adapters (SPAI / AIDE).

These are **evaluation-only** adapters: they are not registered in the Agent
toolbox, not added to `main.build_pipeline`, and not described to the model.
They exist so `scripts/g8_external_eval.py` can measure the candidates on the
same inputs as PROBE without changing the Agent's prompts, stopping rules or
evidence semantics.

Two loading facts drive this module:

* the checkpoints are large (AIDE 3.59 GB each).  The host cgroup is 2 GB, so
  loading the whole state dict into CPU RAM is not possible.  The model is
  built on the `meta` device and materialised directly on CUDA; each tensor is
  streamed from the mmap checkpoint to the GPU and released.
* the official code is pinned under `third_party/` and is imported read-only.
  Compatibility shims are explicit and limited to what inference needs; they
  never rewrite the official files.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from typing import Dict, Iterable, Optional

import torch

from config import PROJECT_ROOT

CANDIDATE_CODE_PATH = os.path.join(PROJECT_ROOT, "experts", "candidate_code.json")
CANDIDATE_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "experts", "candidate_weights.json")


def load_candidate_code() -> dict:
    with open(CANDIDATE_CODE_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def load_candidate_weights() -> dict:
    with open(CANDIDATE_WEIGHTS_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def has_module(name: str) -> bool:
    """True when importable or present as an explicit stub with a spec."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        import sys

        return name in sys.modules


@contextlib.contextmanager
def meta_build():
    """
    Build a model on `meta` (no allocation) without tripping `.item()`.

    The official ViT constructor calls `torch.linspace(...).item()` to build
    the stochastic-depth decay list.  On a meta tensor `.item()` is not
    implemented; returning 0 for that one construction-time value is safe for
    inference because drop-path is inactive in `eval()` and every value is
    overwritten by the checkpoint anyway.  The patch is restored afterwards.
    """
    original_item = torch.Tensor.item

    def _item(self):
        if self.device.type == "meta":
            return 0.0
        return original_item(self)

    torch.Tensor.item = _item
    try:
        with torch.device("meta"):
            yield
    finally:
        torch.Tensor.item = original_item


def bind_clip_stub() -> None:
    """The official modules import OpenAI `clip`; the selected models never call it."""
    import sys
    import types

    if "clip" not in sys.modules:
        sys.modules["clip"] = types.ModuleType("clip")


def check_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor],
                     *, label: str) -> Dict[str, list]:
    """Return missing / unexpected / shape-mismatch keys; no tensor data is read."""
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = [
        [key, list(expected[key].shape), list(state[key].shape)]
        for key in sorted(set(expected) & set(state))
        if tuple(expected[key].shape) != tuple(state[key].shape)
    ]
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            f"{label}: checkpoint does not cover the built model "
            f"(missing {len(missing)}, unexpected {len(unexpected)}, "
            f"shape mismatch {len(shape_mismatch)}). "
            f"missing={missing[:8]} unexpected={unexpected[:8]} "
            f"shape={shape_mismatch[:8]}")
    return {"missing": missing, "unexpected": unexpected,
            "shape_mismatch": shape_mismatch}


def stream_state_to_model(model: torch.nn.Module, state: Dict[str, torch.Tensor],
                          device: torch.device, *, label: str) -> None:
    """
    Copy each checkpoint tensor to `device` one at a time.

    The model must already be materialised on `device` (via `to_empty`).  No
    full CPU copy of the checkpoint is made: `torch.load(mmap=True)` gives
    mmap tensors, and only the current tensor is read/copied.
    """
    expected = model.state_dict()
    check_state_dict(model, state, label=label)
    with torch.no_grad():
        for name, target in expected.items():
            source = state[name]
            target.copy_(source.to(device=device, non_blocking=False))
            del source
    model.eval()


def load_mmap_checkpoint(path: str, *, weights_only: bool):
    """
    Load a checkpoint without materialising it in CPU RAM.

    `weights_only=False` is only used for the trusted SPAI official upload
    (its checkpoint contains a `yacs.config.CfgNode`, which the safe
    unpickler rejects).  The caller records the sha256 in
    `experts/candidate_weights.json` before this is called.
    """
    return torch.load(path, map_location="cpu", mmap=True,
                      weights_only=weights_only)


def sidecar_sha256(path: str, digest: str) -> None:
    """Record the already-computed digest so `sha256_file` does not re-read GBs."""
    sidecar = f"{path}.sha256"
    if os.path.exists(sidecar):
        return
    with open(sidecar, "w", encoding="utf-8") as handle:
        handle.write(f"{digest}  {os.path.basename(path)}\n")
