"""
SPAI offline adapter (CVPR 2025, m-ever-team/spai) — evaluation only.

Not registered in the Agent toolbox.  The official code is pinned at
`8ff7b3b6…` under `third_party/spai/repo`; the checkpoint is the user-uploaded
`weights/spai.pth` (Apache-2.0).  Loading compatibility is explicit:

* the release inference config is `configs/spai.yaml`, exactly as the official
  `infer` CLI uses it.  The checkpoint's embedded training config is ignored:
  it has `MINIMUM_PATCHES=1`, while the release yaml has 4;
* `torch.load(weights_only=True)` rejects the `yacs.config.CfgNode` embedded
  in the checkpoint; the trusted official upload is loaded with
  `weights_only=False` and `mmap=True`;
* the official `spai.utils` imports seaborn/pandas, and `spai.data` imports
  `timm.data.transforms._pil_interp` (removed in timm 1.x).  Those are import
  shims only; no official file is edited;
* the 140M-parameter model is built on `meta` and materialised directly on
  the target device, then the checkpoint is streamed tensor by tensor, so the
  host never holds the full state dict.

The score is the official `sigmoid(logit)` for the positive (AI-generated)
class; official `validate()` uses `output[:, 0]` and threshold 0.5.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch

from experts.candidate_common import (bind_clip_stub, check_state_dict,
                                      has_module, load_candidate_weights,
                                      load_mmap_checkpoint, meta_build,
                                      sidecar_sha256, stream_state_to_model)
from experts.external_base import ExternalScoreExpert

DEFAULT_REPO = "/root/autodl-tmp/third_party/spai/repo"
DEFAULT_WEIGHTS = "/root/autodl-tmp/third_party/spai/weights/spai.pth"
DEFAULT_CONFIG = "/root/autodl-tmp/third_party/spai/repo/configs/spai.yaml"
SOURCE_NAME = "spai_expert"
REQUIRED_MODULES = ("einops", "albumentations", "filetype", "lmdb",
                    "seaborn", "pandas", "timm")

PREPROCESS_DESCRIPTION = (
    "官方 SPAI 测试变换：原分辨率（不缩放/不裁剪），PadIfNeeded 到 ≥224×224，"
    "[0,1] 归一化，ToTensor；测量范围为整图（any-resolution）")


def _install_import_shims() -> None:
    """Import-time shims for the pinned code; no official file is modified."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    bind_clip_stub()
    # spai.utils imports seaborn/pandas only for an unused visualisation helper;
    # lightweight stubs keep the 2 GB cgroup from loading two large packages.
    import importlib.machinery
    import types

    for _name in ("seaborn", "pandas"):
        if _name in sys.modules:
            continue
        module = types.ModuleType(_name)
        # A stub without __spec__ makes importlib.util.find_spec() raise on
        # every later call; give it a real (loader-less) spec.
        module.__spec__ = importlib.machinery.ModuleSpec(_name, loader=None)
        sys.modules[_name] = module
    import timm.data.transforms as timm_transforms
    if not hasattr(timm_transforms, "_pil_interp"):
        from PIL import Image

        _interp = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC,
                   "nearest": Image.NEAREST}

        def _pil_interp(method: str):
            return _interp[method]

        timm_transforms._pil_interp = _pil_interp


class SPAIEvaluator:
    """Loads the official SPAI checkpoint and scores one PIL image."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS,
                 repo: str = DEFAULT_REPO,
                 device: Optional[str] = None):
        from experts.device import resolve_device

        self.weights_path = weights_path
        self.repo = repo
        self.device = torch.device(resolve_device(device))
        self._model = None
        self._transform = None
        self._cfg = None
        self._load_report: dict = {}

    # ------------------------------------------------------------------

    def _prepare_imports(self) -> None:
        _install_import_shims()
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

    @staticmethod
    def _official_config():
        """The release inference config, exactly as the official `infer` CLI uses it."""
        from spai.config import get_config

        return get_config({"cfg": DEFAULT_CONFIG, "opts": ()})

    def load(self) -> "SPAIEvaluator":
        if self._model is not None:
            return self
        self._prepare_imports()
        from spai.models.build import build_cls_model
        from spai.data.data_finetune import build_transform

        config = self._official_config()
        checkpoint = load_mmap_checkpoint(self.weights_path, weights_only=False)
        state = checkpoint["model"]
        with meta_build():
            model = build_cls_model(config)
        model.to_empty(device=self.device)
        stream_state_to_model(model, state, self.device, label="SPAI")
        self._model = model
        self._cfg = config
        self._transform = build_transform(False, config)
        self._load_report = {"model_state_keys": len(model.state_dict()),
                             "device": str(self.device)}
        # Drop the mmap container; the model now owns its copy on the device.
        del checkpoint, state
        return self

    # ------------------------------------------------------------------

    def preprocess(self, image) -> torch.Tensor:
        """Official test transform on a PIL RGB image -> float tensor [3,H,W]."""
        self.load()
        array = np.asarray(image.convert("RGB"))
        return self._transform(image=array)["image"]

    def check_preprocess(self, image) -> dict:
        """Official transform only (no model load); for CPU reports."""
        self._prepare_imports()
        from spai.data.data_finetune import build_transform

        config = self._official_config()
        transform = build_transform(False, config)
        tensor = transform(image=np.asarray(image.convert("RGB")))["image"]
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "min": float(tensor.min()), "max": float(tensor.max()),
                "config_original_resolution": bool(config.TEST.ORIGINAL_RESOLUTION),
                "config_img_size": int(config.DATA.IMG_SIZE)}

    @torch.no_grad()
    def raw_logits(self, image) -> torch.Tensor:
        # Official `validate` feeds arbitrary-resolution models a LIST of
        # 1xCxHxW tensors, not a stacked batch; the model has two different
        # forwards (`forward_batch` vs `forward_arbitrary_resolution_batch`).
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return self._model([tensor], self._cfg.MODEL.FEATURE_EXTRACTION_BATCH)

    def score(self, image) -> float:
        logits = self.raw_logits(image)
        return float(torch.sigmoid(logits).reshape(-1)[0].item())

    # ------------------------------------------------------------------

    def check_coverage(self) -> dict:
        """Build on meta and compare checkpoint keys/shapes; no tensor data read."""
        self._prepare_imports()
        from spai.models.build import build_cls_model

        config = self._official_config()
        checkpoint = load_mmap_checkpoint(self.weights_path, weights_only=False)
        state = checkpoint["model"]
        with meta_build():
            model = build_cls_model(config)
        result = check_state_dict(model, state, label="SPAI")
        return {"ok": True, "model_state_keys": len(model.state_dict()),
                "checkpoint_state_keys": len(state), **result,
                "config_source": DEFAULT_CONFIG,
                "checkpoint_config_ignored": True}

    def identity(self) -> dict:
        manifest = load_candidate_weights()["candidates"]["spai"]
        return {
            "source_name": SOURCE_NAME,
            "weights_path": self.weights_path,
            "weights_sha256": manifest["sha256"],
            "weights_bytes": manifest["bytes"],
            "repo": self.repo,
            "preprocess": PREPROCESS_DESCRIPTION,
            "scope": "global",
            "threshold": 0.5,
            "direction": "higher_is_fake",
        }


def _bgr_to_pil(img_patch: np.ndarray):
    from PIL import Image

    rgb = np.ascontiguousarray(img_patch[:, :, ::-1])
    return Image.fromarray(rgb)


class SPAIExpert(ExternalScoreExpert):
    """Evaluation-only expert boundary for SPAI (not in the Agent toolbox)."""

    source_name = SOURCE_NAME
    evidence_name = "spai_spectral_reconstruction_score"
    weights_env = ""
    weight_source = ("mever-team/spai（CVPR 2025）· 官方 Google Drive 权重，"
                     "用户 2026-10-19 上传")
    licence_note = "Apache-2.0（代码与权重，README 声明）"
    score_reference = "SPAI（CVPR 2025，频谱重建相似度，any-resolution）"

    def __init__(self, weights_path: Optional[str] = DEFAULT_WEIGHTS,
                 device: Optional[str] = None, **kwargs):
        super().__init__(weights_path=weights_path, device=device, **kwargs)
        self._evaluator: Optional[SPAIEvaluator] = None

    @classmethod
    def _resolve_weights_path(cls) -> str:
        return DEFAULT_WEIGHTS

    @staticmethod
    def _missing_dependency() -> str:
        missing = [name for name in REQUIRED_MODULES if not has_module(name)]
        if not os.path.isdir(DEFAULT_REPO):
            missing.append("pinned repo")
        return ", ".join(missing)

    def unavailable_reason(self) -> str:
        reason = super().unavailable_reason()
        if reason:
            return reason
        if not os.path.isdir(DEFAULT_REPO):
            return f"{self.source_name}：官方代码未固定到 {DEFAULT_REPO}"
        return ""

    def preload(self) -> None:
        if self._evaluator is None:
            self._evaluator = SPAIEvaluator(self.weights_path, device=self.device)
        self._evaluator.load()

    def _score(self, img_patch: np.ndarray):
        if self._evaluator is None:
            self._evaluator = SPAIEvaluator(self.weights_path, device=self.device)
        value = self._evaluator.score(_bgr_to_pil(img_patch))
        return self._result(value, preprocess=PREPROCESS_DESCRIPTION)
