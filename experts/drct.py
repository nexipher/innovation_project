"""
DRCT offline adapter (ICML 2024, beibuwandeluori/DRCT) — evaluation only.

Not registered in the Agent toolbox.  The official code is pinned at
`01aa7d0b6de903cf3208801f8f6b469edffbc762` under
`third_party/drct/repo`; the selected checkpoint is the ConvB / DRCT-2M /
SDv1.4 (`sdv14`) checkpoint from the official `pretrained.zip`:

  DRCT-2M/sdv14/convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth

Official test path (`scripts/test_DRCT-2M.sh`):

* `--model_name convnext_base_in22k --embedding_size 1024`
* `--is_crop`, so `create_val_transforms(224, is_crop=True)`:
  PadIfNeeded(224) -> CenterCrop(224) -> PadIfNeeded(224) ->
  ImageNet Normalize -> ToTensorV2.  No resize.  The official reader is
  `cv2.imread` then BGR->RGB.
* output is `1 - softmax(logits, dim=1)[:, 0]`, i.e. the class-1
  probability; class 1 is "Fake" in `data/dataset.py`.  Threshold 0.5.
* inference does **not** run diffusion reconstruction (`--is_dire` is not
  passed); the checkpoint is the detector itself.

The model is built on `meta`, materialised on the target device and loaded
from an mmap checkpoint, so no full CPU copy of the state dict is made.
"""

from __future__ import annotations

import os
import sys
import types
from typing import Optional

import numpy as np
import torch

from experts.candidate_common import (check_state_dict, has_module,
                                      load_candidate_weights,
                                      load_mmap_checkpoint, meta_build,
                                      stream_state_to_model)
from experts.external_base import ExternalScoreExpert

DEFAULT_REPO = "/root/autodl-tmp/third_party/drct/repo"
DEFAULT_WEIGHTS = (
    "/root/autodl-tmp/third_party/drct/weights/DRCT-2M/sdv14/"
    "convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth"
)
SOURCE_NAME = "drct_expert"
REQUIRED_MODULES = ("timm", "albumentations", "cv2", "transformers")
MODEL_NAME = "convnext_base_in22k"
EMBEDDING_SIZE = 1024
INPUT_SIZE = 224
PREPROCESS_DESCRIPTION = (
    "官方 DRCT 测试变换（is_crop=True）：RGB 图 PadIfNeeded 到 ≥224×224 → "
    "CenterCrop 224×224 → 再 PadIfNeeded → ImageNet 归一化 → ToTensor；"
    "不缩放。测量范围为整图（中心裁剪 224）。"
)


def _install_import_shims() -> None:
    """The official model module imports OpenAI clip; ConvB never uses it."""
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    if "clip" not in sys.modules:
        sys.modules["clip"] = types.ModuleType("clip")


class DRCTEvaluator:
    """Loads the official ConvB DRCT checkpoint and scores one PIL image."""

    def __init__(self, weights_path: str = DEFAULT_WEIGHTS,
                 repo: str = DEFAULT_REPO, device: Optional[str] = None):
        from experts.device import resolve_device

        self.weights_path = weights_path
        self.repo = repo
        self.device = torch.device(resolve_device(device))
        self._model = None
        self._transform = None
        self._load_report: dict = {}

    def _prepare_imports(self) -> None:
        _install_import_shims()
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

    def _build_model(self):
        self._prepare_imports()
        from network.models import get_models

        with meta_build():
            return get_models(model_name=MODEL_NAME, num_classes=2,
                              pretrained=False,
                              embedding_size=EMBEDDING_SIZE)

    def load(self) -> "DRCTEvaluator":
        if self._model is not None:
            return self
        model = self._build_model()
        model.to_empty(device=self.device)
        state = load_mmap_checkpoint(self.weights_path, weights_only=True)
        stream_state_to_model(model, state, self.device, label="DRCT")
        self._model = model
        self._load_report = {"model_state_keys": len(model.state_dict()),
                             "device": str(self.device)}
        del state
        return self

    def _get_transform(self):
        if self._transform is None:
            self._prepare_imports()
            from data.transform import create_val_transforms

            self._transform = create_val_transforms(
                size=INPUT_SIZE, is_crop=True)
        return self._transform

    def preprocess(self, image) -> torch.Tensor:
        """Official val transform on a PIL RGB image -> float tensor [3,224,224]."""
        array = np.asarray(image.convert("RGB"))
        return self._get_transform()(image=array)["image"]

    def check_preprocess(self, image) -> dict:
        """Official transform only (no model load); for CPU reports."""
        tensor = self.preprocess(image)
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "min": float(tensor.min()), "max": float(tensor.max()),
                "input_size": INPUT_SIZE, "is_crop": True}

    @torch.no_grad()
    def raw_logits(self, image) -> torch.Tensor:
        self.load()
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        return self._model(batch)

    def score(self, image) -> float:
        logits = self.raw_logits(image)
        # Official eval_model: outputs.append(1 - y_pred[:, 0]).
        return float(1.0 - torch.softmax(logits, dim=1)[:, 0].item())

    def check_coverage(self) -> dict:
        """Build on meta and compare checkpoint keys/shapes; no tensor data read."""
        model = self._build_model()
        state = load_mmap_checkpoint(self.weights_path, weights_only=True)
        result = check_state_dict(model, state, label="DRCT")
        return {"ok": True, "model_state_keys": len(model.state_dict()),
                "checkpoint_state_keys": len(state),
                "params_millions": sum(v.numel() for v in state.values()) / 1e6,
                **result}

    def identity(self) -> dict:
        manifest = load_candidate_weights()["candidates"]["drct"]
        return {
            "source_name": SOURCE_NAME,
            "weights_path": self.weights_path,
            "weights_sha256": manifest["sha256"],
            "weights_bytes": manifest["bytes"],
            "archive_path": manifest["archive_path"],
            "archive_sha256": manifest["archive_sha256"],
            "checkpoint": manifest["checkpoint"],
            "repo": self.repo,
            "preprocess": PREPROCESS_DESCRIPTION,
            "scope": "global",
            "threshold": 0.5,
            "direction": "higher_is_fake",
        }


def _bgr_to_pil(img_patch: np.ndarray):
    from PIL import Image

    array = np.ascontiguousarray(img_patch)
    if array.ndim == 3 and array.shape[2] >= 3:
        array = array[:, :, :3][:, :, ::-1]
    return Image.fromarray(array.astype(np.uint8), mode="RGB")


class DRCTExpert(ExternalScoreExpert):
    """Evaluation-only expert boundary for DRCT ConvB (not in the Agent toolbox)."""

    source_name = SOURCE_NAME
    evidence_name = "drct_diffusion_reconstruction_score"
    weights_env = ""
    weight_source = ("beibuwandeluori/DRCT（ICML 2024）· 官方 ModelScope "
                     "DRCT-2M pretrained.zip，用户 2026-10-26 上传")
    licence_note = "仓库无 LICENSE；权重许可未单独声明"
    score_reference = "DRCT（ICML 2024，扩散重建对比训练的 ConvB 检测器）"

    def __init__(self, weights_path: Optional[str] = DEFAULT_WEIGHTS,
                 device: Optional[str] = None, **kwargs):
        super().__init__(weights_path=weights_path, device=device, **kwargs)
        self._evaluator: Optional[DRCTEvaluator] = None

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
            self._evaluator = DRCTEvaluator(self.weights_path, device=self.device)
        self._evaluator.load()

    def _score(self, img_patch: np.ndarray):
        if self._evaluator is None:
            self._evaluator = DRCTEvaluator(self.weights_path, device=self.device)
        value = self._evaluator.score(_bgr_to_pil(img_patch))
        return self._result(value, preprocess=PREPROCESS_DESCRIPTION)
