"""
AIDE offline adapter (ICLR 2025, shilinyan99/AIDE) — evaluation only.

Not registered in the Agent toolbox.  The official code is pinned at
`6725b710…` under `third_party/aide/repo`; the primary checkpoint is the
user-uploaded `progan_train.pth` (the README's example training set).  The
other two checkpoints (`sd14_train`, `GenImage_train`) are retained but not
silently mixed into the primary candidate.

The 3.59 GB checkpoint already contains the frozen OpenCLIP ConvNeXt-XXLarge
trunk (`openclip_convnext_xxl.*`, 378 tensors, 897.9M parameters total), so no
separate ConvNeXt checkpoint is downloaded.  The official `AIDE_Model` asks
open_clip to load that trunk with `pretrained=convnext_path`; the adapter
passes `pretrained=None` and loads the checkpoint instead.  Loading is
streamed onto the device from an mmap checkpoint so the host never holds the
full state dict (the earlier observed 2 GB cgroup could not have held it).

Official preprocessing (TestDataset): PIL RGB -> ToTensor -> DCT patch
selection (4 variants) -> Resize 256x256 -> ImageNet Normalize; the original
image is also resized/normalized.  Stack order is
`[x_minmin, x_maxmax, x_minmin1, x_maxmax1, x_0]`.  Output is
`softmax(logits, dim=1)[:, 1]` (fake), threshold 0.5.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch

from experts.candidate_common import (bind_clip_stub, check_state_dict,
                                      has_module, load_candidate_weights,
                                      load_mmap_checkpoint, meta_build,
                                      stream_state_to_model)
from experts.external_base import ExternalScoreExpert

DEFAULT_REPO = "/root/autodl-tmp/third_party/aide/repo"
DEFAULT_WEIGHTS_DIR = "/root/autodl-tmp/third_party/aide/weights"
CHECKPOINTS = {
    "progan_train": "progan_train.pth",
    "sd14_train": "sd14_train.pth",
    "GenImage_train": "GenImage_train.pth",
}
PRIMARY_CHECKPOINT = "progan_train"
SOURCE_NAME = "aide_expert"
REQUIRED_MODULES = ("open_clip", "einops", "timm", "torchvision")

PREPROCESS_DESCRIPTION = (
    "官方 AIDE 测试预处理：PIL RGB → ToTensor → DCT 极值 patch（4 个变体）→ "
    "Resize 256×256 → ImageNet 归一化；原图同样 Resize 256 并归一化；"
    "输入为 5×3×256×256。测量为整图分类（patch 用于特征，不定位）")


class AIDEEvaluator:
    """Loads one official AIDE checkpoint and scores one PIL image."""

    def __init__(self, checkpoint: str = PRIMARY_CHECKPOINT,
                 weights_dir: str = DEFAULT_WEIGHTS_DIR,
                 repo: str = DEFAULT_REPO,
                 device: Optional[str] = None):
        from experts.device import resolve_device

        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"未知 AIDE checkpoint {checkpoint!r}；"
                             f"只有 {sorted(CHECKPOINTS)}")
        self.checkpoint = checkpoint
        self.weights_path = os.path.join(weights_dir, CHECKPOINTS[checkpoint])
        self.repo = repo
        self.device = torch.device(resolve_device(device))
        self._model = None
        self._to_tensor = None
        self._dct = None
        self._post = None

    # ------------------------------------------------------------------

    def _prepare_imports(self) -> None:
        bind_clip_stub()
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

    def _build_meta_model(self):
        """
        Build the official AIDE_Model on meta without downloading the trunk.

        The checkpoint contains the ConvNeXt-XXL trunk, so the factory is
        asked for the architecture with `pretrained=None`; `device="meta"`
        keeps the 3.6 GB trunk unallocated until `to_empty`.
        """
        import open_clip
        from models.AIDE import AIDE_Model

        original = open_clip.create_model_and_transforms

        def _factory(model_name, pretrained=None, **kwargs):
            kwargs.setdefault("device", "meta")
            return original(model_name, pretrained=None, **kwargs)

        open_clip.create_model_and_transforms = _factory
        try:
            with meta_build():
                return AIDE_Model(resnet_path=None, convnext_path=None)
        finally:
            open_clip.create_model_and_transforms = original

    def load(self) -> "AIDEEvaluator":
        if self._model is not None:
            return self
        self._prepare_imports()
        from torchvision import transforms
        from data.dct import DCT_base_Rec_Module

        checkpoint = load_mmap_checkpoint(self.weights_path, weights_only=True)
        state = checkpoint["model"]
        model = self._build_meta_model()
        model.to_empty(device=self.device)
        stream_state_to_model(model, state, self.device, label="AIDE")
        self._model = model
        del checkpoint, state

        self._to_tensor = transforms.ToTensor()
        self._dct = DCT_base_Rec_Module()
        self._post = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        return self

    # ------------------------------------------------------------------

    def preprocess(self, image) -> torch.Tensor:
        """Official TestDataset pipeline -> float tensor [5,3,256,256]."""
        self.load()
        tensor = self._to_tensor(image.convert("RGB"))
        parts = self._dct(tensor)
        transformed = [self._post(part) for part in parts]
        transformed.append(self._post(tensor))
        return torch.stack(transformed, dim=0)

    def check_preprocess(self, image) -> dict:
        """Official TestDataset pipeline only (no model load)."""
        self._prepare_imports()
        from torchvision import transforms
        from data.dct import DCT_base_Rec_Module

        to_tensor = transforms.ToTensor()
        dct = DCT_base_Rec_Module()
        post = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        tensor = to_tensor(image.convert("RGB"))
        parts = dct(tensor)
        batch = torch.stack([post(part) for part in parts] + [post(tensor)],
                            dim=0)
        return {"shape": list(batch.shape), "dtype": str(batch.dtype),
                "min": float(batch.min()), "max": float(batch.max()),
                "patch_shapes": [list(part.shape) for part in parts],
                "original_shape": list(tensor.shape)}

    @torch.no_grad()
    def raw_logits(self, image) -> torch.Tensor:
        batch = self.preprocess(image).unsqueeze(0).to(self.device)
        return self._model(batch)

    def score(self, image) -> float:
        logits = self.raw_logits(image)
        return float(torch.softmax(logits, dim=1)[:, 1].item())

    # ------------------------------------------------------------------

    def check_coverage(self) -> dict:
        """Build on meta and compare checkpoint keys/shapes; no tensor data read."""
        self._prepare_imports()
        checkpoint = load_mmap_checkpoint(self.weights_path, weights_only=True)
        state = checkpoint["model"]
        model = self._build_meta_model()
        result = check_state_dict(model, state, label="AIDE")
        backbone = sorted(key for key in state
                          if key.startswith("openclip_convnext_xxl."))
        return {"ok": True, "checkpoint": self.checkpoint,
                "model_state_keys": len(model.state_dict()),
                "checkpoint_state_keys": len(state),
                "backbone_keys": len(backbone),
                "params_millions": sum(v.numel() for v in state.values()) / 1e6,
                **result}

    def identity(self) -> dict:
        manifest = load_candidate_weights()["candidates"]["aide"]
        entry = manifest["checkpoints"][self.checkpoint]
        return {
            "source_name": SOURCE_NAME,
            "checkpoint": self.checkpoint,
            "weights_path": self.weights_path,
            "weights_sha256": entry["sha256"],
            "weights_bytes": entry["bytes"],
            "backbone_included": manifest["backbone_included"],
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


class AIDEExpert(ExternalScoreExpert):
    """Evaluation-only expert boundary for AIDE (not in the Agent toolbox)."""

    source_name = SOURCE_NAME
    evidence_name = "aide_hybrid_feature_score"
    weights_env = ""
    weight_source = ("shilinyan99/AIDE（ICLR 2025）· 官方 Google Drive 权重，"
                     "用户 2026-10-19 上传")
    licence_note = "代码 MIT；checkpoint 许可未单独声明"
    score_reference = "AIDE（ICLR 2025，CLIP 语义 + DCT/SRM patch 混合特征）"

    def __init__(self, weights_path: Optional[str] = None,
                 checkpoint: str = PRIMARY_CHECKPOINT,
                 device: Optional[str] = None, **kwargs):
        self.checkpoint = checkpoint
        if weights_path is None:
            weights_path = os.path.join(DEFAULT_WEIGHTS_DIR,
                                        CHECKPOINTS[checkpoint])
        super().__init__(weights_path=weights_path, device=device, **kwargs)
        self._evaluator: Optional[AIDEEvaluator] = None

    @classmethod
    def _resolve_weights_path(cls) -> str:
        return os.path.join(DEFAULT_WEIGHTS_DIR, CHECKPOINTS[PRIMARY_CHECKPOINT])

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

    def _score(self, img_patch: np.ndarray):
        if self._evaluator is None:
            self._evaluator = AIDEEvaluator(checkpoint=self.checkpoint,
                                            device=self.device,
                                            weights_dir=os.path.dirname(
                                                self.weights_path))
        value = self._evaluator.score(_bgr_to_pil(img_patch))
        return self._result(value, preprocess=PREPROCESS_DESCRIPTION)
