"""
PROBE-DINOv2 adapter (G7, plan.md §4.16) — external AI-generated detector.

What it is
----------
PROBE ("Probing Robustness via Boundary Exploration", ICML 2026) fine-tunes a
DINOv2-L backbone with a linear head on hard samples the detector itself
steered a generator towards.  Inference is: pad the image to 336, tile it into
non-overlapping 336×336 windows, run each window, and average the logits
before a sigmoid — one **P(AI-generated)** per image, no localisation.

Reproducing their pipeline
--------------------------
`_score` mirrors `Detector/evaluate_dino.py` exactly: RGB PIL → pad to 336
(pad, not resize — `pad_image` returns the image untouched when both sides are
already ≥ 336) → `ToTensor` + ImageNet normalisation → unfold tiles →
`logits.mean().sigmoid()`.  Any deviation here would produce numbers that are
not comparable with the published ones, so the deviations are called out
rather than silently smoothed over.

Licence posture (G7-5)
----------------------
The upstream repository carries no LICENSE file and its weights are published
on ModelScope without stated terms.  This adapter is **our** interface code:
the weights are never bundled or redistributed with this project, there is no
run permission implied by being able to read this file, and the environment
variable that points at them has no default.  Only when G7-1 has been run and
the terms confirmed does this expert take part in the pipeline.

Usage
-----
    PROBE_WEIGHTS=/path/to/DINOv2_best_model_step_34999.pth \
    HF_ENDPOINT=https://hf-mirror.com \
    python -c "from experts.probe_dino import ProbeDinoExpert; ..."
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from experts.external_base import (
    ExternalScoreExpert,
    ExternalScoreExpert as _Base,
    MockExternalScoreExpert,
    tile_positions,
)

DEFAULT_BACKBONE = "facebook/dinov2-with-registers-large"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ProbeDinoExpert(ExternalScoreExpert):
    """Whole-image AI-generated score from PROBE's fine-tuned DINOv2."""

    source_name = "probe_dino_expert"
    evidence_name = "aigi_discriminator_score"
    weights_env = "PROBE_WEIGHTS"
    weight_source = ("ModelScope shuinishaojiu/PROBE-AIGI-Detection · "
                     "DINOv2_best_model_step_34999.pth（ICML 2026, arXiv 2605.24906）")
    licence_note = ("许可待确认：上游仓库无 LICENSE 文件；本项目不附带、不再分发其代码与权重，"
                    "论文引用仅为学术归属，不构成使用授权")
    score_reference = "PROBE-DINOv2 判别器（DINOv2-L 主干 + 线性头）"

    crop_size = 336
    backbone = DEFAULT_BACKBONE

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu",
                 backbone: Optional[str] = None, enabled: bool = True):
        super().__init__(weights_path, device, enabled)
        if backbone:
            self.backbone = backbone

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @staticmethod
    def _missing_dependency() -> str:
        try:
            import torch          # noqa: F401
            import transformers   # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            return f"torch/transformers（{exc}）"
        return ""

    def unavailable_reason(self) -> str:
        reason = super().unavailable_reason()
        if reason:
            return reason
        if not os.environ.get("HF_ENDPOINT") and not os.path.exists(self.backbone):
            # The backbone is a second download (≈1.2 GB).  On this network
            # huggingface.co is unreachable, so say what to set rather than
            # failing inside transformers with a connection error.
            return (f"{self.source_name}：主干 {self.backbone} 需先获取；"
                    f"本机需设置 HF_ENDPOINT=https://hf-mirror.com")
        return ""

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _load(self):  # pragma: no cover - needs the weights and a GPU/CPU load
        import torch
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(self.backbone)
        head = torch.nn.Linear(backbone.config.hidden_size, 1)
        model = torch.nn.Module()
        model.backbone = backbone
        model.fc = head

        checkpoint = torch.load(self.weights_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        self._load_report = {"missing": list(missing), "unexpected": list(unexpected)}
        model.eval().to(self.device)
        self._model = model

    @property
    def load_report(self) -> Dict[str, List[str]]:
        """Keys the checkpoint did not provide / that did not fit (diagnostics)."""
        return getattr(self, "_load_report", {"missing": [], "unexpected": []})

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _prepare(self, img_bgr: np.ndarray):  # pragma: no cover - torch path
        """Their preprocessing, in their order: pad → tensor → ImageNet norm."""
        import torch
        from PIL import Image

        rgb = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
        width, height = rgb.size
        if width < self.crop_size or height < self.crop_size:
            padded = Image.new("RGB", (max(width, self.crop_size),
                                       max(height, self.crop_size)))
            padded.paste(rgb, (0, 0))
            rgb = padded
        tensor = torch.from_numpy(np.asarray(rgb)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (tensor - mean) / std

    def _score(self, img_bgr: np.ndarray):  # pragma: no cover - needs the model
        import torch

        if self._model is None:
            self._load()
        tensor = self._prepare(img_bgr)
        _, height, width = tensor.shape
        tiles = []
        for (x, y, w, h) in tile_positions(height, width, self.crop_size):
            tiles.append(tensor[:, y:y + h, x:x + w])
        batch = torch.stack(tiles).to(self.device)
        with torch.no_grad():
            logits = self._model.fc(self._model.backbone(batch).last_hidden_state[:, 0])
            mean_logit = float(logits.mean().item())
            score = float(torch.sigmoid(torch.tensor(mean_logit)).item())
        return self._result(
            score,
            preprocess=(f"{width}×{height} → pad to ≥{self.crop_size} → "
                        f"{len(tiles)} 个 {self.crop_size}² 滑窗 → 逐窗 logit 均值 → sigmoid"),
            extra={"tiles": len(tiles), "mean_logit": round(mean_logit, 6),
                   "crop_size": self.crop_size, "backbone": self.backbone,
                   "load_report": self.load_report},
        )


class MockProbeExpert(MockExternalScoreExpert):
    """Deterministic PROBE stand-in: same source name, no model, CPU-only."""

    source_name = "probe_dino_expert"
    evidence_name = "aigi_discriminator_score"
    score_reference = "PROBE-DINOv2 判别器（Mock）"
    weights_env = "PROBE_WEIGHTS"
    weight_source = "mock"
    licence_note = "mock（不涉及上游权重）"
