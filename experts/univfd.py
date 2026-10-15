"""
UnivFD (CVPR 2023) — scoring adapter for the stage-3 evaluation.

The model is a CLIP ViT-L/14 backbone plus a linear head; the repo ships only
the head (`fc_weights.pth`, 768→1), so the backbone has to come from OpenAI's
official `ViT-L-14.pt`.  That file is a **TorchScript archive** — OpenAI's own
`clip.load` does `torch.jit.load` on it — which is what this adapter uses: it
needs no `ftfy`/CLIP python package (neither is installed here), and it is the
same object the official code would have produced.  Which route was used is
recorded in `identity()`, because "same weights, different loader" is exactly
the kind of difference that changes results quietly.

Preprocessing is the official one and is **not** the CLIP default: RGB →
`CenterCrop(224)` → `ToTensor` → CLIP normalisation, with **no resize**.  That
centre crop is why the registry records this expert's measurement scope as
`center_crop` rather than `global`: it does not see the image edges, and a
token that calls its result a whole-image measurement is wrong.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms

from utils.score_semantics import entry

SOURCE_NAME = "univfd_expert"
DEFAULT_HEAD = "/root/autodl-tmp/third_party/univfd/weights/fc_weights.pth"
DEFAULT_BACKBONE = "/root/autodl-tmp/third_party/univfd/clip_cache/ViT-L-14.pt"

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class UnivFDEvaluator:
    def __init__(self, head_path: str = DEFAULT_HEAD,
                 backbone_path: str = DEFAULT_BACKBONE,
                 device: str = "cpu"):
        self.head_path = head_path
        self.backbone_path = backbone_path
        self.device = device
        self._record = entry(SOURCE_NAME, variant="univfd")
        self._clip = None
        self._head = None
        self._dtype = None
        self._loader_used: Optional[str] = None
        self._transform = transforms.Compose([
            transforms.CenterCrop(224),          # 官方流程：先裁后张量化，没有 resize
            transforms.ToTensor(),
            transforms.Normalize(mean=list(CLIP_MEAN), std=list(CLIP_STD)),
        ])

    @property
    def name(self) -> str:
        return "UnivFD(CLIP:ViT-L/14)"

    def identity(self) -> dict:
        weights = self._record["weights"]
        return {
            "expert": SOURCE_NAME,
            "head_file": self.head_path,
            "head_bytes": os.path.getsize(self.head_path) if os.path.exists(self.head_path) else None,
            "head_sha256": weights["head"]["sha256"],
            "backbone_file": self.backbone_path,
            "backbone_url": weights["backbone"]["url"],
            "backbone_sha256_from_url": weights["backbone"]["sha256_from_url"],
            "backbone_loader": self._loader_used,
            "backbone_dtype": str(self._dtype) if self._dtype is not None else None,
            "repo_commit": weights.get("repo_commit"),
            "preprocess": self._record["preprocess"]["name"],
            "preprocess_detail": self._record["preprocess"]["detail"],
            "measurement_scope": self._record["preprocess"]["measurement_scope"],
            "licence": weights["licence"],
        }

    # ------------------------------------------------------------------

    def load(self) -> "UnivFDEvaluator":
        if self._clip is not None:
            return self
        if not os.path.exists(self.backbone_path):
            raise FileNotFoundError(
                f"缺 CLIP 主干：{self.backbone_path}\n"
                f"官方地址：{self._record['weights']['backbone']['url']}")
        # OpenAI 的 .pt 是 TorchScript 存档；官方 clip.load 也是这么用的。
        self._clip = torch.jit.load(self.backbone_path, map_location="cpu")
        # 设备与精度必须一致：.pt 里的权重是 fp16，CPU 上 fp16 卷积又慢又未必支持，
        # 所以 CPU 跑时把主干整体转 fp32，GPU 跑时保持 fp16 并一起搬过去。
        if self.device.startswith("cuda"):
            self._clip = self._clip.to(self.device)
        else:
            self._clip = self._clip.float()
        self._clip.eval()
        self._dtype = next(self._clip.parameters()).dtype
        self._loader_used = "torch.jit.load"

        head = torch.nn.Linear(768, 1)
        state = torch.load(self.head_path, map_location="cpu")
        if isinstance(state, dict) and "fc.weight" in state:      # 有人会把整个模型存进来
            state = {"weight": state["fc.weight"], "bias": state["fc.bias"]}
        head.load_state_dict(state)
        head.eval().to(self.device)
        self._head = head
        return self

    # ------------------------------------------------------------------

    def features(self, image: Image.Image) -> torch.Tensor:
        """CLIP image features — separated so a later stage can reuse them."""
        self.load()
        if isinstance(image, str):
            image = Image.open(image)
        tensor = self._transform(image.convert("RGB")).unsqueeze(0)
        # 主干是 fp16（.pt 里的权重如此），输入也必须同 dtype
        tensor = tensor.to(self._dtype)
        with torch.no_grad():
            return self._clip.encode_image(tensor).float()

    def score(self, image: Image.Image | str) -> float:
        """Official: linear head on CLIP features, then sigmoid (0=Fake? no: Real=0/Fake=1)."""
        features = self.features(image)
        with torch.no_grad():
            logit = self._head(features.to(self.device))
        return float(torch.sigmoid(logit).flatten()[0].item())
