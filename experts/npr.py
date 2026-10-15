"""
NPR (CVPR 2024) — scoring adapter for the stage-3 evaluation.

Stage 3 measures experts on their own; the evidence interface comes later
(stage 4), so this file is deliberately just a scorer: load the official code
and weights, apply the registered preprocessing, return the raw score.

Three things it refuses to guess, all of which the official repo forces you to
get right:

  * **which checkpoint** — the repo ships two, same architecture, different
    save formats and different documented recipes.  The registry names them
    (`npr-a`, `npr-b`); this adapter takes the id and reads the file it names.
  * **how to load that file** — `validate.py` reads `state_dict['model']`,
    `test.py` loads the raw dict.  For `NPR.pth` only the first works; for
    `model_epoch_last_3090.pth` only the second.  Verified locally, both ways,
    on 2026-10-15.  The adapter tries the recorded convention, falls back to
    the other, and **raises if neither works** — and if the fallback was the
    one that worked, it says so, because then the registry is wrong.
  * **the recipe** — the AIGCDetectBenchmark recipe (no resize, no crop: the
    whole image) is used here, the GenImage recipe (tile-pad + centre crop)
    is not, and the registry records which one this is so a later reader
    cannot mix them.

The official code is imported from its clone under `third_party/`; it is not
copied into this repository (its licence is unstated — see the registry).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from utils.score_semantics import entry

DEFAULT_REPO = os.environ.get("NPR_REPO", "/root/autodl-tmp/third_party/npr/repo")
SOURCE_NAME = "npr_expert"
LOADERS = {"state_dict['model']": lambda blob: blob["model"] if isinstance(blob, dict) else None,
           "raw_state_dict": lambda blob: blob if isinstance(blob, dict) else None}


def _import_resnet(repo: str):
    """Import `networks/resnet.py` from the clone without importing the package."""
    path = os.path.join(repo, "networks", "resnet.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPR 仓库里没有 {path}（NPR_REPO 指对了吗？）")
    spec = importlib.util.spec_from_file_location("npr_networks_resnet", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["npr_networks_resnet"] = module
    spec.loader.exec_module(module)
    return module


class NPREvaluator:
    """Loads one named NPR checkpoint and scores whole images."""

    def __init__(self, checkpoint: str = "npr-a", device: str = "cpu",
                 repo: str = DEFAULT_REPO, variant: str = "npr"):
        self.checkpoint_id = checkpoint
        self.device = device
        self.repo = repo
        self._record = entry(SOURCE_NAME, variant=variant)
        self._checkpoint = self._find_checkpoint(checkpoint)
        self._model = None
        self._load_notes: dict = {}
        self.last_call: dict = {}
        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    # ------------------------------------------------------------------

    def _find_checkpoint(self, checkpoint: str) -> dict:
        for item in self._record["weights"]["checkpoints"]:
            if item["id"] == checkpoint:
                return item
        known = [item["id"] for item in self._record["weights"]["checkpoints"]]
        raise ValueError(f"登记表里没有 checkpoint {checkpoint!r}；只有 {known}")

    @property
    def name(self) -> str:
        return f"NPR({self.checkpoint_id})"

    def identity(self) -> dict:
        """Everything a later reader needs to reproduce this number."""
        return {
            "expert": SOURCE_NAME,
            "checkpoint_id": self.checkpoint_id,
            "weights_file": self._checkpoint["file"],
            "weights_bytes": self._checkpoint["bytes"],
            "weights_sha256": self._checkpoint["sha256"],
            "load_as_recorded": self._checkpoint["load_as"],
            "load_as_used": self._load_notes.get("load_as_used"),
            "load_fallback_used": self._load_notes.get("fallback", False),
            "repo": self.repo,
            "repo_commit": _repo_commit(self.repo),
            "code_file": "networks/resnet.py",
            "preprocess": self._record["preprocess"]["name"],
            "preprocess_detail": self._record["preprocess"]["detail"],
            "measurement_scope": self._record["preprocess"]["measurement_scope"],
            "licence": self._record["weights"]["licence"],
        }

    # ------------------------------------------------------------------

    def load(self) -> "NPREvaluator":
        if self._model is not None:
            return self
        resnet = _import_resnet(self.repo)
        model = resnet.resnet50(num_classes=1)

        blob = torch.load(self._checkpoint["file"], map_location="cpu",
                          weights_only=False)
        target = dict(model.state_dict())
        order = [self._checkpoint["load_as"]] + [
            how for how in LOADERS if how != self._checkpoint["load_as"]]
        errors = []
        for how in order:
            state = LOADERS[how](blob)
            if state is None:
                errors.append(f"{how}: 顶层没有该键（实际键：{list(blob)[:5] if isinstance(blob, dict) else type(blob).__name__}）")
                continue
            stripped = {key.replace("module.", "", 1): value for key, value in state.items()}
            missing = [key for key in target if key not in stripped]
            extra = [key for key in stripped if key not in target]
            wrong = [key for key in stripped
                     if key in target and tuple(stripped[key].shape) != tuple(target[key].shape)]
            if missing or extra or wrong:
                errors.append(f"{how}: 缺 {len(missing)} / 多 {len(extra)} / 形状不符 {len(wrong)}")
                continue
            model.load_state_dict(stripped, strict=True)
            self._load_notes = {"load_as_used": how,
                                "fallback": how != self._checkpoint["load_as"]}
            break
        else:
            raise RuntimeError(
                f"{self.checkpoint_id} 的权重两种写法都加载不了——不要静默换模型，"
                f"先核对文件：\n  " + "\n  ".join(errors))
        if self._load_notes.get("fallback"):
            print(f"[NPR] ⚠ 登记表写的加载方式（{self._checkpoint['load_as']}）不适用本文件，"
                  f"实际用了 {self._load_notes['load_as_used']}——登记表需要更正。")

        model.eval().to(self.device)
        self._model = model
        return self

    # ------------------------------------------------------------------

    def score(self, image: Image.Image | str) -> float:
        """
        The official `model(x).sigmoid()`, whole image, no resize/crop.

        One exception, because the official README prescribes it for exactly this
        recipe: the model computes `x - interpolate(x, 0.5)`, and an **odd**
        width or height makes the two operands differ by one pixel
        (`327 vs 326`).  The README's AIGCDetectBenchmark note says to drop the
        last row/column in that case, so that is what happens here — and it is
        recorded (`odd_size_cropped`) rather than hidden, because it changes the
        input by one pixel.
        """
        self.load()
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        tensor = self._transform(image)                 # (3, H, W)
        cropped = False
        _, height, width = tensor.shape
        if height % 2 == 1 or width % 2 == 1:
            tensor = tensor[:, :height - height % 2, :width - width % 2]
            cropped = True
        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self._model(tensor)
        value = float(torch.sigmoid(logit).flatten()[0].item())
        self.last_call = {"odd_size_cropped": cropped,
                          "input_hw": [int(tensor.shape[-2]), int(tensor.shape[-1])]}
        return value


def _repo_commit(repo: str) -> Optional[str]:
    head = os.path.join(repo, ".git", "HEAD")
    try:
        with open(head, encoding="utf-8") as handle:
            ref = handle.read().strip()
        if ref.startswith("ref: "):
            with open(os.path.join(repo, ".git", ref[5:]), encoding="utf-8") as handle:
                return handle.read().strip()
        return ref
    except OSError:
        return None
