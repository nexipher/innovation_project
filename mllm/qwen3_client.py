"""
Qwen3-VL-8B-Instruct client (the g8 generation's judge).

Same protocol as `QwenVLClient` — the message builder, the format-correction
loop and `ask()` are inherited unchanged, because the *protocol* is what the
controller depends on and it must not change with the model.  What differs is
the model class, the processor, and the fact that the weights are pinned:

    model:    Qwen/Qwen3-VL-8B-Instruct
    revision: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b   (2025-10-15)
    licence:  apache-2.0 (stated by the model card)

Two things this client refuses to do quietly:

  * **claim an inference mode it could not set.**  Thinking / non-thinking is a
    chat-template switch; if the template has no such switch, asking for
    "thinking" raises instead of silently running non-thinking and recording
    the wrong mode in the batch's `VERSIONS.json`.
  * **report weights without saying how it knows.**  The revision pin is the
    content identity (the hub resolves it to the exact bytes); the per-shard
    sizes and the index digest are recorded for free at load time, and a full
    digest is available on demand rather than assumed.
"""

from __future__ import annotations

import hashlib
import os
import time
from typing import List, Optional

import torch

from .qwen_client import QwenVLClient

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
# Pinned by the user on 2026-10-15.  A revision is a content address: the hub
# serves exactly this commit's files, so "which weights produced this batch?"
# has an answer that does not depend on the model's name.
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
MODEL_LICENCE = "apache-2.0"

INFERENCE_MODES = ("thinking", "non-thinking")


class Qwen3VLClient(QwenVLClient):
    """Qwen3-VL-8B-Instruct, loaded by repo id + pinned revision."""

    def __init__(self,
                 model_id: str = MODEL_ID,
                 revision: str = MODEL_REVISION,
                 inference_mode: str = "non-thinking",
                 dtype: str = "bfloat16",
                 max_retries: int = 2,
                 system_prompt: Optional[str] = None,
                 adapter_path: Optional[str] = None,
                 temperature: float = 0.1,
                 do_sample: bool = True,
                 max_new_tokens: int = 1024,
                 local_files_only: bool = False):
        super().__init__(max_retries=max_retries, system_prompt=system_prompt,
                         adapter_path=adapter_path)
        if inference_mode not in INFERENCE_MODES:
            raise ValueError(
                f"inference_mode 必须是 {INFERENCE_MODES} 之一，收到 {inference_mode!r}")
        self._model_id = model_id
        self._revision = revision
        self._inference_mode = inference_mode
        self._dtype_name = dtype
        self._temperature = temperature
        self._do_sample = do_sample
        self._max_new_tokens = max_new_tokens
        self._local_files_only = local_files_only
        self._supports_thinking = None      # decided from the real template
        self._loaded_identity: Optional[dict] = None
        # What the model actually received on the last call — the mechanical
        # form of "the picture entered the input" (a path in a message is not).
        self.last_input_stats: Optional[dict] = None

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def inference_mode(self) -> str:
        return self._inference_mode

    def template_supports_thinking(self) -> Optional[bool]:
        """Whether the loaded chat template has a thinking switch (None: not loaded)."""
        return self._supports_thinking

    def identity(self) -> dict:
        """
        What to write into a batch's `VERSIONS.json`.

        `weights_sha256` is deliberately absent until `hash_weights()` is asked
        for: a 17.5 GB digest costs real time, and a *missing* field is honest
        where a stale or partial one is not.
        """
        identity = {
            "name": self.name,
            "checkpoint": self._model_id,
            "revision": self._revision,
            "licence": MODEL_LICENCE,
            "dtype": self._dtype_name,
            "inference_mode": self._inference_mode,
            "thinking_supported_by_template": self._supports_thinking,
            "generation": {"temperature": self._temperature,
                           "do_sample": self._do_sample,
                           "max_new_tokens": self._max_new_tokens},
        }
        if self._loaded_identity:
            identity.update(self._loaded_identity)
        return identity

    def hash_weights(self) -> dict:
        """
        Full digest of the weight shards, streamed once.

        ~17.5 GB of reads; call it when the digest is actually being recorded
        (smoke report, batch VERSIONS), not on every session.
        """
        self._ensure_loaded()
        snap = self._snapshot_dir()
        digest = hashlib.sha256()
        files = []
        for name in sorted(os.listdir(snap)):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(snap, name)
            size = os.path.getsize(path)
            digest.update(name.encode())
            digest.update(str(size).encode())
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            files.append({"name": name, "bytes": size})
        return {"weights_sha256": digest.hexdigest()[:16],
                "weights_files": files,
                "weights_bytes": sum(f["bytes"] for f in files)}

    def _snapshot_dir(self) -> str:
        """Where the pinned revision actually landed on disk."""
        from huggingface_hub import snapshot_download

        return snapshot_download(self._model_id, revision=self._revision,
                                 local_files_only=True)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Qwen3VLClient requires CUDA.  CPU 协议验证请用 MockMLLMClient"
                "（scripts/g8_smoke.py --client mock）。")

        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

        print(f"[Qwen3VLClient] Loading {self._model_id}@{self._revision[:8]} "
              f"({self._dtype_name}, mode={self._inference_mode})...")
        t0 = time.time()

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[self._dtype_name]
        self._processor = Qwen3VLProcessor.from_pretrained(
            self._model_id, revision=self._revision,
            local_files_only=self._local_files_only)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            self._model_id, revision=self._revision,
            # `dtype=`, not `torch_dtype=`: transformers 5.x keeps the old name
            # only for backward compatibility and warns on every load.
            dtype=dtype, device_map="auto",
            local_files_only=self._local_files_only)
        if self._adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
        self._model.eval()

        # Does the template actually have the switch we are about to claim?
        template = (getattr(self._processor, "chat_template", None)
                    or getattr(self._processor.tokenizer, "chat_template", "") or "")
        self._supports_thinking = "enable_thinking" in template
        if self._inference_mode == "thinking" and not self._supports_thinking:
            raise RuntimeError(
                "inference_mode=thinking 但该模板没有 enable_thinking 开关："
                "要么换模板，要么改用 non-thinking——不记录一个没真正设上的模式。")

        snapshot = self._snapshot_dir()
        index = os.path.join(snapshot, "model.safetensors.index.json")
        index_digest = None
        if os.path.exists(index):
            with open(index, "rb") as handle:
                index_digest = hashlib.sha256(handle.read()).hexdigest()[:16]
        shards = sorted(name for name in os.listdir(snapshot)
                        if name.endswith(".safetensors"))
        self._loaded_identity = {
            "snapshot": snapshot,
            "shards": shards,
            "shard_bytes": sum(os.path.getsize(os.path.join(snapshot, name))
                               for name in shards),
            "index_sha256": index_digest,
            "local_files_only": self._local_files_only,
        }

        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[Qwen3VLClient] Loaded in {elapsed:.1f}s｜峰值显存 {vram:.2f} GB｜"
              f"thinking 模板支持：{self._supports_thinking}")
        self._loaded = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _inference(self, messages: List[dict]) -> str:
        template_kwargs = {}
        if self._supports_thinking:
            template_kwargs["enable_thinking"] = (self._inference_mode == "thinking")
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs)

        images = self._collect_images(messages)
        inputs = self._processor(text=[text],
                                 images=images if images else None,
                                 return_tensors="pt").to("cuda")
        input_ids = inputs["input_ids"]
        # Count the image placeholder tokens: if the processor dropped the
        # picture, this is 0 and the smoke says so instead of guessing.
        image_token_id = getattr(getattr(self._model, "config", None),
                                 "image_token_id", None)
        image_tokens = (int((input_ids == image_token_id).sum())
                        if image_token_id is not None else None)
        self.last_input_stats = {
            "input_tokens": int(input_ids.shape[1]),
            "image_tokens": image_tokens,
            "pil_images_attached": len(images),
        }
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                do_sample=self._do_sample,
                pad_token_id=getattr(self._processor.tokenizer, "pad_token_id", None),
            )
        return self._processor.decode(
            generated[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True).strip()

    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Qwen3-VL-8B-Instruct"

    @property
    def mode(self) -> str:
        return "qwen3_real"
