"""
Qwen2.5-VL-7B-Instruct client for the Active Forensic Agent system.

Loads the Qwen2.5-VL model (FP16 on GPU) and provides the generate() interface
conforming to BaseMLLMClient.  Includes a format-correction feedback loop that
re-prompts the model if XML tags are missing or malformed.
"""

import json
import os
import re
import time
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from .base import BaseMLLMClient
from .message_builder import (
    FORENSIC_SYSTEM_PROMPT,
    build_messages,
    collect_images,
)
from config import QWEN_MODEL_PATH
from utils.parser import Parser


def with_image_marker(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Guarantee the `<image>` marker the message builder keys on."""
    if not history:
        return history
    first = history[0]
    if "<image>" in (first.get("value") or ""):
        return history
    return [{**first, "value": "<image>\n" + (first.get("value") or "")}] + list(history[1:])



def _sha256(value: Optional[str]) -> Optional[str]:
    """提示文本的哈希：与原文并存，用来在事后核对"这段文本没被改过"。"""
    if not value:
        return None
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

class QwenVLClient(BaseMLLMClient):
    """
    Real Qwen2.5-VL-7B-Instruct client.

    Loads the model in FP16 on GPU.  Implements the BaseMLLMClient interface
    so it is a drop-in replacement for MockMLLMClient in the state machine.
    """

    def __init__(self, max_retries: int = 2, system_prompt: str = None,
                 adapter_path: str = None):
        self._max_retries = max_retries
        self._processor = None
        self._model = None
        self._loaded = False
        self._retry_count = 0  # per-session retry counter
        # 每次模型调用的记录（阶段 4.6，用户第 7 条）：原文、格式校验结果、
        # 注入的纠正、以及**那次调用真正发出去的完整提示文本**。曾经这些都不留，
        # 于是 trace 里"一轮模型输出"可能其实是内部重试后的第三次结果，
        # 而中间那两次与纠正文本无从核对。逐会话累积，`reset()` 清空。
        self._call_log: List[dict] = []
        self._call_index = 0
        # 上一次 `_inference` 真正送进模型的文本与图像数（由子类填写）
        self.last_prompt_text: Optional[str] = None
        self.last_prompt_image_count = 0
        # G2-d: optional prompt variant (the no-tool baseline uses its own)
        self._system_prompt = system_prompt
        # G4-g: optional LoRA adapter, so the four-arm comparison can measure a
        # fine-tuned model with the same harness that measured the base one.
        self._adapter_path = adapter_path

    # ------------------------------------------------------------------
    # Lazy loading (model is heavy — load once, reuse across sessions)
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "QwenVLClient requires CUDA GPU. "
                "Use MockMLLMClient for CPU mode."
            )
        # 权重不在就**明说**。不查的话，`from_pretrained(不存在的本地路径)` 会把路径
        # 当成 hub 仓库名，报 "Repo id must be in the form 'namespace/repo_name'" ——
        # 一句和真实原因毫无关系的话，而且发生在第一次推理时（构造是懒加载）。
        # 阶段 4.6 的旧模型下架就是靠这一步暴露的。
        if not os.path.exists(QWEN_MODEL_PATH):
            raise RuntimeError(
                f"Qwen2.5-VL 权重不在：{QWEN_MODEL_PATH}\n"
                "这是**历史复算**用的旧模型（g8 主线用 Qwen3-VL，不受影响）。\n"
                "若它已被改名/删除：把目录改回原名，或把 config.QWEN_MODEL_PATH "
                "指向一份完整的 ModelScope 快照。")

        print("[QwenVLClient] Loading Qwen2.5-VL-7B-Instruct (FP16)...")
        t0 = time.time()

        self._processor = AutoProcessor.from_pretrained(
            QWEN_MODEL_PATH, trust_remote_code=True,
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            QWEN_MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self._adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, self._adapter_path)
            print(f"[QwenVLClient] LoRA adapter loaded: {self._adapter_path}")
        self._model.eval()

        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[QwenVLClient] Loaded in {elapsed:.1f}s, VRAM: {vram:.1f} GB")
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, image_path: str, history: List[Dict[str, str]]) -> str:
        """
        One plain call: no forensic protocol, no format-correction loop.

        For tooling that needs the model as a reader rather than as the agent —
        the G4-f visual-grounding screen asks it to compare a description with
        an image, and a retry loop that demands `<verdict>` tags would waste
        two calls per sample answering a question it was not asked.

        The primary image is attached whether or not the caller remembered the
        `<image>` marker: `build_messages` only attaches it when the marker is
        present, and a caller that does not know that — as the screen's first
        three rounds did not — gets a text-only call and an invented answer.
        """
        self._ensure_loaded()
        return self._inference(self._build_messages(image_path, with_image_marker(history)))

    @property
    def system_prompt(self):
        """The prompt this client builds messages with (G7 entry 4).

        Read fresh on every `generate`, so a batch run can serve several tool
        policies from one 16.7 GB model instead of either loading one model per
        session (slow) or keeping one per policy (out of memory).
        """
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value) -> None:
        self._system_prompt = value

    def generate(
        self,
        image_path: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Run Qwen2.5-VL inference with format-correction feedback loop.

        Args:
            image_path: Absolute path to the image under analysis.
            history: Conversation history (list of {"from": "...", "value": "..."}).

        Returns:
            Raw text output containing XML forensic tags.
        """
        self._ensure_loaded()
        self._retry_count = 0
        self._call_index += 1
        turn = self._call_index

        # Build Qwen-format messages from our conversation history
        messages = self._build_messages(image_path, history)

        # Generate with retry loop for format correction
        correction = None
        for attempt in range(self._max_retries + 1):
            raw_output = self._inference(messages)

            # Validate format
            valid, error_msg = Parser.validate_tag_structure(raw_output)
            self._call_log.append({
                "turn": turn,
                "attempt": attempt,
                "valid": bool(valid),
                "format_error": None if valid else error_msg,
                "raw_output": raw_output,
                # **这次调用真正发出去的东西**：完整提示文本 + 图像数 + 统计。
                "prompt_text": self.last_prompt_text,
                "prompt_sha256": _sha256(self.last_prompt_text),
                "images_attached": self.last_prompt_image_count,
                "input_stats": dict(self.last_input_stats or {}),
                "messages_len": len(messages),
                # 上一次尝试被拒后注入的纠正（第一次尝试为 None）
                "correction_from_previous_attempt": correction,
            })
            if valid:
                return raw_output

            # Format error — inject correction and retry
            if attempt < self._max_retries:
                self._retry_count += 1
                correction = (
                    f"[System: 输出格式错误 — {error_msg} "
                    f"请严格按照 SOP 格式重新输出。"
                    f"必须包含 <call_*> 或 <verdict> 标签。]"
                )
                # Append correction as a user message
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": correction}],
                })

        # All retries exhausted — return last output anyway
        return raw_output

    # ------------------------------------------------------------------
    # 会话级记录（阶段 4.6）
    # ------------------------------------------------------------------

    def call_log(self) -> List[dict]:
        """本次会话里每次模型调用的记录（逐次尝试，含被格式校验拒掉的）。"""
        return [dict(entry) for entry in self._call_log]

    def system_prompt_record(self) -> Optional[dict]:
        """
        初始 system prompt 的原文与哈希。

        "模型看了什么提示词"是根事实：以前只留 digest，等于只能证明"提示词变过"，
        不能证明"当时用的是哪一版"。原文照存（8 KB 量级）。
        """
        if not self._system_prompt:
            return None
        return {"text": self._system_prompt, "sha256": _sha256(self._system_prompt),
                "chars": len(self._system_prompt)}

    def reset(self) -> None:
        self._retry_count = 0
        self._call_log = []
        self._call_index = 0

    @property
    def name(self) -> str:
        return "Qwen2.5-VL-7B-Instruct"

    @property
    def mode(self) -> str:
        return "qwen_real"

    @property
    def retry_count(self) -> int:
        return self._retry_count

    # ------------------------------------------------------------------
    # Internal: message building
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        image_path: str,
        history: List[Dict[str, str]],
    ) -> List[dict]:
        """Delegate to the shared, CPU-testable message builder (G1)."""
        if self._system_prompt:
            return build_messages(image_path, history, system_prompt=self._system_prompt)
        return build_messages(image_path, history)

    def _inference(self, messages: List[dict]) -> str:
        """Run a single forward pass and decode the output."""
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        # 阶段 4.6：这次调用到底发了什么——原文由 generate() 记进调用日志。
        self.last_prompt_text = text

        # Collect images from messages for processor
        images = self._collect_images(messages)
        self.last_prompt_image_count = len(images)

        inputs = self._processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens
        output = self._processor.decode(
            generated[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return output.strip()

    def _collect_images(self, messages: List[dict]) -> List[Image.Image]:
        """Delegate to the shared image collector (G1)."""
        return collect_images(messages)
