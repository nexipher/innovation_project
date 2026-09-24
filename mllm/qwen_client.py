"""
Qwen2.5-VL-7B-Instruct client for the Active Forensic Agent system.

Loads the Qwen2.5-VL model (FP16 on GPU) and provides the generate() interface
conforming to BaseMLLMClient.  Includes a format-correction feedback loop that
re-prompts the model if XML tags are missing or malformed.
"""

import json
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


class QwenVLClient(BaseMLLMClient):
    """
    Real Qwen2.5-VL-7B-Instruct client.

    Loads the model in FP16 on GPU.  Implements the BaseMLLMClient interface
    so it is a drop-in replacement for MockMLLMClient in the state machine.
    """

    def __init__(self, max_retries: int = 2):
        self._max_retries = max_retries
        self._processor = None
        self._model = None
        self._loaded = False
        self._retry_count = 0  # per-session retry counter

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
        self._model.eval()

        elapsed = time.time() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f"[QwenVLClient] Loaded in {elapsed:.1f}s, VRAM: {vram:.1f} GB")
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Build Qwen-format messages from our conversation history
        messages = self._build_messages(image_path, history)

        # Generate with retry loop for format correction
        for attempt in range(self._max_retries + 1):
            raw_output = self._inference(messages)

            # Validate format
            valid, error_msg = Parser.validate_tag_structure(raw_output)
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

    def reset(self) -> None:
        self._retry_count = 0

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
        return build_messages(image_path, history)

    def _inference(self, messages: List[dict]) -> str:
        """Run a single forward pass and decode the output."""
        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Collect images from messages for processor
        images = self._collect_images(messages)

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
