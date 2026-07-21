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
from config import QWEN_MODEL_PATH, SYSTEM_PROMPT, MAX_STEPS
from utils.parser import Parser


# ---------------------------------------------------------------------------
# Strengthened system prompt — forces XML-structured forensic output
# ---------------------------------------------------------------------------
FORENSIC_SYSTEM_PROMPT = """You are an AI forensic image analyst. You MUST follow this EXACT format in EVERY response. Do NOT write free-form analysis.

AVAILABLE ACTIONS (use EXACTLY these tag names — do NOT invent variations):
- <call_freq>[ymin, xmin, ymax, xmax]</call_freq>  → call frequency-domain expert
- <call_noise>[ymin, xmin, ymax, xmax]</call_noise> → call noise residual expert
- <call_jpeg>[ymin, xmin, ymax, xmax]</call_jpeg>   → call JPEG compression expert

FORBIDDEN: Do NOT use <call_call_freq>, <call_call_noise>, <call_frequency>, or any other variation.

COORDINATES: bbox values are integers in range [0, 1000], format [ymin, xmin, ymax, xmax].

RESPONSE FORMAT (MANDATORY — every response must contain one of these two structures):

Structure A — When you need forensic evidence:
<planning>
Suspected Region: [ymin, xmin, ymax, xmax]
Visual Anomalies: [describe what looks suspicious in this specific image]
Expert Target & Hypothesis: [which expert to call and why]
</planning>
<call_EXPERT_NAME>[ymin, xmin, ymax, xmax]</call_EXPERT_NAME>

Structure B — When you have enough evidence to conclude:
<reasoning>
[Cross-reference the expert's physical findings with your visual observations.
If different experts conflict, explain why and apply "presumption of innocence".
If the image has compression artifacts that may weaken certain signals, note it.]
</reasoning>
<verdict>
{"verdict": "Real"|"Fake"|"Uncertain", "confidence": 0.0-1.0, "primary_evidence": ["evidence_name1"], "report": "concise forensic report in Chinese"}
</verdict>

RULES:
1. For blurry/spliced edges or unnatural sharpening → call noise or freq first.
2. For overly smooth/regular textures → call freq first.
3. For low-res, blocky, or social-media-recompressed images → call jpeg first.
4. NEVER output only natural-language analysis without the required XML tags.
5. NEVER fabricate evidence — only reference evidence tokens you have received.
6. After receiving 2+ evidence tokens, you MUST produce a verdict.
"""


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
        """
        Convert our internal conversation history into Qwen2.5-VL
        chat-template-compatible messages.
        """
        messages = [
            {"role": "system", "content": FORENSIC_SYSTEM_PROMPT},
        ]

        # Load image once for the first user turn
        img = Image.open(image_path).convert("RGB")

        for turn in history:
            role = turn.get("from", "user")
            value = turn.get("value", "")

            if role == "user":
                # Check if this is the first user message (contains <image> marker)
                # or an evidence injection (JSON)
                if "<image>" in value:
                    # Initial prompt with image
                    text_content = value.replace("<image>\n", "").replace("<image>", "")
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": text_content},
                        ],
                    })
                else:
                    # Evidence Token injection or system message — text only
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": value}],
                    })
            else:
                # Assistant turn
                messages.append({
                    "role": "assistant",
                    "content": value,
                })

        # If history is empty, this is the first turn — add initial prompt
        if not history:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": (
                        "请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"
                        "首先输出 <planning> 标签，然后根据需要调用法证专家。"
                    )},
                ],
            })

        return messages

    # ------------------------------------------------------------------
    # Internal: inference
    # ------------------------------------------------------------------

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
        """Extract PIL Image objects from message content blocks."""
        images = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        img = block.get("image")
                        if isinstance(img, Image.Image):
                            images.append(img)
        return images
