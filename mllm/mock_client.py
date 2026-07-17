"""
Template-driven Mock MLLM client for CPU-only development and testing.

Replaces a real 7B+ parameter vision-language model with deterministic,
mode-driven template responses.  Produces correctly-formatted XML forensic
tags that the Parser and StateMachine can process end-to-end.

Behaviour Modes:
  fast_verdict  — minimal calls, quick conclusion (tests halting-by-verdict).
  two_calls     — balanced 1-2 expert calls before verdict (default).
  explore_all   — max out expert calls; tests max-steps halting.
  conflict      — deliberately creates contradictory evidence → Uncertain verdict.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import BaseMLLMClient
from config import NORMALIZATION_SCALE, MOCK_MLLM_SEED


class MockMLLMClient(BaseMLLMClient):
    """
    Template-based mock that simulates Qwen2.5-VL behaviour.

    Does NOT load any model weights.  Generates XML responses from a
    curated template library, parameterised by image source (real/fake),
    turn number, and selected behaviour mode.
    """

    # All valid modes
    MODES = ("fast_verdict", "two_calls", "explore_all", "conflict")

    def __init__(self, mode: str = "two_calls", seed: int = MOCK_MLLM_SEED):
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Choose from {self.MODES}")
        self._mode = mode
        self._rng = np.random.default_rng(seed)
        self._turn_count = 0
        self._image_label: Optional[str] = None  # "Real" | "Fake" | None
        self._source_model: str = "Unknown"
        self._image_shape: Tuple[int, int] = (0, 0)
        self._called_experts: List[str] = []      # track which experts were called
        self._last_evidence_strength: float = 0.5

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        image_path: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Produce the next mock MLLM response based on conversation state.

        On first call (empty history or history with only system prompt),
        inspects the image path to determine real vs fake and initialises
        internal state.
        """
        # Detect image label on first call
        if self._image_label is None:
            self._detect_label(image_path)

        # Count how many assistant turns have already happened
        assistant_turns = sum(1 for h in history if h.get("from") == "gpt")
        self._turn_count = assistant_turns

        # Check what's in the latest user message (may contain evidence)
        last_evidence = self._extract_last_evidence(history)

        response = self._route_response(last_evidence)

        self._turn_count += 1
        return response

    def reset(self) -> None:
        self._turn_count = 0
        self._image_label = None
        self._source_model = "Unknown"
        self._image_shape = (0, 0)
        self._called_experts = []
        self._last_evidence_strength = 0.5

    @property
    def name(self) -> str:
        return f"MockMLLMClient({self._mode})"

    @property
    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------
    # Internal: label detection
    # ------------------------------------------------------------------

    def _detect_label(self, image_path: str) -> None:
        """Infer ground-truth label from directory structure."""
        norm = image_path.replace("\\", "/")
        if "/Real/" in norm or "dataset/Real" in norm:
            self._image_label = "Real"
            self._source_model = "Real"
        else:
            self._image_label = "Fake"
            # Try to identify the generator model
            for model in ("ADM", "BigGAN", "Glide", "Midjourney",
                          "SD14", "SD15", "VQDM", "Wukong"):
                if f"/{model}/" in norm:
                    self._source_model = model
                    break

        # Infer image shape from file (lightweight — just read header)
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                self._image_shape = (img.height, img.width)
        except Exception:
            self._image_shape = (512, 512)  # fallback

    def _extract_last_evidence(self, history: List[Dict]) -> Optional[dict]:
        """Extract the most recent Evidence Token JSON from conversation history."""
        for h in reversed(history):
            if h.get("from") != "user":
                continue
            val = h.get("value", "")
            if '"evidence_name"' in val and '"strength"' in val:
                try:
                    evidence = json.loads(val)
                    self._last_evidence_strength = evidence.get("strength", 0.5)
                    return evidence
                except json.JSONDecodeError:
                    pass
        return None

    # ------------------------------------------------------------------
    # Internal: routing
    # ------------------------------------------------------------------

    def _route_response(self, evidence: Optional[dict]) -> str:
        """Choose the appropriate template based on mode and turn."""

        if self._mode == "fast_verdict":
            return self._handle_fast_verdict(evidence)
        elif self._mode == "two_calls":
            return self._handle_two_calls(evidence)
        elif self._mode == "explore_all":
            return self._handle_explore_all(evidence)
        elif self._mode == "conflict":
            return self._handle_conflict(evidence)
        return self._handle_two_calls(evidence)  # fallback

    # ------------------------------------------------------------------
    # Mode handlers
    # ------------------------------------------------------------------

    def _handle_fast_verdict(self, evidence: Optional[dict]) -> str:
        """Minimal calling — tests quick verdict path."""
        if evidence is None:
            # Turn 0: planning + call
            if self._image_label == "Real":
                return self._planning("blurred edge, possible smoothing") + self._call("jpeg")
            else:
                return self._planning("overly smooth textures, lack of natural noise") + self._call("freq")
        else:
            # Turn 1: reasoning + verdict
            return self._reasoning(evidence) + self._verdict()

    def _handle_two_calls(self, evidence: Optional[dict]) -> str:
        """Balanced 1-2 expert calls — default mode."""
        n_calls = len(self._called_experts)

        if evidence is None:
            # Turn 0: first call
            if self._image_label == "Real":
                return self._planning("mild compression artifacts, possible re-save") + self._call("jpeg")
            else:
                return self._planning("unnaturally smooth textures, suspicious edges") + self._call("freq")

        elif n_calls < 2 and self._should_call_another(evidence):
            # Turn 1: reason over evidence, then call second expert
            reasoning = self._reasoning(evidence)
            # Pick an expert not yet called
            remaining = [e for e in ("freq", "noise", "jpeg")
                         if e not in self._called_experts]
            next_expert = remaining[0] if remaining else "noise"
            return reasoning + self._call(next_expert)
        else:
            # Final turn: reasoning + verdict
            return self._reasoning(evidence) + self._verdict()

    def _handle_explore_all(self, evidence: Optional[dict]) -> str:
        """Max out expert calls — tests max-steps halting."""
        n_calls = len(self._called_experts)

        if evidence is None:
            return self._planning("comprehensive multi-region scan needed") + self._call("freq")
        else:
            reasoning = self._reasoning(evidence)
            # Keep calling while experts remain
            remaining = [e for e in ("freq", "noise", "jpeg")
                         if e not in self._called_experts]
            if remaining:
                # Cycle through experts
                next_expert = remaining[0]
                return reasoning + self._call(next_expert)
            else:
                # All called — wrap up
                return reasoning + self._verdict()

    def _handle_conflict(self, evidence: Optional[dict]) -> str:
        """Create contradictory evidence → Uncertain verdict."""
        n_calls = len(self._called_experts)

        if evidence is None:
            return self._planning("suspicious frequency patterns detected") + self._call("freq")
        elif n_calls == 1:
            # After freq, call noise (which will likely give opposite result)
            return self._reasoning(evidence) + self._call("noise")
        else:
            # Conflict reflection + uncertain verdict
            return self._conflict_reasoning() + self._uncertain_verdict()

    # ------------------------------------------------------------------
    # Template generators
    # ------------------------------------------------------------------

    def _planning(self, visual_anomaly: str) -> str:
        """Generate a <planning> block with a plausible bbox."""
        bbox = self._gen_bbox_str()
        return (
            f"<planning>\n"
            f"Suspected Region: {bbox}\n"
            f"Visual Anomalies: {visual_anomaly}\n"
            f"Expert Target & Hypothesis: 根据视觉异常特征，需要调用法证专家进行微观物理痕迹分析。\n"
            f"</planning>\n"
        )

    def _call(self, expert: str) -> str:
        """Generate a <call_*> tag and record the call."""
        self._called_experts.append(expert)
        bbox = self._gen_bbox_str()
        return f"<call_{expert}>{bbox}</call_{expert}>\n"

    def _reasoning(self, evidence: Optional[dict]) -> str:
        """Generate a <reasoning> block that cross-references the evidence."""
        if evidence is None:
            return (
                "<reasoning>\n"
                "【物理-语义一致性校验】根据已有的视觉观察，需结合法证专家的微观物理信号进行交叉验证。\n"
                "</reasoning>\n"
            )

        src = evidence.get("source", "").replace("_expert", "")
        strength = evidence.get("strength", 0.5)
        phenomenon = evidence.get("phenomenon", "")
        support = evidence.get("support", "")

        if strength > 0.7:
            level = "强异常"
        elif strength > 0.3:
            level = "中等异常"
        else:
            level = "低异常"

        return (
            "<reasoning>\n"
            f"【物理-语义一致性校验】底层 {src} 专家反馈{level}（strength={strength:.2f}，判定: {support}）。\n"
            f"物理现象: {phenomenon[:120]}。\n"
            f"这与视觉层的观察在因果链上{'吻合' if strength > 0.5 else '不完全吻合，需进一步分析'}。\n"
            f"已考虑图像后处理（如社交媒体压缩）可能对物理指纹造成的衰减影响。\n"
            "</reasoning>\n"
        )

    def _conflict_reasoning(self) -> str:
        """Reasoning block for conflicting evidence."""
        return (
            "<reasoning>\n"
            "【证据冲突反思】不同法证专家给出了互相矛盾的物理信号：\n"
            "频域专家判定存在生成模型的网格伪迹（强判假），但噪声残差分析表明微观噪声分布符合真实相机特征（强判真）。\n"
            "这种底层物理痕迹的强冲突表明图像可能经过了精细的后处理，或者两种线索分别指向不同的来源。\n"
            "根据'疑罪从无'原则，在当前证据冲突且无法调和的情况下，不能做出确定结论。\n"
            "</reasoning>\n"
        )

    def _verdict(self) -> str:
        """Generate a <verdict> with Fake or Real conclusion."""
        if self._image_label == "Fake":
            verdict = "Fake"
            confidence = self._rng.uniform(0.82, 0.95)
            report = (
                "经多轮法证分析，图像在频域和噪声层面展现出典型的AI生成特征。"
                "微观物理痕迹与视觉层面的异常在因果链上高度吻合，"
                "综合判定为伪造图像。"
            )
        else:
            verdict = "Real"
            confidence = self._rng.uniform(0.78, 0.92)
            report = (
                "经法证分析，图像的微观物理痕迹（噪声分布、压缩特征）"
                "与真实相机成像过程一致。未发现AI生成或恶意篡改的明确证据，"
                "综合判定为真实照片。"
            )

        verdict_json = json.dumps({
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "primary_evidence": self._called_experts,
            "report": report,
        }, ensure_ascii=False)

        return f"<verdict>\n{verdict_json}\n</verdict>\n"

    def _uncertain_verdict(self) -> str:
        """Generate an Uncertain verdict for conflict mode."""
        verdict_json = json.dumps({
            "verdict": "Uncertain",
            "confidence": 0.45,
            "primary_evidence": self._called_experts,
            "report": (
                "多项法证证据存在根本性冲突：频域分析提示AI生成特征，"
                "但噪声残差分析支持真实相机来源。疑罪从无，判定为不确定，"
                "建议人工复核或使用更高精度的分析方法。"
            ),
        }, ensure_ascii=False)

        return f"<verdict>\n{verdict_json}\n</verdict>\n"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _gen_bbox_str(self) -> str:
        """
        Generate a deterministic bbox string based on image dimensions and turn.
        BBox uses [ymin, xmin, ymax, xmax] in [0, NORMALIZATION_SCALE].
        """
        h, w = self._image_shape
        if h == 0 or w == 0:
            h = w = 512

        # Use deterministic offsets based on turn and called experts
        # to simulate focusing on different regions
        t = self._turn_count
        n = len(self._called_experts)

        # Different regions for different experts
        regions = [
            (0.15, 0.15, 0.85, 0.85),   # center 70%
            (0.05, 0.05, 0.45, 0.45),   # top-left
            (0.55, 0.55, 0.95, 0.95),   # bottom-right
            (0.10, 0.50, 0.50, 0.90),   # mid-right
            (0.50, 0.10, 0.90, 0.50),   # mid-bottom
        ]
        ry1, rx1, ry2, rx2 = regions[n % len(regions)]

        ymin = int(ry1 * NORMALIZATION_SCALE)
        xmin = int(rx1 * NORMALIZATION_SCALE)
        ymax = int(ry2 * NORMALIZATION_SCALE)
        xmax = int(rx2 * NORMALIZATION_SCALE)

        return f"[{ymin}, {xmin}, {ymax}, {xmax}]"

    def _should_call_another(self, evidence: Optional[dict]) -> bool:
        """Decide whether to call another expert or proceed to verdict."""
        if evidence is None:
            return True
        strength = evidence.get("strength", 0.5)
        # If evidence is strong, we can conclude; otherwise dig deeper
        if strength > 0.8 or strength < 0.2:
            return False
        if len(self._called_experts) >= 2:
            return False
        return True
