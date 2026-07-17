"""
Evidence Tokenizer — converts raw expert output into structured
Evidence Token JSON ready for injection into the MLLM conversation.

Per the specification (§4.1-4.2):
  - Maps scalar strength to semantic "soft description" text.
  - Assembles the full Evidence Token Schema dict.
  - Determines support label (Real / AI-generated / Uncertain).
"""

import json
from typing import List

from config import (
    NORMALIZATION_SCALE,
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class EvidenceTokenizer:
    """Stateless converter: ExpertResult → Evidence Token dict."""

    @classmethod
    def tokenize(
        cls,
        expert_result,  # ExpertResult
        bbox: List[int],
        image_shape: tuple,
    ) -> dict:
        """
        Build a complete Evidence Token dict from an ExpertResult.

        Args:
            expert_result: ExpertResult dataclass from an expert's analyze().
            bbox: Absolute pixel bbox [ymin, xmin, ymax, xmax] that was analysed.
            image_shape: (height, width) of the full image.

        Returns:
            Evidence Token dict matching the project schema.
        """
        region_str = f"patch_coordinates_{bbox}"

        return {
            "evidence_name": expert_result.evidence_name,
            "region": region_str,
            "phenomenon": expert_result.phenomenon,
            "reasoning": expert_result.reasoning,
            "strength": round(expert_result.strength, 4),
            "source": expert_result.source,
            "support": expert_result.support,
            "interpretation_text": expert_result.interpretation_text,
        }

    @classmethod
    def strength_to_text(cls, strength: float) -> str:
        """
        Map a normalised strength value to its semantic description.

        0.0 ≤ s < 0.3 → normal hardware camera capture
        0.3 ≤ s < 0.7 → mild distortions
        0.7 ≤ s ≤ 1.0 → severe anomaly matching AI fingerprints
        """
        if strength < STRENGTH_THRESHOLD_LOW:
            return STRENGTH_TEXT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return STRENGTH_TEXT_MAP["medium"]
        else:
            return STRENGTH_TEXT_MAP["high"]

    @classmethod
    def strength_to_support(cls, strength: float) -> str:
        """Map strength → support label."""
        if strength < STRENGTH_THRESHOLD_LOW:
            return STRENGTH_SUPPORT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return STRENGTH_SUPPORT_MAP["medium"]
        else:
            return STRENGTH_SUPPORT_MAP["high"]

    @classmethod
    def to_json(cls, token: dict) -> str:
        """Serialize an Evidence Token dict to a JSON string for MLLM injection."""
        return json.dumps(token, ensure_ascii=False)
