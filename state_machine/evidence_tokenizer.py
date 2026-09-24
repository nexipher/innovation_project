"""
Evidence Tokenizer — converts raw expert output into structured
Evidence Token JSON ready for injection into the MLLM conversation.

Per the specification (§4.1-4.2):
  - Maps scalar strength to semantic "soft description" text.
  - Assembles the full Evidence Token Schema dict.
  - Determines support label (Real / AI-generated / Uncertain).

G1 additions (plan.md §4.8):
  - Explicit coordinate spaces: `region_pixels` + `region_normalized_1000`
    + `coordinate_space`, so pixel coordinates can never be re-read as
    normalized coordinates (and vice versa);
  - Stable `evidence_id` for deduplication and evidence-bound reporting;
  - `region_semantics` labels bboxes as diagnostic evidence regions.
"""

import hashlib
import json
from typing import List, Optional

from config import (
    NORMALIZATION_SCALE,
    REGION_SEMANTICS_DIAGNOSTIC,
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class EvidenceTokenizer:
    """Stateless converter: ExpertResult → Evidence Token dict."""

    @classmethod
    def evidence_id(
        cls,
        source: str,
        region_pixels: List[int],
        strength: float,
        evidence_name: str,
    ) -> str:
        """
        Deterministic evidence identifier.

        Same expert + same region + same result digest → same id, which makes
        duplicate suppression idempotent across runs.
        """
        payload = "|".join([
            str(source),
            ",".join(str(int(v)) for v in region_pixels),
            f"{round(float(strength), 4):.4f}",
            str(evidence_name),
        ])
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]
        return f"E-{digest}"

    @classmethod
    def tokenize(
        cls,
        expert_result,  # ExpertResult
        bbox: List[int],
        image_shape: tuple,
        region_normalized: Optional[List[int]] = None,
    ) -> dict:
        """
        Build a complete Evidence Token dict from an ExpertResult.

        Args:
            expert_result: ExpertResult dataclass from an expert's analyze().
            bbox: Absolute pixel bbox [ymin, xmin, ymax, xmax] that was analysed.
            image_shape: (height, width) of the full image.
            region_normalized: The model's requested bbox in [0, 1000] space,
                recorded for audit; None when the call did not originate from
                a normalized-space request.

        Returns:
            Evidence Token dict matching the project schema.
        """
        region_str = f"patch_coordinates_{bbox}"
        region_pixels = [int(v) for v in bbox]

        return {
            "evidence_id": cls.evidence_id(
                expert_result.source, region_pixels,
                expert_result.strength, expert_result.evidence_name,
            ),
            "evidence_name": expert_result.evidence_name,
            "region": region_str,
            "region_pixels": region_pixels,
            "region_normalized_1000": (
                [int(v) for v in region_normalized]
                if region_normalized is not None else None
            ),
            "coordinate_space": "pixels",
            "region_semantics": REGION_SEMANTICS_DIAGNOSTIC,
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
