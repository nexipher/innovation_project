"""
Base class and data types for forensic expert modules.

All experts share the same interface:
    analyze(img_np: np.ndarray, bbox: list[int]) -> ExpertResult

The caller (state machine) is responsible for cropping the bbox from the full image
before calling analyze().  The expert receives the pre-cropped patch.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class ExpertResult:
    """
    Structured output from a forensic expert, mapping directly to the
    Evidence Token Schema defined in the project specification.

    Attributes:
        evidence_name:  Short identifier, e.g. "abnormal_high_frequency_residual".
        region:         Stringified bbox, e.g. "patch_coordinates_[210,150,480,420]".
        phenomenon:     Human-readable description of the physical phenomenon found.
        reasoning:      Physical / mathematical explanation of why this indicates
                        real or synthetic origin.
        strength:       Normalised anomaly score in [0, 1].  0 = normal,
                        1 = severe anomaly.
        source:         Which expert produced this: "frequency_expert" |
                        "noise_expert" | "jpeg_expert".
        support:        "AI-generated" | "Real" | "Uncertain".
        interpretation_text: Semantic soft-description from the strength mapping.
    """
    evidence_name: str
    region: str
    phenomenon: str
    reasoning: str
    strength: float
    source: str
    support: str
    interpretation_text: str

    # Extra fields that may be useful for debugging / GRPO reward computation
    raw_metric: float = 0.0
    metadata: dict = field(default_factory=dict)


class BaseExpert(ABC):
    """
    Abstract base class for all forensic experts.

    Subclasses must implement analyze(), which receives a pre-cropped image
    patch (BGR uint8 numpy array) and returns an ExpertResult.
    """

    # Each subclass overrides this
    source_name: str = "base_expert"

    @abstractmethod
    def analyze(self, img_patch: np.ndarray) -> ExpertResult:
        """
        Run forensic analysis on a pre-cropped image patch.

        Args:
            img_patch: BGR uint8 numpy array (H, W, 3) — already cropped to bbox.

        Returns:
            ExpertResult with strength ∈ [0, 1] and support label.
        """
        ...

    def _build_result(
        self,
        evidence_name: str,
        phenomenon: str,
        reasoning: str,
        strength: float,
        support: str,
        interpretation_text: str,
        bbox: List[int] = None,
        raw_metric: float = 0.0,
        **metadata,
    ) -> ExpertResult:
        """Convenience factory for building an ExpertResult."""
        region_str = (
            f"patch_coordinates_{bbox}" if bbox else "full_image"
        )
        return ExpertResult(
            evidence_name=evidence_name,
            region=region_str,
            phenomenon=phenomenon,
            reasoning=reasoning,
            strength=float(np.clip(strength, 0.0, 1.0)),
            source=self.source_name,
            support=support,
            interpretation_text=interpretation_text,
            raw_metric=raw_metric,
            metadata=metadata,
        )
