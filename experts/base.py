"""
Base class and data types for forensic expert modules.

All experts share the same interface:
    analyze(img_np: np.ndarray, bbox: list[int]) -> ExpertResult

The caller (state machine) is responsible for cropping the bbox from the full image
before calling analyze().  The expert receives the pre-cropped patch.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from config import (
    STRENGTH_SUPPORT_MAP,
    STRENGTH_TEXT_MAP,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_THRESHOLD_LOW,
)


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

    # G2 §4.9 Evidence Bundle fields
    counter_explanation: str = ""   # alternative, benign explanation of the phenomenon
    condition_metadata: dict = field(default_factory=dict)
    reliability: float = 0.0        # empirical reliability for the current condition (filled by controller)
    reliability_note: str = ""


class BaseExpert(ABC):
    """
    Abstract base class for all forensic experts.

    Subclasses must implement analyze(), which receives a pre-cropped image
    patch (BGR uint8 numpy array) and returns an ExpertResult.
    """

    # Each subclass overrides this
    source_name: str = "base_expert"

    # G2 §4.9: benign explanations for the phenomenon this expert measures,
    # used to keep the MLLM from over-reading a single physical signal.
    counter_explanation: str = ""

    # ------------------------------------------------------------------
    # G2-b measured metric direction (plan.md §4.9 G2-e)
    #
    #   +1: a higher metric points to Fake — the expert's original theory;
    #   -1: a higher metric points to Real — a measured inversion.  G2-b
    #       found this for the noise and JPEG experts on the format-balanced
    #       calibration set, so their tokens must not reuse the aligned
    #       sentence map: the model reads the text, not the calibration table.
    # ------------------------------------------------------------------
    metric_polarity: int = 1

    # Interpretation sentences for polarised experts (band -> text), used
    # instead of STRENGTH_TEXT_MAP when metric_polarity is negative.
    inverted_text_map: Dict[str, str] = {}

    @classmethod
    def classify_metric(cls, strength: float) -> Tuple[str, str]:
        """
        Map a normalised strength to (support, interpretation text).

        Experts whose metric direction is inverted must override
        inverted_text_map, because the shared map asserts the generative
        reading the calibration set contradicts.
        """
        if strength < STRENGTH_THRESHOLD_LOW:
            band = "low"
        elif strength < STRENGTH_THRESHOLD_HIGH:
            band = "medium"
        else:
            band = "high"

        if cls.metric_polarity >= 0:
            return STRENGTH_SUPPORT_MAP[band], STRENGTH_TEXT_MAP[band]

        inverted_support = {"low": "AI-generated", "medium": "Uncertain",
                            "high": "Real"}
        return inverted_support[band], cls.inverted_text_map[band]

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

    def render_artifacts(self, img_patch: np.ndarray) -> dict:
        """
        Render named visual artifacts (BGR uint8 images) for this analysis.

        G2 §4.9: artifacts are attached to the MLLM conversation alongside
        the textual evidence so the model can inspect the machine's view.
        The default implementation renders nothing.
        """
        return {}

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
            counter_explanation=self.counter_explanation,
        )
