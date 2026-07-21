"""
Regex-based parser for extracting structured forensic tags from MLLM output.

Handles the following XML-like tags:
  <planning>...</planning>
  <call_freq>[ymin, xmin, ymax, xmax]</call_freq>
  <call_noise>[ymin, xmin, ymax, xmax]</call_noise>
  <call_jpeg>[ymin, xmin, ymax, xmax]</call_jpeg>
  <reasoning>...</reasoning>
  <verdict>{"verdict": "...", "confidence": ..., ...}</verdict>
"""

import json
import re
from typing import Dict, List, Optional, Tuple

from config import NORMALIZATION_SCALE


class Parser:
    """Stateless regex-based parser for MLLM forensic output tags."""

    # ------------------------------------------------------------------
    # Regex patterns (compiled once at class level)
    # ------------------------------------------------------------------

    _RE_PLANNING = re.compile(
        r"<planning>\s*(.*?)\s*</planning>",
        re.DOTALL | re.IGNORECASE,
    )

    _RE_REASONING = re.compile(
        r"<reasoning>\s*(.*?)\s*</reasoning>",
        re.DOTALL | re.IGNORECASE,
    )

    _RE_VERDICT = re.compile(
        r"<verdict>\s*(.*?)\s*</verdict>",
        re.DOTALL | re.IGNORECASE,
    )

    # Generic call tag: captures expert name and bbox
    _RE_CALL = re.compile(
        r"<call_(\w+)>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_\1>",
        re.DOTALL | re.IGNORECASE,
    )

    # Specific call tag patterns for extraction by name
    _RE_CALL_FREQ = re.compile(
        r"<call_freq>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_freq>",
        re.DOTALL | re.IGNORECASE,
    )

    _RE_CALL_NOISE = re.compile(
        r"<call_noise>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_noise>",
        re.DOTALL | re.IGNORECASE,
    )

    _RE_CALL_JPEG = re.compile(
        r"<call_jpeg>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_jpeg>",
        re.DOTALL | re.IGNORECASE,
    )

    _CALL_PATTERNS = {
        "freq": _RE_CALL_FREQ,
        "noise": _RE_CALL_NOISE,
        "jpeg": _RE_CALL_JPEG,
    }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def parse_planning(cls, text: str) -> Optional[str]:
        """Extract the content of the <planning> tag, or None."""
        m = cls._RE_PLANNING.search(text)
        return m.group(1).strip() if m else None

    @classmethod
    def parse_reasoning(cls, text: str) -> Optional[str]:
        """Extract the content of the <reasoning> tag, or None."""
        m = cls._RE_REASONING.search(text)
        return m.group(1).strip() if m else None

    @classmethod
    def parse_verdict(cls, text: str) -> Optional[dict]:
        """
        Extract and parse the JSON inside the <verdict> tag.
        Returns a dict or None if no valid verdict found.
        """
        m = cls._RE_VERDICT.search(text)
        if not m:
            return None
        json_str = m.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to repair common issues: single quotes, trailing commas
            try:
                repaired = json_str.replace("'", '"')
                return json.loads(repaired)
            except json.JSONDecodeError:
                return None

    @classmethod
    def parse_call(cls, expert_name: str, text: str) -> List[List[int]]:
        """
        Extract all bbox coordinates from <call_{expert_name}> tags.

        Args:
            expert_name: One of 'freq', 'noise', 'jpeg'.
            text: Raw MLLM output string.

        Returns:
            List of bboxes, each as [ymin, xmin, ymax, xmax] (integers, 0-1000 range).
        """
        pattern = cls._CALL_PATTERNS.get(expert_name)
        if pattern is None:
            return []
        results = []
        for m in pattern.finditer(text):
            bbox = [int(m.group(i)) for i in range(1, 5)]
            results.append(bbox)
        return results

    # ------------------------------------------------------------------
    # Tag normalisation — maps common LLM mistakes to correct tag names
    # ------------------------------------------------------------------
    _TAG_NORMALIZE = {
        # <call_call_*> patterns — LLM doubles the "call_" prefix
        "call_call_freq": "freq",
        "call_call_noise": "noise",
        "call_call_jpeg": "jpeg",
        "call_freq": "freq",
        "call_noise": "noise",
        "call_jpeg": "jpeg",
        # Other common LLM variations
        "call_frequency": "freq",
        "call_noise_residual": "noise",
        "call_jpeg_compression": "jpeg",
    }

    @classmethod
    def _normalize_expert_name(cls, name: str) -> str:
        """Normalise common LLM hallucinated expert names to canonical names."""
        return cls._TAG_NORMALIZE.get(name.lower(), name.lower())

    @classmethod
    def extract_all_calls(cls, text: str) -> List[Tuple[str, List[int]]]:
        """
        Extract ALL expert calls (any type) from the text.
        Uses the generic _RE_CALL pattern + tag normalisation.

        Returns:
            List of (expert_name, bbox) tuples.
        """
        results = []
        for m in cls._RE_CALL.finditer(text):
            raw_name = m.group(1)
            expert_name = cls._normalize_expert_name(raw_name)
            if expert_name in ("freq", "noise", "jpeg"):
                bbox = [int(m.group(i)) for i in range(2, 6)]
                results.append((expert_name, bbox))
        return results

    @classmethod
    def has_verdict(cls, text: str) -> bool:
        """Check if the text contains a <verdict> tag."""
        return bool(cls._RE_VERDICT.search(text))

    @classmethod
    def has_call(cls, text: str) -> bool:
        """Check if the text contains any <call_*> tag."""
        return bool(cls._RE_CALL.search(text))

    @classmethod
    def validate_tag_structure(cls, text: str) -> Tuple[bool, str]:
        """
        Validate that the output contains the minimum required tag structure.

        Returns:
            (is_valid, error_message). is_valid=False means the output
            is malformed and should trigger a format-correction feedback loop.
        """
        # Must have at least a planning block (first turn) OR reasoning (later turns)
        has_planning = cls.parse_planning(text) is not None
        has_reasoning = cls.parse_reasoning(text) is not None
        has_verdict = cls.has_verdict(text)
        has_call = cls.has_call(text)

        if not has_planning and not has_reasoning:
            return False, "缺少 <planning> 或 <reasoning> 标签，请按 SOP 格式输出。"
        if not has_verdict and not has_call:
            return False, "缺少 <verdict> 或 <call_*> 标签，请输出判定结论或调用法证专家。"
        return True, ""

    @classmethod
    def extract_bbox_from_text(cls, text: str) -> Optional[List[int]]:
        """
        Fallback: try to extract any bracketed 4-number tuple from text,
        even if XML tags are malformed.  Useful for error recovery.

        Returns:
            [ymin, xmin, ymax, xmax] or None.
        """
        # Look for patterns like [123, 456, 789, 1011] or [123,456,789,1011]
        m = re.search(
            r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]",
            text,
        )
        if m:
            bbox = [int(m.group(i)) for i in range(1, 5)]
            # Basic sanity: coords should be in [0, NORMALIZATION_SCALE]
            if all(0 <= v <= NORMALIZATION_SCALE for v in bbox):
                return bbox
            # If clearly pixel coords (some > 1000), return as-is for downstream clipping
            return bbox
        return None
