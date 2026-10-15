"""
Regex-based parser for extracting structured forensic tags from MLLM output.

Handles the following XML-like tags:
  <planning>...</planning>
  <call_probe>[ymin, xmin, ymax, xmax]</call_probe>
  <call_npr>[ymin, xmin, ymax, xmax]</call_npr>
  <call_univfd>[ymin, xmin, ymax, xmax]</call_univfd>
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
    # §4.17 R9: the bbox is optional.  Every expert measures the whole image,
    # so which region the model names is a diagnostic note, not a requirement
    # — and a model with nothing to point at must not be forced to invent a
    # region before it may consult the statistics.
    _RE_CALL_ANY = re.compile(
        r"<call_(\w+)>\s*"
        r"(?:\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?)?\s*"
        r"(?:</call_\w+>)?",
        re.I)

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

    _RE_CALL_PROBE = re.compile(
        r"<call_probe>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_probe>",
        re.DOTALL | re.IGNORECASE,
    )

    _RE_CALL_NPR = re.compile(
        r"<call_npr>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_npr>",
        re.DOTALL | re.IGNORECASE,
    )

    _RE_CALL_UNIVFD = re.compile(
        r"<call_univfd>\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?\s*</call_univfd>",
        re.DOTALL | re.IGNORECASE,
    )

    _CALL_PATTERNS = {
        # 当前工具箱（阶段 4.4）
        "probe": _RE_CALL_PROBE,
        "npr": _RE_CALL_NPR,
        "univfd": _RE_CALL_UNIVFD,
        "noise": _RE_CALL_NOISE,
        # 退役工具：旧 trace 的解析与 v2 复算仍要认
        "freq": _RE_CALL_FREQ,
        "jpeg": _RE_CALL_JPEG,
    }

    # The names `extract_all_calls` will hand to the controller.  A call tag
    # outside this set is dropped silently — which is why adding a tool means
    # adding it here, and why a test asserts this set covers every expert in
    # the score-semantics registry rather than trusting the absence of an error.
    CALLABLE_TOOLS = tuple(_CALL_PATTERNS)

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
        # G7 external detector: the model may reach for the paper's name, the
        # expert's name or the model family.  Keys are what the regex
        # *captures*: `<call_call_probe>` arrives as "call_probe" and
        # `<call_probe_dino>` as "probe_dino".
        "call_probe": "probe",
        "probe_dino": "probe",
        "dino": "probe",
        "aigi": "probe",
        "call_probe_dino": "probe",
    }

    @classmethod
    def _normalize_expert_name(cls, name: str) -> str:
        """Normalise common LLM hallucinated expert names to canonical names."""
        return cls._TAG_NORMALIZE.get(name.lower(), name.lower())

    @classmethod
    def extract_all_calls(cls, text: str) -> List[Tuple[str, Optional[List[int]]]]:
        """
        Extract ALL expert calls (any type) from the text.

        A call may carry a bbox or not: `bbox is None` means "measure the
        image as a whole, I have no region to point at", which is a legal
        request (R9).

        Returns:
            List of (expert_name, bbox_or_None) tuples.
        """
        results = []
        for m in cls._RE_CALL_ANY.finditer(text):
            raw_name = m.group(1)
            expert_name = cls._normalize_expert_name(raw_name)
            if expert_name not in cls.CALLABLE_TOOLS:
                continue
            if m.group(2) is None:
                results.append((expert_name, None))
            else:
                results.append((expert_name, [int(m.group(i)) for i in range(2, 6)]))
        return results

    # A call tag *inside* the planning block is the model naming the tool it
    # is about to consult ("Expert Target & Hypothesis: <call_probe>"), not a
    # second request.  The acceptance batch showed every one of 56 sessions
    # doing this — and the strict reading refused all 160 such turns as
    # "multiple calls", which measured the parser rather than the model.
    _RE_PLANNING_BLOCK = re.compile(r"<planning>.*?</planning>", re.S | re.I)
    _RE_REASONING_BLOCK = re.compile(r"<reasoning>.*?</reasoning>", re.S | re.I)

    @classmethod
    def strip_prose(cls, text: str) -> str:
        """
        The text the executor acts on: prose blocks removed.

        A <call_x> inside <planning> or <reasoning> is the model *referring*
        to a tool it used or intends to use ("1. <call_probe> 的结果表明…"),
        not a second request.  The acceptance batch showed both shapes in real
        output; the strict reading refused the turns and measured the parser.
        """
        stripped = cls._RE_PLANNING_BLOCK.sub(" ", text or "")
        return cls._RE_REASONING_BLOCK.sub(" ", stripped)

    # kept for callers written before the reasoning block was included
    strip_planning = strip_prose

    @classmethod
    def executable_calls(cls, text: str):
        """Calls a session may execute: those outside the prose blocks."""
        return cls.extract_all_calls(cls.strip_prose(text))

    @classmethod
    def prose_only_calls(cls, text: str) -> List[str]:
        """
        Experts whose call tags appear *only* inside a prose block.

        This is the shape the acceptance batch showed three times in one
        session: the model writes <call_probe> inside <planning>, so a
        correction that says "include a <call_*> tag" is telling it to do what
        it already did.  Naming the buried tags is what makes the correction
        actionable.
        """
        seen = {name for name, _ in cls.extract_all_calls(text)}
        outside = {name for name, _ in cls.executable_calls(text)}
        return sorted(seen - outside)

    @classmethod
    def protocol_action(cls, text: str) -> dict:
        """
        What one turn did, judged by the single-action protocol (§4.17 R9).

        One tool request or one conclusion — never both, never two requests.
        The parser is where "mixed" is *decided* (rather than the executor
        quietly picking a half), so the controller can answer it with a
        protocol correction and execute nothing.
        """
        actionable = cls.strip_prose(text)
        calls = cls.extract_all_calls(actionable)
        verdict = cls.has_verdict(actionable)
        violation = None
        if calls and verdict:
            violation = "call_and_verdict"
        elif len(calls) > 1:
            violation = "multiple_calls"
        return {"calls": calls, "verdict": verdict, "violation": violation}

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
