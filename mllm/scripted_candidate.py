"""Scripted assistant turns for the first-round SFT candidate exemplars.

The tool calls still run through the real controller and the real PROBE/TruFor
adapters.  Only the assistant text is pre-written, so a candidate can
demonstrate evidence reading without pretending that Qwen wrote it.  The class
fills `{{PROBE_ID}}` / `{{TRUFOR_ID}}` (and score placeholders) from the
evidence tokens that have actually been delivered so far; it never invents an
evidence id.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List

from mllm.base import BaseMLLMClient

_EVIDENCE_ID = re.compile(r'"evidence_id"\s*:\s*"(E-[0-9a-f]+)"')
_EXPERT = re.compile(r'"expert"\s*:\s*"([^"]+)"')


class ScriptedCandidateMLLM(BaseMLLMClient):
    def __init__(self, outputs: List[str], system_prompt: str = "",
                 name: str = "sft_candidate_scripted",
                 call_order: List[str] | None = None):
        self._outputs = list(outputs)
        self._index = 0
        self.system_prompt = system_prompt
        self._name = name
        self._call_order = list(call_order or [])
        self._delivered: List[dict] = []
        self._generated: List[dict] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode(self) -> str:
        return "sft_candidate_scripted"

    def reset(self) -> None:
        self._index = 0
        self._delivered = []
        self._generated = []

    def _evidence(self, history: List[Dict[str, str]]) -> Dict[str, dict]:
        found: Dict[str, dict] = {}
        for turn in history:
            value = turn.get("value") or ""
            if not value.lstrip().startswith("{"):
                continue
            try:
                payload = json.loads(value)
            except ValueError:
                continue
            if not isinstance(payload, dict):
                continue
            expert = payload.get("expert")
            evidence_id = payload.get("evidence_id")
            if expert and evidence_id:
                found[expert] = payload
        return found

    def _fill(self, text: str, history: List[Dict[str, str]]) -> str:
        evidence = self._evidence(history)
        probe = evidence.get("probe_dino_expert") or {}
        trufor = evidence.get("trufor_expert") or {}
        replacements = {
            "{{PROBE_ID}}": probe.get("evidence_id", ""),
            "{{TRUFOR_ID}}": trufor.get("evidence_id", ""),
            "{{PROBE_SCORE}}": str((probe.get("score") or {}).get("value", "")),
            "{{TRUFOR_SCORE}}": str((trufor.get("score") or {}).get("value", "")),
        }
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text

    def generate(self, image_path: str, history: List[Dict[str, str]]) -> str:
        if self._index >= len(self._outputs):
            raise RuntimeError(
                f"scripted candidate exhausted after {len(self._outputs)} outputs; "
                "the controller asked for another turn")
        output = self._fill(self._outputs[self._index], history)
        self._index += 1
        self._generated.append({"turn": self._index, "output": output,
                                "history_len": len(history)})
        return output

    def call_log(self) -> list:
        return list(self._generated)

    def system_prompt_record(self) -> dict:
        return {"text": self.system_prompt,
                "sha256": __import__("hashlib").sha256(
                    self.system_prompt.encode("utf-8")).hexdigest(),
                "chars": len(self.system_prompt)}
