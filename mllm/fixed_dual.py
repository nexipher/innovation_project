"""Force one PROBE call and one TruFor call before the real Qwen speaks.

This is an evaluation-only dialogue mode.  It does **not** replace the model,
the experts, the evidence package or the controller: the wrapper returns two
protocol-valid call turns, the existing state machine executes them and injects
their normal evidence tokens, and every later turn is delegated to the same
real Qwen client.  The fixed calls are recorded separately in the session
record so a trace cannot be mistaken for autonomous tool choice.
"""

from __future__ import annotations

from typing import Iterable, List


class FixedDualMLLM:
    """A thin, resettable wrapper around the real Qwen client."""

    def __init__(self, client, calls: Iterable[str] = ("probe", "trufor")):
        self._client = client
        self._calls = tuple(calls)
        self._index = 0
        self._forced: List[dict] = []

    def reset(self) -> None:
        self._index = 0
        self._forced = []
        reset = getattr(self._client, "reset", None)
        if callable(reset):
            reset()

    def generate(self, image_path: str, history: List[dict]) -> str:
        if self._index < len(self._calls):
            name = self._calls[self._index]
            self._index += 1
            output = (
                "<planning>\nSuspected Region: none\n</planning>\n"
                f"<call_{name}></call_{name}>"
            )
            self._forced.append({
                "turn": self._index,
                "call": name,
                "raw_output": output,
                "forced": True,
            })
            return output
        return self._client.generate(image_path, history)

    @property
    def forced_calls(self) -> List[dict]:
        return [dict(entry) for entry in self._forced]

    @property
    def mode(self):
        return getattr(self._client, "mode", "")

    @property
    def name(self):
        return getattr(self._client, "name", type(self._client).__name__)

    @property
    def system_prompt(self):
        return getattr(self._client, "system_prompt", "")

    @system_prompt.setter
    def system_prompt(self, value):
        self._client.system_prompt = value

    def identity(self):
        identity = getattr(self._client, "identity", None)
        return identity() if callable(identity) else {}

    def __getattr__(self, name):
        # Everything not overridden (call_log, last_prompt_text, model ids,
        # generation settings, ...) is the real client's own attribute.
        return getattr(self._client, name)
