"""
Abstract base class for MLLM (Multi-modal Large Language Model) clients.

The MLLM serves as the "judge" in the forensic pipeline — it analyses the
image visually, decides which forensic experts to call, interprets their
Evidence Tokens, and produces a final reasoned verdict.

Concrete implementations:
  - MockMLLMClient   : template-driven mock for CPU-only development / testing.
  - (future) Qwen25VLClient : real Qwen2.5-VL via API or local inference.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional


class BaseMLLMClient(ABC):
    """Abstract interface for an MLLM forensic analyst."""

    @abstractmethod
    def generate(
        self,
        image_path: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Generate the next response from the MLLM.

        Args:
            image_path: Absolute path to the image under analysis.
            history: Conversation history as a list of {"from": "...", "value": "..."}
                     dicts.  The system prompt is prepended by the caller (state machine).

        Returns:
            Raw text output containing XML forensic tags
            (<planning>, <call_*>, <reasoning>, <verdict>).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state (turn counter, etc.) for a new session."""
        ...

    def call_log(self) -> list:
        """
        本次会话每次模型调用的记录（阶段 4.6）。真实客户端覆盖它；没有可记的
        （mock、旧实现）就返回空列表——**空不等于没记**：页面据此标"未记录"。
        """
        return []

    def system_prompt_record(self) -> dict:
        """初始 system prompt 的原文与哈希；拿不到就返回空 dict。"""
        return {}

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging / SFT metadata."""
        ...

    @property
    @abstractmethod
    def mode(self) -> str:
        """Current behaviour mode (for logging)."""
        ...
