"""MLLM client abstraction for the Active Forensic Agent system."""

from .base import BaseMLLMClient
from .mock_client import MockMLLMClient

__all__ = ["BaseMLLMClient", "MockMLLMClient", "QwenVLClient"]


def __getattr__(name: str):
    """
    Lazy re-export of the real client (PEP 562).

    `qwen_client` imports torch + transformers (~450 MB RSS).  Importing it
    eagerly here would make every CPU-only path — mock runs, the test suite,
    the G2-d dry run — pay that cost, which does not fit the 2 GB cgroup
    budget this environment shares with other processes.  Importing
    `mllm.qwen_client` directly still works as before.
    """
    if name == "QwenVLClient":
        from .qwen_client import QwenVLClient

        return QwenVLClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
