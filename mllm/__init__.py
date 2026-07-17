"""MLLM client abstraction for the Active Forensic Agent system."""

from .base import BaseMLLMClient
from .mock_client import MockMLLMClient

__all__ = ["BaseMLLMClient", "MockMLLMClient"]
