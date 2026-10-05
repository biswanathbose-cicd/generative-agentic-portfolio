"""Multi-agent retail customer-support assistant (offline-first, LLM-pluggable)."""

from .assistant import SupportAssistant, Session, Result, build_assistant
from .config import AssistantConfig, V1_BASELINE, V2_IMPROVED, V2_1_FIXED

__all__ = [
    "SupportAssistant",
    "Session",
    "Result",
    "build_assistant",
    "AssistantConfig",
    "V1_BASELINE",
    "V2_IMPROVED",
    "V2_1_FIXED",
]
