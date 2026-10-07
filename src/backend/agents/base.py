from __future__ import annotations

from backend.agents.utils.helper import MediaProcessor
from backend.runtime.model_gateway import ModelGateway


class BaseAgent:
    """Shared base for all agent nodes."""

    def __init__(self):
        gateway = ModelGateway()
        self.llm = gateway.fast
        self.reserve_llm = gateway.reasoning
        self.media_processor = MediaProcessor()
