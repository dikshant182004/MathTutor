from __future__ import annotations

from backend.agents.utils.helper import _get_secret, MediaProcessor
from backend.llm.router import ModelRouter


class BaseAgent:
    """Shared base for agent nodes; model configuration lives in ModelRouter."""

    def __init__(self):
        key = _get_secret("GROQ_API_KEY")
        key_2 = _get_secret("GROQ_API_KEY_2")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets."
            )
        self.model_router = ModelRouter(api_key=key, reasoning_api_key=key_2 or None)
        self.llm = self.model_router.chat("fast", profile_id="balanced")
        self.reserve_llm = self.model_router.chat("reasoning", profile_id="balanced")
        self.media_processor = MediaProcessor()
