from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain_groq import ChatGroq

from backend.agents.utils.helper import _get_secret


ModelRole = Literal["fast", "reasoning"]


@dataclass(frozen=True)
class ModelConfig:
    """Provider/model configuration resolved from environment or Streamlit secrets."""

    provider: str
    fast_model: str
    reasoning_model: str
    temperature: float = 0.1
    max_tokens: int = 2048
    max_retries: int = 2

    @classmethod
    def from_environment(cls) -> "ModelConfig":
        return cls(
            provider=_get_secret("LLM_PROVIDER", "groq").strip().lower(),
            fast_model=_get_secret(
                "LLM_FAST_MODEL", "openai/gpt-oss-20b"
            ).strip(),
            reasoning_model=_get_secret(
                "LLM_REASONING_MODEL", "openai/gpt-oss-120b"
            ).strip(),
            temperature=float(_get_secret("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(_get_secret("LLM_MAX_TOKENS", "2048")),
            max_retries=int(_get_secret("LLM_MAX_RETRIES", "2")),
        )

    def model_for(self, role: ModelRole) -> str:
        return self.fast_model if role == "fast" else self.reasoning_model


class ModelGateway:
    """Small provider boundary used by agents.

    Agent nodes should depend on roles (fast/reasoning), not concrete model IDs.
    Additional providers can be added here without changing the graph nodes.
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig.from_environment()

        if self.config.provider != "groq":
            raise ValueError(
                f"Unsupported LLM_PROVIDER={self.config.provider!r}. "
                "V2 currently enables Groq; the gateway is intentionally "
                "provider-neutral so additional providers can be added next."
            )

        self.primary_key = _get_secret("GROQ_API_KEY")
        self.reserve_key = _get_secret("GROQ_API_KEY_2") or self.primary_key

        if not self.primary_key:
            raise ValueError(
                "GROQ_API_KEY is not set — add it to .env or Streamlit secrets."
            )

    def chat(self, role: ModelRole) -> ChatGroq:
        key = self.primary_key if role == "fast" else self.reserve_key

        return ChatGroq(
            api_key=key,
            model_name=self.config.model_for(role),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            max_retries=self.config.max_retries,
        )

    @property
    def fast(self) -> ChatGroq:
        return self.chat("fast")

    @property
    def reasoning(self) -> ChatGroq:
        return self.chat("reasoning")
