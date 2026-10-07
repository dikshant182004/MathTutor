from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from backend.agents.utils.helper import _get_secret

ModelRole = Literal["fast", "reasoning"]

@dataclass(frozen=True)
class ModelConfig:
    provider: str
    fast_model: str
    reasoning_model: str
    temperature: float = 0.1
    max_tokens: int = 2048
    max_retries: int = 2
    base_url: str | None = None

    @classmethod
    def from_environment(cls):
        return cls(
            provider=_get_secret("LLM_PROVIDER", "groq").strip().lower(),
            fast_model=_get_secret("LLM_FAST_MODEL", "openai/gpt-oss-20b").strip(),
            reasoning_model=_get_secret("LLM_REASONING_MODEL", "openai/gpt-oss-120b").strip(),
            temperature=float(_get_secret("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(_get_secret("LLM_MAX_TOKENS", "2048")),
            max_retries=int(_get_secret("LLM_MAX_RETRIES", "2")),
            base_url=_get_secret("LLM_BASE_URL", "").strip() or None,
        )

    def model_for(self, role: ModelRole) -> str:
        return self.fast_model if role == "fast" else self.reasoning_model

class ModelGateway:
    """Provider boundary: groq, openrouter, openai, anthropic, ollama or custom."""

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig.from_environment()

    def _key(self, provider: str) -> str:
        keys={"groq":"GROQ_API_KEY","openrouter":"OPENROUTER_API_KEY","openai":"OPENAI_API_KEY",
              "anthropic":"ANTHROPIC_API_KEY","ollama":"LLM_API_KEY","custom":"LLM_API_KEY"}
        return _get_secret(keys.get(provider, "LLM_API_KEY"), "")

    def _openai_compatible(self, role: ModelRole) -> BaseChatModel:
        provider=self.config.provider
        base=self.config.base_url
        if not base:
            base={"openrouter":"https://openrouter.ai/api/v1","ollama":"http://localhost:11434/v1","custom":None}.get(provider)
        key=self._key(provider) or ("ollama" if provider=="ollama" else "")
        if provider not in {"ollama","custom"} and not key:
            raise ValueError(f"Missing API key for {provider}")
        kwargs={"api_key":key or "none","model":self.config.model_for(role),
                "temperature":self.config.temperature,"max_tokens":self.config.max_tokens,
                "max_retries":self.config.max_retries}
        if base: kwargs["base_url"]=base
        return ChatOpenAI(**kwargs)

    def chat(self, role: ModelRole) -> BaseChatModel:
        p=self.config.provider
        if p=="groq":
            key=self._key("groq")
            if not key: raise ValueError("GROQ_API_KEY is not set")
            return ChatGroq(api_key=key,model_name=self.config.model_for(role),
                            temperature=self.config.temperature,max_tokens=self.config.max_tokens,
                            max_retries=self.config.max_retries)
        if p=="anthropic":
            key=self._key("anthropic")
            if not key: raise ValueError("ANTHROPIC_API_KEY is not set")
            return ChatAnthropic(model=self.config.model_for(role),api_key=key,
                                 temperature=self.config.temperature,max_tokens=self.config.max_tokens,
                                 max_retries=self.config.max_retries)
        if p in {"openrouter","openai","ollama","custom"}:
            return self._openai_compatible(role)
        raise ValueError(f"Unsupported LLM_PROVIDER={p!r}")

    @property
    def fast(self): return self.chat("fast")
    @property
    def reasoning(self): return self.chat("reasoning")
