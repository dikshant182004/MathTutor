"""Construct Groq chat models from validated tiers and profiles."""
from __future__ import annotations
from langchain_groq import ChatGroq
from backend.llm.profiles import ModelProfile, get_profile
from backend.llm.registry import resolve_model_id

class ModelRouter:
    """Single source of truth for model IDs, API keys, and reasoning effort."""
    def __init__(self, api_key: str, reasoning_api_key: str | None = None) -> None:
        if not api_key:
            raise ValueError("A Groq API key is required to construct the model router.")
        self.api_key = api_key
        self.reasoning_api_key = reasoning_api_key

    def model_id(self, tier: str, profile_id: str | None = None) -> str:
        profile = get_profile(profile_id)
        if tier not in ("fast", "reasoning", "safety"):
            raise ValueError(f"Unknown model tier: {tier!r}")
        default = {"fast": profile.fast_model, "reasoning": profile.reasoning_model, "safety": None}[tier]
        return resolve_model_id(tier, default)

    def chat(self, tier: str, profile_id: str | None = None, *, reasoning_effort: str | None = None) -> ChatGroq:
        profile: ModelProfile = get_profile(profile_id)
        model_id = self.model_id(tier, profile.id)
        if tier == "fast":
            effort, key = "low", self.api_key
        elif tier == "reasoning":
            effort, key = reasoning_effort or profile.reasoning_effort, self.reasoning_api_key or self.api_key
        else:
            effort, key = reasoning_effort, self.api_key
        kwargs = {
            "api_key": key, "model": model_id, "temperature": 0.1,
            "max_tokens": 2048, "max_retries": 1, "reasoning_format": "hidden",
        }
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        return ChatGroq(**kwargs)

    @staticmethod
    def fallback_profile(profile_id: str) -> str | None:
        if profile_id not in ("fast", "balanced", "deep"):
            raise ValueError(f"Unsupported model profile: {profile_id!r}")
        return {"deep": "balanced", "balanced": "fast", "fast": None}[profile_id]

    @staticmethod
    def retry_effort(profile_id: str, attempt: int) -> str:
        profile = get_profile(profile_id)
        if attempt <= 0:
            return profile.reasoning_effort
        if profile.final_retry_effort and attempt >= profile.solver_retries:
            return profile.final_retry_effort
        if profile.reasoning_effort in ("low", "medium"):
            return "medium" if profile.reasoning_effort == "low" else "high"
        return "high"
