"""Construct Groq chat models and apply bounded transient-error fallback."""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

from langchain_core.runnables import RunnableLambda
from langchain_groq import ChatGroq

from backend.llm.profiles import ModelProfile, get_profile
from backend.llm.registry import resolve_model_id

logger = logging.getLogger(__name__)


def _status_code(error: BaseException) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient_provider_error(error: BaseException) -> bool:
    status = _status_code(error)
    return status == 429 or (status is not None and 500 <= status < 600)


class RoutedChatModel:
    """Small ChatGroq-compatible proxy with one retry and one profile downgrade."""

    def __init__(self, router: "ModelRouter", tier: str, profile_id: str) -> None:
        self.router = router
        self.tier = tier
        self.profile_id = get_profile(profile_id).id
        self._primary = router.chat(tier, self.profile_id)

    def _execute(self, operation: Callable[[Any], Any]) -> Any:
        try:
            return operation(self._primary)
        except Exception as first_error:
            if not _is_transient_provider_error(first_error):
                raise

        # Exactly one backoff retry on the same model.
        time.sleep(0.5)
        try:
            return operation(self._primary)
        except Exception as retry_error:
            if not _is_transient_provider_error(retry_error):
                raise

        fallback_profile = self.router.fallback_profile(self.profile_id)
        if fallback_profile is None:
            raise retry_error

        primary_signature = (
            self.router.model_id(self.tier, self.profile_id),
            self.router.effort(self.tier, self.profile_id),
        )
        fallback_signature = (
            self.router.model_id(self.tier, fallback_profile),
            self.router.effort(self.tier, fallback_profile),
        )
        # Do not repeat the same model/effort after exhausting the retry budget.
        if primary_signature == fallback_signature:
            raise retry_error

        logger.warning(
            "model_fallback tier=%s profile=%s fallback_profile=%s status=%s",
            self.tier, self.profile_id, fallback_profile, _status_code(retry_error),
        )
        fallback_model = self.router.chat(self.tier, fallback_profile)
        return operation(fallback_model)

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        def call(model: Any) -> Any:
            if config is None:
                return model.invoke(input, **kwargs)
            return model.invoke(input, config=config, **kwargs)
        return self._execute(call)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        return RunnableLambda(
            lambda input: self._execute(
                lambda model: model.with_structured_output(schema, **kwargs).invoke(input)
            )
        )

    def bind_tools(self, tools: Any, **kwargs: Any) -> RunnableLambda:
        return RunnableLambda(
            lambda input: self._execute(
                lambda model: model.bind_tools(tools, **kwargs).invoke(input)
            )
        )

    def __getattr__(self, name: str) -> Any:
        # Preserve read-only ChatGroq attributes used by existing helper code.
        return getattr(self._primary, name)


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

    @staticmethod
    def effort(tier: str, profile_id: str | None = None) -> str | None:
        profile = get_profile(profile_id)
        if tier == "fast":
            return "low"
        if tier == "reasoning":
            return profile.reasoning_effort
        if tier == "safety":
            return None
        raise ValueError(f"Unknown model tier: {tier!r}")

    def chat(
        self,
        tier: str,
        profile_id: str | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> ChatGroq:
        """Build a fresh ChatGroq instance; callers pass profile IDs, never model IDs."""
        profile: ModelProfile = get_profile(profile_id)
        model_id = self.model_id(tier, profile.id)
        if tier == "fast":
            effort, key = "low", self.api_key
        elif tier == "reasoning":
            effort = reasoning_effort or profile.reasoning_effort
            key = self.reasoning_api_key or self.api_key
        else:
            effort, key = reasoning_effort, self.api_key
        kwargs = {
            "api_key": key,
            "model": model_id,
            "temperature": 0.1,
            "max_tokens": 2048,
            # RoutedChatModel owns the single retry and fallback budget.
            "max_retries": 0,
            "reasoning_format": "hidden",
        }
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        return ChatGroq(**kwargs)

    def routed_chat(self, tier: str, profile_id: str | None = None) -> RoutedChatModel:
        return RoutedChatModel(self, tier, get_profile(profile_id).id)

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
