"""Allowlisted Groq model IDs and environment overrides."""
from __future__ import annotations
import os
from dataclasses import dataclass

GPT_OSS_20B = "openai/gpt-oss-20b"
GPT_OSS_120B = "openai/gpt-oss-120b"
GPT_OSS_SAFEGUARD_20B = "openai/gpt-oss-safeguard-20b"
ALLOWED_MODELS = frozenset({GPT_OSS_20B, GPT_OSS_120B, GPT_OSS_SAFEGUARD_20B})
DEFAULT_MODEL_BY_TIER = {"fast": GPT_OSS_20B, "reasoning": GPT_OSS_120B, "safety": GPT_OSS_SAFEGUARD_20B}
ENV_BY_TIER = {"fast": "MODEL_FAST", "reasoning": "MODEL_REASONING", "safety": "MODEL_SAFETY"}

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    tier: str
    supports_json_schema: bool
    supports_tool_calling: bool
    supports_reasoning_effort: bool

MODEL_SPECS = {
    GPT_OSS_20B: ModelSpec(GPT_OSS_20B, "fast", True, True, True),
    GPT_OSS_120B: ModelSpec(GPT_OSS_120B, "reasoning", True, True, True),
    GPT_OSS_SAFEGUARD_20B: ModelSpec(GPT_OSS_SAFEGUARD_20B, "safety", False, True, False),
}

def resolve_model_id(tier: str, profile_default: str | None = None) -> str:
    if tier not in ENV_BY_TIER:
        raise ValueError(f"Unknown model tier: {tier!r}")
    configured = os.getenv(ENV_BY_TIER[tier])
    model_id = (configured or profile_default or DEFAULT_MODEL_BY_TIER[tier]).strip()
    if model_id not in ALLOWED_MODELS:
        raise ValueError(f"{ENV_BY_TIER[tier]} must be one of the allowlisted Groq model IDs; got {model_id!r}")
    return model_id
