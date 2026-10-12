"""The only model profiles exposed to MathTutor users."""
from __future__ import annotations
from dataclasses import dataclass
from backend.llm.registry import GPT_OSS_20B, GPT_OSS_120B

@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    description: str
    fast_model: str
    reasoning_model: str
    reasoning_effort: str
    solver_retries: int
    final_retry_effort: str | None = None

PROFILES: dict[str, ModelProfile] = {
    "fast": ModelProfile("fast", "Quick", "Faster, lighter reasoning", GPT_OSS_20B, GPT_OSS_20B, "low", 1),
    "balanced": ModelProfile("balanced", "Balanced", "A balance of speed and reasoning depth", GPT_OSS_20B, GPT_OSS_120B, "medium", 2),
    "deep": ModelProfile("deep", "Deep", "More deliberate reasoning for harder problems", GPT_OSS_20B, GPT_OSS_120B, "high", 2, "high"),
}

def get_profile(profile_id: str | None = None) -> ModelProfile:
    selected = profile_id or "balanced"
    try:
        return PROFILES[selected]
    except KeyError as exc:
        raise ValueError(f"Unsupported model profile: {selected!r}") from exc

def public_profiles() -> list[dict[str, str]]:
    return [{"id": p.id, "label": p.label, "description": p.description} for p in PROFILES.values()]
