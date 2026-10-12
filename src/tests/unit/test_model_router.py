from __future__ import annotations

import os
import pytest

from backend.llm.profiles import PROFILES, get_profile, public_profiles
from backend.llm.registry import (
    GPT_OSS_20B, GPT_OSS_120B, GPT_OSS_SAFEGUARD_20B, resolve_model_id,
)
from backend.llm.router import ModelRouter


@pytest.mark.unit
def test_profiles_match_instruction_document():
    assert set(PROFILES) == {"fast", "balanced", "deep"}
    assert PROFILES["fast"].label == "Quick"
    assert PROFILES["fast"].reasoning_model == GPT_OSS_20B
    assert PROFILES["fast"].reasoning_effort == "low"
    assert PROFILES["fast"].solver_retries == 1
    assert PROFILES["balanced"].reasoning_model == GPT_OSS_120B
    assert PROFILES["balanced"].reasoning_effort == "medium"
    assert PROFILES["balanced"].solver_retries == 2
    assert PROFILES["deep"].reasoning_effort == "high"
    assert PROFILES["deep"].solver_retries == 2
    assert PROFILES["deep"].final_retry_effort == "high"


@pytest.mark.unit
def test_public_profiles_do_not_expose_provider_model_ids():
    profiles = public_profiles()
    assert [p["id"] for p in profiles] == ["fast", "balanced", "deep"]
    assert all("openai/" not in str(p) for p in profiles)


@pytest.mark.unit
def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="Unsupported model profile"):
        get_profile("client-supplied-model-name")


@pytest.mark.unit
def test_model_id_override_is_allowlisted(monkeypatch):
    monkeypatch.setenv("MODEL_FAST", GPT_OSS_20B)
    assert resolve_model_id("fast") == GPT_OSS_20B
    monkeypatch.setenv("MODEL_FAST", "some/unapproved-model")
    with pytest.raises(ValueError, match="allowlisted"):
        resolve_model_id("fast")


@pytest.mark.unit
def test_reasoning_tier_uses_second_key_when_configured(monkeypatch):
    captured = {}
    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr("backend.llm.router.ChatGroq", FakeChatGroq)
    ModelRouter(api_key="primary", reasoning_api_key="secondary").chat("reasoning", "balanced")
    assert captured["api_key"] == "secondary"
    assert captured["model"] == GPT_OSS_120B
    assert captured["reasoning_effort"] == "medium"
    assert captured["max_retries"] == 1


@pytest.mark.unit
def test_fast_profile_uses_fast_model_and_low_effort(monkeypatch):
    captured = {}
    class FakeChatGroq:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    monkeypatch.setattr("backend.llm.router.ChatGroq", FakeChatGroq)
    ModelRouter(api_key="primary").chat("reasoning", "fast")
    assert captured["model"] == GPT_OSS_20B
    assert captured["reasoning_effort"] == "low"


@pytest.mark.unit
def test_fallback_and_retry_effort_are_bounded():
    assert ModelRouter.fallback_profile("deep") == "balanced"
    assert ModelRouter.fallback_profile("balanced") == "fast"
    assert ModelRouter.fallback_profile("fast") is None
    assert ModelRouter.retry_effort("balanced", 0) == "medium"
    assert ModelRouter.retry_effort("balanced", 1) == "high"
    assert ModelRouter.retry_effort("deep", 2) == "high"


@pytest.mark.unit
def test_safety_model_id_is_allowlisted():
    assert resolve_model_id("safety") == GPT_OSS_SAFEGUARD_20B


@pytest.mark.live_smoke
def test_live_structured_output_smoke_all_profiles():
    from pydantic import BaseModel, Field

    if os.getenv("MATH_TUTOR_LIVE_SMOKE") != "1" or not os.getenv("GROQ_API_KEY"):
        pytest.skip("Set MATH_TUTOR_LIVE_SMOKE=1 and GROQ_API_KEY to run live Groq smoke tests.")

    class SmokeResult(BaseModel):
        result: str = Field(description="The exact word requested by the smoke test.")

    router = ModelRouter(
        api_key=os.environ["GROQ_API_KEY"],
        reasoning_api_key=os.getenv("GROQ_API_KEY_2"),
    )
    for profile_id in ("fast", "balanced", "deep"):
        model = router.chat("reasoning", profile_id).with_structured_output(
            SmokeResult, method="json_schema", strict=True
        )
        parsed = model.invoke('Return JSON with result set to exactly "ok".')
        assert parsed.result == "ok"
