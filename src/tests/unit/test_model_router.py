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
    assert captured["max_retries"] == 0


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
def test_transient_errors_retry_once_then_downgrade_profile(monkeypatch):
    calls = []
    class ProviderError(Exception):
        status_code = 503
    class FakeModel:
        def __init__(self, model, effort):
            self.model = model
            self.effort = effort
        def invoke(self, _input, **_kwargs):
            calls.append((self.model, self.effort))
            if self.model == GPT_OSS_120B:
                raise ProviderError("temporary upstream failure")
            return "fallback-result"
    class FakeChatGroq:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.effort = kwargs.get("reasoning_effort")
        def invoke(self, input, **kwargs):
            return FakeModel(self.model, self.effort).invoke(input, **kwargs)
    monkeypatch.setattr("backend.llm.router.ChatGroq", FakeChatGroq)
    monkeypatch.setattr("backend.llm.router.time.sleep", lambda _seconds: None)
    result = ModelRouter(api_key="primary").routed_chat("reasoning", "balanced").invoke("question")
    assert result == "fallback-result"
    assert calls == [
        (GPT_OSS_120B, "medium"),
        (GPT_OSS_120B, "medium"),
        (GPT_OSS_20B, "low"),
    ]


@pytest.mark.unit
def test_non_transient_errors_do_not_retry_or_fallback(monkeypatch):
    calls = []
    class FakeModel:
        def invoke(self, _input, **_kwargs):
            calls.append("called")
            raise ValueError("bad request")
    class FakeChatGroq:
        def __init__(self, **_kwargs):
            pass
        def invoke(self, input, **kwargs):
            return FakeModel().invoke(input, **kwargs)
    monkeypatch.setattr("backend.llm.router.ChatGroq", FakeChatGroq)
    routed = ModelRouter(api_key="primary").routed_chat("reasoning", "balanced")
    with pytest.raises(ValueError, match="bad request"):
        routed.invoke("question")
    assert calls == ["called"]


@pytest.mark.unit
def test_safety_model_id_is_allowlisted(monkeypatch):
    monkeypatch.delenv("MODEL_SAFETY", raising=False)
    assert resolve_model_id("safety") == GPT_OSS_SAFEGUARD_20B


@pytest.mark.unit
def test_live_structured_output_smoke_all_profiles():
    from pydantic import BaseModel, Field
    if os.getenv("MATH_TUTOR_LIVE_SMOKE") != "1" or not os.getenv("GROQ_API_KEY"):
        pytest.skip("Set MATH_TUTOR_LIVE_SMOKE=1 and GROQ_API_KEY to run live Groq smoke tests.")
    class SmokeResult(BaseModel):
        result: str = Field(description="The exact word requested by the smoke test.")
    router = ModelRouter(api_key=os.environ["GROQ_API_KEY"], reasoning_api_key=os.getenv("GROQ_API_KEY_2"))
    for profile_id in ("fast", "balanced", "deep"):
        model = router.chat("reasoning", profile_id).with_structured_output(
            SmokeResult, method="json_schema", strict=True
        )
        parsed = model.invoke('Return JSON with result set to exactly "ok".')
        assert parsed.result == "ok"


@pytest.mark.unit
def test_live_structured_output_smoke_existing_agent_schemas():
    from backend.agents.utils.artifacts import (
        ExplainerOutput, GuardrailOutput, IntentRouterOutput, ParserOutput,
        SafetyOutput, VerifierOutput,
    )
    if os.getenv("MATH_TUTOR_LIVE_SMOKE") != "1" or not os.getenv("GROQ_API_KEY"):
        pytest.skip("Set MATH_TUTOR_LIVE_SMOKE=1 and GROQ_API_KEY to test the six live agent schemas.")
    router = ModelRouter(api_key=os.environ["GROQ_API_KEY"], reasoning_api_key=os.getenv("GROQ_API_KEY_2"))
    examples = [
        (ParserOutput, 'Parse "x + 1 = 2". Return problem_text="x + 1 = 2", topic="algebra", variables=["x"], constraints=[], needs_clarification=false, clarification_reason=null.'),
        (IntentRouterOutput, 'Classify "x + 1 = 2". Return topic="algebra", difficulty="easy", solver_strategy="Subtract 1 from both sides.", intent_type="solve".'),
        (GuardrailOutput, 'This is a safe algebra equation. Return passed=true, topic="algebra", block_reason=null, message=null.'),
        (SafetyOutput, 'The output "x = 1" is safe. Return passed=true, violation_type=null, reason="No safety issue found."'),
        (VerifierOutput, 'The equation x + 1 = 2 has answer x = 1. Return status="correct", verdict="Substitution confirms x=1.", suggested_fix=null, hitl_reason=null, confidence=0.99.'),
        (ExplainerOutput, 'Explain x + 1 = 2. Return approach_summary="Subtract 1 from both sides.", steps=[{step_number:1, heading:"Subtract 1", working:"x + 1 - 1 = 2 - 1", result:"x = 1", why:"Keep the equation balanced.", inline_diagram:null}], final_answer="x = 1", key_formulae=[], key_concepts=["equation solving"], common_mistakes=[], difficulty_rating="easy".'),
    ]
    model = router.chat("fast", "fast")
    for schema, prompt in examples:
        parsed = model.with_structured_output(schema, method="json_schema", strict=False).invoke(prompt)
        assert isinstance(parsed, schema)
