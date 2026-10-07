from backend.runtime.model_gateway import ModelConfig


def test_default_model_roles_are_current_groq_models():
    config = ModelConfig.from_environment()

    assert config.provider == "groq"
    assert config.fast_model == "openai/gpt-oss-20b"
    assert config.reasoning_model == "openai/gpt-oss-120b"


def test_model_roles_are_independent():
    config = ModelConfig(
        provider="groq",
        fast_model="fast-test",
        reasoning_model="reasoning-test",
    )

    assert config.model_for("fast") == "fast-test"
    assert config.model_for("reasoning") == "reasoning-test"
