from pathlib import Path

import pytest


@pytest.mark.unit
def test_env_example_contains_required_api_keys():
    env_example = Path(__file__).resolve().parents[3] / ".env.example"
    text        = env_example.read_text(encoding="utf-8")

    present_keys: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        if key:
            present_keys.add(key)

    required = {
        "GROQ_API_KEY",
        "GROQ_API_KEY_2",
        "COHERE_API_KEY",
        "TAVILY_API_KEY",
        "GOOGLE_CLIENT_ID",
        "GOOGLE_CLIENT_SECRET",
    }
    missing = sorted(required - present_keys)
    assert not missing, f"Missing required keys in .env.example: {missing}"