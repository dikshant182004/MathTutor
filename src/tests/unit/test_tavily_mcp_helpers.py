"""
tests/unit/test_tavily_mcp_helpers.py
======================================
Unit tests for Tavily MCP client helper functions:
  _parse_mcp_result
  _format_search_results
No network calls — all inputs are in-memory fake objects.
"""
import importlib.util
from pathlib import Path

import pytest


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


SRC        = Path(__file__).resolve().parents[2]
tavily_mod = _load(
    SRC / "backend" / "agents" / "nodes" / "tools" / "mcp" / "tavily_mcp_client.py",
    "tavily_mcp_module",
)


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResult:
    def __init__(self, data=None, content=None):
        self.data    = data
        self.content = content or []


# ── _parse_mcp_result ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_parse_mcp_result_direct_data_shape():
    payload = {"answer": "ok", "results": []}
    parsed  = tavily_mod._parse_mcp_result(_FakeResult(data=payload))
    assert parsed == payload


@pytest.mark.unit
def test_parse_mcp_result_text_json_shape():
    parsed = tavily_mod._parse_mcp_result(
        _FakeResult(content=[_FakeContent('{"answer":"ok","results":[]}')])
    )
    assert parsed["answer"] == "ok"


@pytest.mark.unit
def test_parse_mcp_result_returns_none_or_empty_on_empty_content():
    """Empty content list must not raise — return None or {}."""
    result = tavily_mod._parse_mcp_result(_FakeResult(data=None, content=[]))
    assert result is None or result == {}


@pytest.mark.unit
def test_parse_mcp_result_handles_invalid_json_in_content():
    """Invalid JSON inside content text must not raise an unhandled exception."""
    try:
        result = tavily_mod._parse_mcp_result(
            _FakeResult(content=[_FakeContent("not-json-at-all")])
        )
        # Either returns None/{} or raises a controlled exception
        assert result is None or isinstance(result, dict)
    except (ValueError, KeyError):
        pass   # controlled failure is acceptable


# ── _format_search_results ────────────────────────────────────────────────────

@pytest.mark.unit
def test_format_search_results_includes_titles_and_snippets():
    out = tavily_mod._format_search_results({
        "answer": "A short answer",
        "results": [
            {"title": "Result A", "url": "https://a", "content": "snippet A", "score": 0.87},
            {"title": "Result B", "url": "https://b", "content": "snippet B", "score": 0.52},
        ],
    })
    assert "Direct Answer" in out
    assert "Result A"      in out
    assert "snippet B"     in out


@pytest.mark.unit
def test_format_search_results_handles_missing_answer_field():
    """results without an 'answer' key must not raise."""
    out = tavily_mod._format_search_results({
        "results": [
            {"title": "T", "url": "https://t", "content": "c", "score": 0.9},
        ]
    })
    assert isinstance(out, str)
    assert len(out) > 0


@pytest.mark.unit
def test_format_search_results_handles_empty_results_list():
    out = tavily_mod._format_search_results({"answer": "nothing", "results": []})
    assert isinstance(out, str)


@pytest.mark.unit
def test_format_search_results_handles_missing_results_key():
    """Payload with only 'answer' and no 'results' key must not raise."""
    out = tavily_mod._format_search_results({"answer": "standalone answer"})
    assert isinstance(out, str)
    assert len(out) > 0