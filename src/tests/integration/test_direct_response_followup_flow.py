import importlib.util
from pathlib import Path

import pytest

_DR_PATH = Path(__file__).resolve().parents[2] / "backend" / "agents" / "nodes" / "direct_response.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


dr = _load(_DR_PATH, "integration_dr_module")


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeResp("<content>Clear explanation.</content>")


@pytest.mark.integration
def test_direct_response_finishes_without_satisfaction_interrupt(monkeypatch):
    monkeypatch.setattr(dr, "payload", lambda *a, **k: None)
    monkeypatch.setattr(dr, "format_ltm_for_explainer", lambda *_: "")

    llm = _RecordingLLM()
    agent = dr.DirectResponseAgent.__new__(dr.DirectResponseAgent)
    agent.reserve_llm = llm
    state = {
        "parsed_data": {"problem_text": "Explain chain rule", "topic": "calculus"},
        "solution_plan": {"intent_type": "explain", "difficulty": "medium"},
        "raw_text": "Explain chain rule",
        "ltm_context": {},
        "agent_payload_log": [],
        "complexity": "standard",
        "tool_calls_used": 0,
    }

    result = agent.direct_response_agent(state)

    assert result["final_response"].startswith("## 📖 Explanation")
    assert result["hitl_required"] is False
    assert result["hitl_type"] is None
    assert result["llm_calls_made"] == 1
    assert len(llm.calls) == 1
