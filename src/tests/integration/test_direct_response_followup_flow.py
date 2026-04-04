"""
tests/integration/test_direct_response_followup_flow.py
=========================================================
Integration test: dissatisfied follow-up → next DirectResponseAgent call
incorporates the follow-up question into its prompt.
No real LLM — _RecordingLLM records what messages were sent.
"""
import importlib.util
from pathlib import Path

import pytest

_DR_PATH   = Path(__file__).resolve().parents[2] / "backend" / "agents" / "nodes" / "direct_response.py"
_HITL_PATH = Path(__file__).resolve().parents[2] / "backend" / "agents" / "nodes" / "hitl.py"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


dr   = _load(_DR_PATH,   "integration_dr_module")
hitl = _load(_HITL_PATH, "integration_hitl_module")
_process_satisfaction_response = hitl._process_satisfaction_response


class _FakeResp:
    def __init__(self, content: str):
        self.content = content


class _RecordingLLM:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return _FakeResp("<content>Refined explanation.</content>")


@pytest.mark.integration
def test_dissatisfied_follow_up_changes_next_prompt(monkeypatch):
    monkeypatch.setattr(dr, "payload", lambda *a, **k: None)
    monkeypatch.setattr(dr, "format_ltm_for_explainer", lambda *_: "")

    llm   = _RecordingLLM()
    agent = dr.DirectResponseAgent.__new__(dr.DirectResponseAgent)
    agent.reserve_llm = llm

    state = {
        "parsed_data":   {"problem_text": "Explain chain rule", "topic": "calculus"},
        "solution_plan": {"intent_type": "explain", "difficulty": "medium"},
        "raw_text":      "Explain chain rule",
        "ltm_context":   {},
        "agent_payload_log": [],
    }

    first = agent.direct_response_agent(state)
    assert first["hitl_type"] == "satisfaction"

    resume = _process_satisfaction_response(
        {"satisfied": False, "follow_up": "focus on intuition"}, state
    )
    rerun_state = {**state, **resume}

    second = agent.direct_response_agent(rerun_state)
    assert second["final_response"].startswith("## 📖 Explanation")

    last_prompt = llm.calls[-1][1].content   # second message = user prompt
    assert "focus on intuition" in last_prompt


@pytest.mark.integration
def test_satisfied_response_ends_loop(monkeypatch):
    """If the student is satisfied, no follow-up question should be injected."""
    monkeypatch.setattr(dr, "payload", lambda *a, **k: None)
    monkeypatch.setattr(dr, "format_ltm_for_explainer", lambda *_: "")

    state = {
        "parsed_data": {"problem_text": "Explain limits", "topic": "calculus"},
        "raw_text":    "Explain limits",
    }
    resume = _process_satisfaction_response({"satisfied": True, "follow_up": ""}, state)

    assert resume["student_satisfied"]  is True
    assert resume["follow_up_question"] is None
    assert "user_corrected_text" not in resume