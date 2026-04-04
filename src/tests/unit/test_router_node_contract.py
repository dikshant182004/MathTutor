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
router_mod = _load(SRC / "backend" / "agents" / "nodes" / "router.py", "router_module")


class _FakeRouterResult:
    def __init__(self, topic, difficulty, intent_type, solver_strategy):
        self.topic           = topic
        self.difficulty      = difficulty
        self.intent_type     = intent_type
        self.solver_strategy = solver_strategy

    def model_dump(self):
        return {
            "topic":           self.topic,
            "difficulty":      self.difficulty,
            "intent_type":     self.intent_type,
            "solver_strategy": self.solver_strategy,
        }


class _FakeStructuredLLM:
    def __init__(self, result):
        self.result = result

    def invoke(self, _messages):
        return self.result


class _FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, _schema):
        return _FakeStructuredLLM(self.result)


def _make_agent(result: _FakeRouterResult) -> "router_mod.IntentRouterAgent":
    agent     = router_mod.IntentRouterAgent.__new__(router_mod.IntentRouterAgent)
    agent.llm = _FakeLLM(result)
    return agent


# ── solution_plan population ──────────────────────────────────────────────────

@pytest.mark.unit
def test_router_sets_intent_type(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)
    agent = _make_agent(_FakeRouterResult("calculus", "medium", "explain", "concept-first"))
    out   = agent.intent_router_agent({
        "parsed_data": {"problem_text": "Explain chain rule"},
        "agent_payload_log": [],
    })
    assert out["solution_plan"]["intent_type"] == "explain"


@pytest.mark.unit
def test_router_sets_topic(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)
    agent = _make_agent(_FakeRouterResult("calculus", "medium", "explain", "concept-first"))
    out   = agent.intent_router_agent({
        "parsed_data": {"problem_text": "Explain chain rule"},
        "agent_payload_log": [],
    })
    assert out["solution_plan"]["topic"] == "calculus"


@pytest.mark.unit
def test_router_sets_difficulty(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)
    agent = _make_agent(_FakeRouterResult("algebra", "hard", "solve", "substitution"))
    out   = agent.intent_router_agent({
        "parsed_data": {"problem_text": "Solve a complex system"},
        "agent_payload_log": [],
    })
    assert out["solution_plan"]["difficulty"] == "hard"


@pytest.mark.unit
def test_router_sets_solver_strategy(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)
    agent = _make_agent(_FakeRouterResult("algebra", "easy", "solve", "isolate variable"))
    out   = agent.intent_router_agent({
        "parsed_data": {"problem_text": "Solve x + 1 = 2"},
        "agent_payload_log": [],
    })
    assert out["solution_plan"]["solver_strategy"] == "isolate variable"


@pytest.mark.unit
def test_router_handles_research_intent(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)
    agent = _make_agent(_FakeRouterResult("number theory", "hard", "research", "web lookup"))
    out   = agent.intent_router_agent({
        "parsed_data": {"problem_text": "Latest results on Riemann hypothesis"},
        "agent_payload_log": [],
    })
    assert out["solution_plan"]["intent_type"] == "research"