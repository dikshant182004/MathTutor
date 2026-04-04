"""
tests/integration/test_router_to_direct_response_pipeline.py
=============================================================
Integration test: router output feeds into DirectResponseAgent.
Verifies that the solution_plan produced by the router is consumed
correctly and produces a well-formed final_response.
No real LLM — both agents are given fake LLMs.
"""
import importlib.util
from pathlib import Path

import pytest


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


SRC        = Path(__file__).resolve().parents[2]
router_mod = _load(SRC / "backend" / "agents" / "nodes" / "router.py",          "int_router_mod")
dr_mod     = _load(SRC / "backend" / "agents" / "nodes" / "direct_response.py", "int_dr_mod")


class _FakeRouterResult:
    def __init__(self, topic, difficulty, intent_type, solver_strategy):
        self.topic           = topic
        self.difficulty      = difficulty
        self.intent_type     = intent_type
        self.solver_strategy = solver_strategy

    def model_dump(self):
        return {k: getattr(self, k) for k in
                ("topic", "difficulty", "intent_type", "solver_strategy")}


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result
    def invoke(self, _):
        return self._result


class _FakeRouterLLM:
    def __init__(self, result):
        self._result = result
    def with_structured_output(self, _):
        return _FakeStructuredLLM(self._result)


class _FakeDRResp:
    def __init__(self, content):
        self.content = content


class _FakeDRLLM:
    def invoke(self, _msgs):
        return _FakeDRResp("<content>Chain rule: d/dx[f(g(x))] = f'(g(x))·g'(x).</content>")


@pytest.mark.integration
def test_router_explain_feeds_direct_response(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)
    monkeypatch.setattr(dr_mod,     "payload", lambda *a, **k: None)
    monkeypatch.setattr(dr_mod, "format_ltm_for_explainer", lambda *_: "")

    # Step 1 — router classifies the problem
    router_agent     = router_mod.IntentRouterAgent.__new__(router_mod.IntentRouterAgent)
    router_agent.llm = _FakeRouterLLM(
        _FakeRouterResult("calculus", "medium", "explain", "concept-first explanation")
    )
    router_out = router_agent.intent_router_agent({
        "parsed_data":       {"problem_text": "Explain chain rule"},
        "agent_payload_log": [],
    })
    solution_plan = router_out["solution_plan"]
    assert solution_plan["intent_type"] == "explain"

    # Step 2 — direct response agent uses the router's solution_plan
    dr_agent              = dr_mod.DirectResponseAgent.__new__(dr_mod.DirectResponseAgent)
    dr_agent.reserve_llm  = _FakeDRLLM()
    dr_out = dr_agent.direct_response_agent({
        "parsed_data":       {"problem_text": "Explain chain rule", "topic": "calculus"},
        "solution_plan":     solution_plan,
        "ltm_context":       {},
        "agent_payload_log": [],
    })

    assert dr_out["final_response"].startswith("## 📖 Explanation")
    assert "Chain rule" in dr_out["final_response"]
    assert dr_out["hitl_required"] is True
    assert dr_out["hitl_type"]     == "satisfaction"