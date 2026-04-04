"""
tests/integration/test_clarification_to_router_loop.py
=======================================================
Integration test: HITL clarification response → corrected text → router re-route.
Verifies that _process_clarification_response produces a user_corrected_text
that can be fed back into the router for a fresh classification.
No real LLM.
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
hitl_mod   = _load(SRC / "backend" / "agents" / "nodes" / "hitl.py",   "int_hitl_mod")
router_mod = _load(SRC / "backend" / "agents" / "nodes" / "router.py", "int_router_mod2")

_process_clarification_response = hitl_mod._process_clarification_response


class _FakeRouterResult:
    def __init__(self):
        self.topic           = "algebra"
        self.difficulty      = "easy"
        self.intent_type     = "solve"
        self.solver_strategy = "isolate variable"

    def model_dump(self):
        return {k: getattr(self, k) for k in
                ("topic", "difficulty", "intent_type", "solver_strategy")}


class _FakeStructuredLLM:
    def invoke(self, _): return _FakeRouterResult()


class _FakeRouterLLM:
    def with_structured_output(self, _): return _FakeStructuredLLM()


@pytest.mark.integration
def test_clarification_text_reaches_router(monkeypatch):
    monkeypatch.setattr(router_mod, "payload", lambda *a, **k: None)

    original_state = {
        "parsed_data": {"problem_text": "Solve x + 2 = 5"},
        "raw_text":    "Solve x + 2 = 5",
        "ltm_context": {},
    }

    # Student adds a clarification
    clarification_patch = _process_clarification_response(
        {"corrected_text": "x must be a positive integer"},
        original_state,
    )
    assert "user_corrected_text" in clarification_patch
    assert "positive integer" in clarification_patch["user_corrected_text"]

    # Router re-runs with the corrected text as the problem
    corrected_text   = clarification_patch["user_corrected_text"]
    router_agent     = router_mod.IntentRouterAgent.__new__(router_mod.IntentRouterAgent)
    router_agent.llm = _FakeRouterLLM()

    router_out = router_agent.intent_router_agent({
        "parsed_data":       {"problem_text": corrected_text},
        "agent_payload_log": [],
    })
    assert "solution_plan" in router_out
    assert router_out["solution_plan"]["intent_type"] is not None


@pytest.mark.integration
def test_clarification_resets_solution_plan(monkeypatch):
    """After a clarification, solution_plan must be None so the router re-runs."""
    state = {
        "parsed_data": {"problem_text": "Solve x^2 = 9"},
        "raw_text":    "Solve x^2 = 9",
        "ltm_context": None,
    }
    patch = _process_clarification_response(
        {"corrected_text": "only positive root"}, state
    )
    assert patch["solution_plan"] is None