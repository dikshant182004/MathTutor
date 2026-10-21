from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2]
PATH = SRC / "backend/agents/nodes/intake.py"
spec = importlib.util.spec_from_file_location("intake_routing_under_test", PATH)
intake_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = intake_module
spec.loader.exec_module(intake_module)

IntakeOutput = intake_module.IntakeOutput


class _StructuredOutput:
    def __init__(self, result):
        self.result = result

    def invoke(self, _messages):
        return self.result


class _FakeLLM:
    def __init__(self, result):
        self.result = result

    def with_structured_output(self, _schema, **_kwargs):
        return _StructuredOutput(self.result)


def _result(confidence: float) -> IntakeOutput:
    return IntakeOutput(
        safe=True,
        block_reason=None,
        student_message=None,
        cleaned_problem="x + 1 = 2",
        variables=["x"],
        constraints=[],
        topic="algebra",
        difficulty="easy",
        solver_strategy="Subtract 1 from both sides.",
        intent="solve",
        complexity="trivial",
        needs_clarification=False,
        clarification_reason=None,
        needs_web=False,
        confidence=confidence,
    )


@pytest.mark.parametrize(
    ("confidence", "expected_route", "expected_reason", "expected_complexity"),
    [
        (0.95, "solver_agent", "trivial_solve_skip_retrieval", "trivial"),
        (0.4, "retrieve_ltm", "standard_solve_retrieval", "standard"),
    ],
)
def test_intake_persists_route_and_applies_confidence_escalation_once(
    monkeypatch, confidence, expected_route, expected_reason, expected_complexity
):
    monkeypatch.setattr(intake_module, "payload", lambda *args, **kwargs: None)
    agent = intake_module.IntakeAgent.__new__(intake_module.IntakeAgent)
    agent.llm = _FakeLLM(_result(confidence))

    update = agent.intake_node(
        {
            "raw_text": "x + 1 = 2",
            "cleaned_problem": "x + 1 = 2",
            "complexity": "trivial",
            "prefilter_passed": True,
            "guardrail_passed": True,
            "agent_payload_log": [],
        }
    )

    assert update["route"] == expected_route
    assert update["route_reason"] == expected_reason
    assert update["complexity"] == expected_complexity
    assert update["solution_plan"]["complexity"] == expected_complexity
