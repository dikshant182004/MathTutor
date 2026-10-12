from __future__ import annotations

from backend.agents.runtime_telemetry import budget_limits, learning_evidence, record_node_timing


def test_node_timing_accumulates_real_durations_without_mutating_state():
    state = {"node_timings_ms": {"solver": 4.0}}
    result = record_node_timing(state, "solver", 2.25)
    assert result == {"solver": 6.25}
    assert state["node_timings_ms"] == {"solver": 4.0}


def test_budget_limits_are_bounded_and_request_local():
    assert budget_limits({}) == (8, 6)
    assert budget_limits({"max_llm_calls": 999, "max_tool_calls": -3}) == (30, 0)
    assert budget_limits({"max_llm_calls": 2, "max_tool_calls": 3}) == (2, 3)


def test_learning_evidence_only_marks_verified_answer_correct():
    base = {
        "parsed_data": {"topic": "algebra"},
        "solution_plan": {"difficulty": "easy"},
        "solver_output": {"final_answer": "x = 1"},
        "solve_iterations": 2,
    }
    assert learning_evidence({**base, "verifier_output": {"status": "correct"}}) == {
        "topic": "algebra", "difficulty": "easy", "outcome": "correct",
        "verified": True, "attempts": 2, "final_answer": "x = 1",
    }
    assert learning_evidence({**base, "verifier_output": {"status": "needs_human"}})["outcome"] == "unverified"
    assert learning_evidence({**base, "verifier_output": {"status": "incorrect"}})["outcome"] == "incorrect"


def test_timed_node_records_duration_and_counts_model_node_invocation():
    from backend.agents.runtime_telemetry import timed_node

    result = timed_node("fake_model", lambda state: {"ok": True}, model_node=True)(
        {"node_timings_ms": {}, "llm_calls_used": 0, "max_llm_calls": 2}
    )
    assert result["ok"] is True
    assert result["llm_calls_used"] == 1
    assert result["node_timings_ms"]["fake_model"] >= 0


def test_model_budget_stops_before_invoking_node():
    from backend.agents.runtime_telemetry import timed_node

    called = []
    result = timed_node("fake_model", lambda state: called.append(True) or {"ok": True}, model_node=True)(
        {"node_timings_ms": {}, "llm_calls_used": 1, "max_llm_calls": 1}
    )
    assert called == []
    assert result["budget_exhausted"] is True
    assert "budget" in result["final_response"].lower()
