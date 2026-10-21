from __future__ import annotations

from backend.agents.nodes.compose import compose_node


def test_compose_is_deterministic_and_parses_solution_steps():
    state = {
        "solver_output": {
            "solution": "Step 1 — Isolate x:\nSubtract 1 from both sides.\n∴ x = 1\n\n∴ Final Answer: x = 1",
            "final_answer": "x = 1",
        },
        "parsed_data": {"topic": "algebra", "constraints": ["x > 0"]},
        "solution_plan": {"topic": "algebra"},
        "ltm_context": {},
    }

    first = compose_node(state)
    second = compose_node(state)

    assert first == second
    assert first["solution_struct"]["final_answer"] == "x = 1"
    assert first["solution_struct"]["steps"][0]["heading"] == "Step 1 — Isolate x"
    assert "Watch out" in first["final_response"]
    assert "Key formulae" not in first["final_response"]


def test_compose_keeps_existing_final_response_when_solver_output_is_empty():
    result = compose_node({
        "solver_output": {},
        "final_response": "Existing answer",
        "parsed_data": {},
        "solution_plan": {},
        "ltm_context": {},
    })
    assert result["final_response"] == "Existing answer"
