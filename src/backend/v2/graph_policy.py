from __future__ import annotations

from backend.v2.routing import classify_problem


def execution_plan_node(state: dict) -> dict:
    parsed = state.get("parsed_data") or {}
    problem = parsed.get("problem_text") or state.get("raw_text") or ""
    plan = classify_problem(
        problem,
        has_student_notes=bool(state.get("rag_available")),
        explicit_web=bool(state.get("web_requested")),
    )
    return {"execution_plan": plan.__dict__}
