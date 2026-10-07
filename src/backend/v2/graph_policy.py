from __future__ import annotations

from backend.agents.nodes.tools.tools import has_store
from backend.v2.routing import classify_problem


def execution_plan_node(state: dict) -> dict:
    parsed = state.get("parsed_data") or {}
    problem = parsed.get("problem_text") or state.get("raw_text") or ""
    thread_id = state.get("thread_id") or ""
    plan = classify_problem(
        problem,
        has_student_notes=bool(thread_id and has_store(thread_id)),
        explicit_web=bool(state.get("web_requested")),
    )
    return {"execution_plan": plan.__dict__}
