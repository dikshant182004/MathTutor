"""Small pure helpers for per-request orchestration telemetry and budgets."""
from __future__ import annotations

from time import perf_counter
from typing import Any, Callable


def record_node_timing(state: dict[str, Any], node: str, elapsed_ms: float) -> dict[str, float]:
    """Return cumulative, real wall-clock duration per graph node."""
    timings = dict(state.get("node_timings_ms") or {})
    timings[node] = round(float(timings.get(node, 0.0)) + max(0.0, elapsed_ms), 3)
    return timings


def timed_node(
    node_name: str,
    fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    model_node: bool = False,
):
    """Wrap a node while preserving its updates and recording measured duration."""
    def wrapped(state: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        try:
            max_llm, _ = budget_limits(state)
            used = int(state.get("llm_calls_used") or 0)
            if model_node and used >= max_llm:
                return {
                    "budget_exhausted": True,
                    "final_response": (
                        "This request reached its configured model-call budget. "
                        "Please narrow the question or start a new request."
                    ),
                    "node_timings_ms": record_node_timing(
                        state, node_name, (perf_counter() - started) * 1000.0
                    ),
                }
            updates = fn(state) or {}
            if model_node:
                updates = {**updates, "llm_calls_used": used + 1}
            return {
                **updates,
                "node_timings_ms": record_node_timing(
                    state, node_name, (perf_counter() - started) * 1000.0
                ),
            }
        except Exception:
            # Failed node time is still useful telemetry; never swallow the error.
            record_node_timing(state, node_name, (perf_counter() - started) * 1000.0)
            raise
    wrapped.__name__ = getattr(fn, "__name__", node_name)
    return wrapped


def budget_limits(state: dict[str, Any]) -> tuple[int, int]:
    """Read safe request-local defaults; callers may override them per request."""
    return (
        max(1, min(int(state.get("max_llm_calls") or 8), 30)),
        max(0, min(int(state.get("max_tool_calls") if state.get("max_tool_calls") is not None else 6), 20)),
    )


def learning_evidence(state: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic learning evidence; only a verified answer counts as success."""
    verifier = state.get("verifier_output") or {}
    solver = state.get("solver_output") or {}
    parsed = state.get("parsed_data") or {}
    plan = state.get("solution_plan") or {}
    status = verifier.get("status") or "unverified"
    if status == "correct":
        outcome = "correct"
    elif status in {"incorrect", "partially_correct"}:
        outcome = "incorrect"
    else:
        outcome = "unverified"
    return {
        "topic": parsed.get("topic") or plan.get("topic") or "unknown",
        "difficulty": plan.get("difficulty") or "unknown",
        "outcome": outcome,
        "verified": status == "correct",
        "attempts": max(1, int(state.get("solve_iterations") or 1)),
        "final_answer": str(solver.get("final_answer") or "")[:500],
    }
