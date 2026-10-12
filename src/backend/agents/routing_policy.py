"""Pure routing policy for the MathTutor pipeline."""
from typing import Any
DIRECT_INTENTS = {"explain", "research", "generate", "formula_lookup"}
VALID_INTENTS = DIRECT_INTENTS | {"solve", "hint"}
def route_with_reason(state: dict[str, Any]) -> tuple[str, str]:
    if state.get("guardrail_passed") is False or state.get("prefilter_passed") is False: return "END", "input_blocked"
    if state.get("hitl_required") or (state.get("intake_output") or {}).get("needs_clarification"): return "hitl_node", "clarification_required"
    plan = state.get("solution_plan") or {}
    intent = str(plan.get("intent_type") or plan.get("intent") or "solve").casefold()
    complexity = str(state.get("complexity") or plan.get("complexity") or "standard").casefold()
    try: confidence = float(plan.get("confidence", 1.0))
    except (TypeError, ValueError): confidence = 0.0
    if intent not in VALID_INTENTS: intent = "solve"
    if intent in DIRECT_INTENTS: return "direct_response_node", f"direct_intent:{intent}"
    if intent == "hint": return "solver_agent", "hint_mode"
    if complexity == "trivial" and confidence >= 0.6: return "solver_agent", "trivial_solve_skip_retrieval"
    if confidence < 0.6: complexity = {"trivial": "standard", "standard": "hard", "hard": "hard"}.get(complexity, "hard")
    if complexity == "hard": return "retrieve_ltm", "hard_solve_retrieval_web_allowed"
    return "retrieve_ltm", "standard_solve_retrieval"
def route(state: dict[str, Any]) -> str:
    return route_with_reason(state)[0]
