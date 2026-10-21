"""Deterministic response composition; deliberately makes no model calls."""
from __future__ import annotations

import re
from typing import Any

_STEP = re.compile(
    r"(?ms)^\s*(Step\s+\d+\s*[—-]\s*[^:\n]+):?\s*\n(.*?)(?=^\s*Step\s+\d+\s*[—-]|\Z)"
)
_MATH = re.compile(r"\$\$?.+?\$\$?", re.DOTALL)


def compose_node(state: dict[str, Any]) -> dict[str, Any]:
    """Render solver output, structured steps, and a deterministic memory note."""
    solver = state.get("solver_output") or {}
    plan = state.get("solution_plan") or {}
    parsed = state.get("parsed_data") or {}
    solution = str(solver.get("solution") or "").strip()
    answer = str(solver.get("final_answer") or "").strip()
    if not solution:
        solution = str(state.get("final_response") or "").strip()
    if not solution:
        solution = "I couldn't produce a complete solution. Please try rephrasing the problem."

    steps = []
    for match in _STEP.finditer(solution):
        title = match.group(1).strip().rstrip(":")
        working = match.group(2).strip()
        steps.append({"heading": title, "working": working})

    sections = [solution]
    if answer and answer not in solution:
        sections.append(f"**Final answer:** {answer}")

    formulas = []
    for candidate in _MATH.findall(solution):
        cleaned = candidate.strip()
        if cleaned not in formulas:
            formulas.append(cleaned)
        if len(formulas) == 3:
            break
    if formulas and not re.search(r"(?im)^\s*#{1,4}\s*Key formulae\b", solution):
        sections.append("### Key formulae\n" + "\n".join(f"- {item}" for item in formulas))

    constraints = parsed.get("constraints") or []
    if constraints and not re.search(r"(?im)^\s*#{1,4}\s*Watch out\b", solution):
        sections.append("### Watch out\n- Check that the result satisfies the stated constraints.")

    ltm = state.get("ltm_context") or {}
    patterns = ltm.get("mistake_patterns") or []
    topic = parsed.get("topic") or plan.get("topic") or ""
    if patterns:
        pattern = patterns[0].get("pattern") if isinstance(patterns[0], dict) else str(patterns[0])
        if pattern:
            sections.append(
                f"**Personalised note:** When working on {topic or 'this topic'}, "
                f"keep an eye on this previously observed mistake: {str(pattern)[:180]}."
            )

    rendered = "\n\n".join(sections).strip()
    return {
        "final_response": rendered,
        "conversation_log": [rendered],
        "solution_struct": {
            "steps": steps,
            "final_answer": answer,
            "key_formulae": formulas,
        },
    }
