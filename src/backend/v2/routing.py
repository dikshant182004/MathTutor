from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ExecutionTier(StrEnum):
    DIRECT = "direct"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass(frozen=True)
class ExecutionPlan:
    tier: ExecutionTier
    use_rag: bool
    use_web: bool
    use_calculator: bool
    deterministic_verify: bool
    llm_verify: bool
    socratic: bool


def classify_problem(text: str, *, has_student_notes: bool = False, explicit_web: bool = False) -> ExecutionPlan:
    """Cheap pre-routing; no LLM call required."""
    normalized = text.lower().strip()
    words = len(normalized.split())
    complex_markers = (
        "prove", "derive", "show that", "olympiad", "aime", "multi-step",
        "integral", "differential equation", "eigenvalue", "probability distribution",
    )
    direct_markers = ("what is", "define", "formula for", "meaning of", "convert")
    solve_markers = ("solve", "calculate", "find", "evaluate", "simplify")
    deep = any(marker in normalized for marker in complex_markers) or words > 90
    direct = not deep and (any(normalized.startswith(m) for m in direct_markers) or (words < 12 and not any(normalized.startswith(m) for m in solve_markers)))

    if deep:
        tier = ExecutionTier.DEEP
    elif direct:
        tier = ExecutionTier.DIRECT
    else:
        tier = ExecutionTier.STANDARD

    return ExecutionPlan(
        tier=tier,
        use_rag=has_student_notes and tier != ExecutionTier.DIRECT,
        use_web=explicit_web,
        use_calculator=tier != ExecutionTier.DIRECT,
        deterministic_verify=tier != ExecutionTier.DIRECT,
        llm_verify=tier == ExecutionTier.DEEP,
        socratic=False,
    )
