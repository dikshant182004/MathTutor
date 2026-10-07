from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SocraticStep:
    kind: str
    prompt: str
    reveal_level: int


def next_socratic_step(
    *,
    problem: str,
    student_attempt: str,
    skill: str,
    hint_level: int = 0,
) -> SocraticStep:
    """Generate a deterministic tutoring policy; wording remains an LLM concern."""
    level = max(0, min(3, hint_level))
    if not student_attempt.strip():
        return SocraticStep(
            "diagnose",
            f"Before solving, what information is given and what quantity are you asked to find?",
            level,
        )
    if level == 0:
        return SocraticStep("prompt", f"What mathematical concept connects this problem to {skill}?", 0)
    if level == 1:
        return SocraticStep("prompt", "What equation or relationship could you write down first?", 1)
    if level == 2:
        return SocraticStep("hint", "Try the first algebraic step and explain why it is valid.", 2)
    return SocraticStep("worked_hint", "Use the relevant formula, substitute the known values, and simplify one step at a time.", 3)
