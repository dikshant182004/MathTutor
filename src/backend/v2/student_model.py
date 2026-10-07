from __future__ import annotations

from dataclasses import dataclass, field
from math import exp


@dataclass
class SkillState:
    skill: str
    mastery: float = 0.5
    attempts: int = 0
    correct: int = 0
    recent_errors: list[str] = field(default_factory=list)


@dataclass
class StudentModel:
    """Small deterministic student model; storage is intentionally external."""

    skills: dict[str, SkillState] = field(default_factory=dict)

    def observe(self, skill: str, correct: bool, difficulty: float = 0.5, error: str | None = None) -> SkillState:
        state = self.skills.setdefault(skill, SkillState(skill=skill))
        state.attempts += 1
        if correct:
            state.correct += 1
        elif error:
            state.recent_errors.append(error[:240])
            state.recent_errors = state.recent_errors[-10:]

        # Lightweight Bayesian-style update with difficulty adjustment.
        target = 0.9 if correct else 0.1
        weight = 0.15 + 0.15 * max(0.0, min(1.0, difficulty))
        state.mastery = max(0.0, min(1.0, state.mastery + weight * (target - state.mastery)))
        return state

    def weakest_skills(self, limit: int = 5) -> list[SkillState]:
        return sorted(self.skills.values(), key=lambda s: s.mastery)[:limit]

    def next_difficulty(self, skill: str) -> str:
        mastery = self.skills.get(skill, SkillState(skill)).mastery
        if mastery < 0.4:
            return "easy"
        if mastery < 0.7:
            return "medium"
        return "hard"
