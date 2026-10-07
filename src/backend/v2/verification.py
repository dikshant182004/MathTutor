from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class VerificationResult:
    status: str
    confidence: float
    checks: tuple[str, ...]
    failures: tuple[str, ...]


def _extract_labeled_answer(solution: str) -> str:
    patterns = (
        r"(?:Final Answer|FINAL ANSWER|∴ Final Answer)\s*:\s*(.+)",
        r"\\boxed\{([^{}]+)\}",
    )
    for pattern in patterns:
        match = re.search(pattern, solution, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip().splitlines()[0].strip()
    return ""


def _parse_expression(value: str) -> Any:
    cleaned = value.strip().strip("$").strip()
    cleaned = cleaned.replace("^", "**")
    return sp.sympify(cleaned, locals={"pi": sp.pi, "e": sp.E})


def deterministic_math_checks(problem: str, solution: str, final_answer: str = "") -> VerificationResult:
    """Run cheap symbolic checks before invoking an LLM verifier."""
    checks: list[str] = []
    failures: list[str] = []
    answer = (final_answer or _extract_labeled_answer(solution)).strip()

    if not answer:
        return VerificationResult("inconclusive", 0.0, (), ("No final answer could be extracted.",))

    if re.search(r"\b(?:zoo|nan|NaN|undefined|infinity|∞)\b", answer, re.IGNORECASE):
        return VerificationResult(
            "failed", 0.99, (), ("Final answer contains an undefined or non-finite value.",)
        )

    try:
        expr = _parse_expression(answer)
        if expr.has(sp.zoo, sp.oo, sp.nan):
            return VerificationResult(
                "failed", 0.99, (), ("Final answer contains an undefined or non-finite value.",)
            )
        checks.append("Final answer parses as a valid symbolic expression.")
    except Exception:
        return VerificationResult(
            "inconclusive", 0.2, (), ("Final answer is not safely parseable as a symbolic expression.",)
        )

    equality = re.search(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^,;\n]+)", problem)
    if equality:
        lhs, rhs = equality.group(1), equality.group(2).strip()
        try:
            rhs_expr = _parse_expression(rhs)
            candidate = sp.solve(sp.Eq(sp.Symbol(lhs), rhs_expr), sp.Symbol(lhs))
            if candidate and any(sp.simplify(c - expr) == 0 for c in candidate):
                checks.append("Final answer is symbolically consistent with the detected equation.")
            else:
                failures.append("Final answer conflicts with the detected symbolic equation.")
        except Exception:
            pass

    if failures:
        return VerificationResult("failed", 0.95, tuple(checks), tuple(failures))
    return VerificationResult("passed", 0.75 if checks else 0.5, tuple(checks), tuple(failures))
