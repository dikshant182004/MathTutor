"""Conservative deterministic mathematics verification helpers."""
from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from backend.agents.verification import deterministic_equation_check

_TRANSFORMS = standard_transformations + (convert_xor,)
_ALLOWED_NAMES = {"x", "y", "z", "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "E"}
_EXPR_CHARS = re.compile(r"^[A-Za-z0-9_+*/().^\-\s]+$")


def _parse_safe_expression(expression: str, symbols: dict[str, object]):
    expression = expression.strip().replace("$", "")
    if not expression or len(expression) > 160 or not _EXPR_CHARS.fullmatch(expression):
        return None
    names = re.findall(r"[A-Za-z_]\w*", expression)
    if any(name not in _ALLOWED_NAMES for name in names):
        return None
    return parse_expr(expression, local_dict=symbols, transformations=_TRANSFORMS)


def _verify_integral(problem: str, final_answer: str) -> dict | None:
    text = problem.strip().replace("$", "")
    match = re.search(
        r"(?:integrate|integral\s+of|antiderivative\s+of)\s+(.+?)(?:\s+d([xyz]))?\s*[?.!]*$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    integrand_text, variable_name = match.group(1).strip(), (match.group(2) or "").lower()
    if not variable_name:
        variables = set(re.findall(r"[xyz]", integrand_text.lower()))
        if len(variables) != 1:
            return None
        variable_name = next(iter(variables))
    symbol = sp.Symbol(variable_name)
    symbols = {
        "x": sp.Symbol("x"), "y": sp.Symbol("y"), "z": sp.Symbol("z"),
        "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
        "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E,
    }
    try:
        integrand = _parse_safe_expression(integrand_text, symbols)
        answer_text = re.sub(r"(?i)\s*\+\s*(?:c|constant)\s*$", "", final_answer.strip())
        answer_text = re.sub(r"(?i)^\s*(?:final\s+answer|answer)\s*:\s*", "", answer_text)
        antiderivative = _parse_safe_expression(answer_text, symbols)
        if integrand is None or antiderivative is None:
            return None
        difference = sp.simplify(sp.diff(antiderivative, symbol) - integrand)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, sp.SympifyError):
        return None
    correct = difference == 0
    return {
        "status": "correct" if correct else "incorrect",
        "verdict": (
            "Symbolic derivative check confirmed the proposed antiderivative."
            if correct else
            "Symbolic derivative check found that differentiating the proposed antiderivative does not recover the integrand."
        ),
        "suggested_fix": None if correct else "Differentiate the proposed antiderivative and correct it so the derivative matches the integrand.",
        "confidence": 1.0,
        "hitl_reason": None,
        "verification_method": "symbolic",
    }


def verify_math(problem: str, final_answer: str) -> dict | None:
    """Return a deterministic verdict when a safe, supported check is conclusive."""
    equation_result = deterministic_equation_check(problem, final_answer)
    if equation_result is not None:
        equation_result["verification_method"] = "symbolic"
        return equation_result
    return _verify_integral(problem, final_answer)
