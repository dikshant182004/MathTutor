"""Conservative deterministic checks for simple single-variable equations."""
from __future__ import annotations

import re
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations


_ALLOWED = re.compile(r"^[0-9xyzXYZ+*/().^=\-\s]+$")
_ANSWER = re.compile(r"^\s*(?:([xyzXYZ])\s*=\s*)?([+-]?(?:\d+(?:/\d+)?|\d*\.\d+))\s*$")
_TRANSFORMS = standard_transformations + (convert_xor,)


def deterministic_equation_check(problem: str, final_answer: str) -> dict | None:
    """Return a verified verdict only for a single safe equation with one unique numeric root."""
    if not problem or not final_answer or len(problem) > 500 or len(final_answer) > 200:
        return None
    # Restrict this check to elementary arithmetic and x/y/z; no function calls or attributes.
    if problem.count("=") != 1:
        return None
    candidates = re.findall(r"([0-9xyzXYZ+*/().^=\-\s]{3,})", problem)
    equations = [item.strip() for item in candidates if item.count("=") == 1]
    if not equations:
        return None
    equation = max(equations, key=len)
    if not _ALLOWED.fullmatch(equation):
        return None
    left_text, right_text = (part.strip() for part in equation.split("=", 1))
    variables = set(re.findall(r"[xyzXYZ]", equation.lower()))
    if len(variables) != 1:
        return None
    variable_name = next(iter(variables))
    answer_match = _ANSWER.fullmatch(final_answer)
    if not answer_match:
        return None
    answer_variable = (answer_match.group(1) or variable_name).lower()
    if answer_variable != variable_name.lower():
        return None
    symbol = sp.Symbol(variable_name.lower())
    local_dict = {variable_name.lower(): symbol}
    try:
        lhs = parse_expr(left_text.lower(), local_dict=local_dict, transformations=_TRANSFORMS)
        rhs = parse_expr(right_text.lower(), local_dict=local_dict, transformations=_TRANSFORMS)
        candidate = sp.Rational(answer_match.group(2))
        solutions = sp.solve(sp.Eq(lhs, rhs), symbol)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, sp.SympifyError):
        return None
    if len(solutions) != 1 or not solutions[0].is_number:
        return None
    correct = sp.simplify(solutions[0] - candidate) == 0
    return {
        "status": "correct" if correct else "incorrect",
        "verdict": (
            "Deterministic SymPy check confirmed the unique solution satisfies the equation."
            if correct else
            "Deterministic SymPy check found that the claimed value is not the unique solution."
        ),
        "suggested_fix": None if correct else f"Solve the equation for {variable_name} and check the claimed final value.",
        "confidence": 1.0,
        "hitl_reason": None,
        "verification_method": "sympy_unique_numeric_equation",
    }
