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
    # Avoid expensive symbolic work on adversarially large exponents.
    if len(equation) > 200 or re.search(r"\^\s*(?:[5-9]|\d{2,})", equation):
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



# Phase C deterministic verification beyond unique-root equations.
import re as _re
import sympy as _sp
from sympy.parsing.sympy_parser import convert_xor as _convert_xor, parse_expr as _parse_expr, standard_transformations as _standard_transformations

_VERIFY_TRANSFORMS = _standard_transformations + (_convert_xor,)
_VERIFY_ALLOWED_NAMES = {"x", "y", "z", "sin", "cos", "tan", "exp", "log", "sqrt", "pi", "E"}
_VERIFY_EXPR_CHARS = _re.compile(r"^[A-Za-z0-9_+*/().^\-\s]+$")


def _parse_safe_expression(expression: str, symbols: dict[str, object]):
    expression = expression.strip().replace("$", "")
    if not expression or len(expression) > 160 or not _VERIFY_EXPR_CHARS.fullmatch(expression):
        return None
    names = _re.findall(r"[A-Za-z_]\w*", expression)
    if any(name not in _VERIFY_ALLOWED_NAMES for name in names):
        return None
    return _parse_expr(expression, local_dict=symbols, transformations=_VERIFY_TRANSFORMS)


def _verify_integral(problem: str, final_answer: str) -> dict | None:
    text = problem.strip().replace("$", "")
    match = _re.search(
        r"(?:integrate|integral\s+of|antiderivative\s+of)\s+(.+?)(?:\s+d([xyz]))?\s*[?.!]*$",
        text,
        flags=_re.IGNORECASE,
    )
    if not match:
        return None
    integrand_text, variable_name = match.group(1).strip(), (match.group(2) or "").lower()
    if not variable_name:
        variables = set(_re.findall(r"[xyz]", integrand_text.lower()))
        if len(variables) != 1:
            return None
        variable_name = next(iter(variables))
    symbol = _sp.Symbol(variable_name)
    symbols = {
        "x": _sp.Symbol("x"), "y": _sp.Symbol("y"), "z": _sp.Symbol("z"),
        "sin": _sp.sin, "cos": _sp.cos, "tan": _sp.tan, "exp": _sp.exp,
        "log": _sp.log, "sqrt": _sp.sqrt, "pi": _sp.pi, "E": _sp.E,
    }
    try:
        integrand = _parse_safe_expression(integrand_text, symbols)
        answer_text = _re.sub(r"(?i)\s*\+\s*(?:c|constant)\s*$", "", final_answer.strip())
        answer_text = _re.sub(r"(?i)^\s*(?:final\s+answer|answer)\s*:\s*", "", answer_text)
        antiderivative = _parse_safe_expression(answer_text, symbols)
        if integrand is None or antiderivative is None:
            return None
        difference = _sp.simplify(_sp.diff(antiderivative, symbol) - integrand)
    except (SyntaxError, TypeError, ValueError, ZeroDivisionError, _sp.SympifyError):
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
