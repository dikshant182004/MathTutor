"""Deterministic input screening and conservative symbolic complexity detection."""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
from typing import Any
import sympy as sp
from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

_POLICY_DIR = Path(__file__).resolve().parent / "security_checks"
_TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.I)
_PHONE_CONTEXT = re.compile(r"(?:\bcall\s+me\b|\bphone\b|\bmobile\b|\bcontact\b|\+91)", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
_AADHAAR_CONTEXT = re.compile(r"\baadhaar\b|\buidai\b|\bidentity\s+number\b", re.I)
_AADHAAR = re.compile(r"(?<!\d)\d{4}[\s-]\d{4}[\s-]\d{4}(?!\d)")
@lru_cache(maxsize=1)
def _patterns() -> tuple[tuple[str, ...], tuple[str, ...]]:
    import yaml
    data: dict[str, Any] = yaml.safe_load((_POLICY_DIR / "injection_patterns.yaml").read_text(encoding="utf-8")) or {}
    return tuple(str(x).casefold() for x in data.get("prompt_injection", [])), tuple(str(x).casefold() for x in data.get("extraction_attempts", []))
def _text_from_state(state: dict[str, Any]) -> str:
    for key in ("user_corrected_text", "raw_text", "ocr_text", "transcript"):
        value = state.get(key)
        if isinstance(value, str) and value.strip(): return value.strip()
    return ""
def _blocked_pii(text: str) -> dict[str, Any]:
    return {"safe": False, "block_reason": "pii", "student_message": "Please remove personal information and rephrase your math problem.", "cleaned_text": text, "complexity": "standard"}
def check_input(text: str) -> dict[str, Any]:
    cleaned = (text or "").strip()
    lowered = cleaned.casefold()
    injections, extractions = _patterns()
    for pattern in (*injections, *extractions):
        if pattern and pattern in lowered:
            return {"safe": False, "block_reason": "prompt_injection", "student_message": "I can help with mathematics, but I can't follow requests to reveal or override internal instructions.", "cleaned_text": cleaned, "complexity": "standard"}
    if _EMAIL.search(cleaned): return _blocked_pii(cleaned)
    if _PHONE_CONTEXT.search(cleaned) and _PHONE.search(cleaned): return _blocked_pii(cleaned)
    if _AADHAAR_CONTEXT.search(cleaned) and _AADHAAR.search(cleaned): return _blocked_pii(cleaned)
    return {"safe": True, "block_reason": None, "student_message": None, "cleaned_text": cleaned, "complexity": detect_complexity(cleaned)}
def detect_complexity(text: str) -> str:
    candidate = (text or "").strip()
    if not candidate or len(candidate) > 500 or not re.fullmatch(r"[\w\s+*/^().,=<>!|%-]+", candidate, re.UNICODE): return "standard"
    if re.search(r"\b(?:integrate|differentiate|limit|solve|matrix|sum|product|int|diff)\b", candidate, re.I): return "standard"
    names = set(re.findall(r"[A-Za-z_]\w*", candidate))
    allowed_names = {"x", "y", "z", "n", "a", "b", "c", "t", "r", "theta", "sqrt", "sin", "cos", "tan", "log", "exp", "pi", "E"}
    if "__" in candidate or names - allowed_names:
        return "standard"
    try:
        if candidate.count("=") > 1: return "standard"
        if "=" in candidate:
            lhs, rhs = candidate.split("=", 1)
            expr = sp.expand(parse_expr(lhs, transformations=_TRANSFORMATIONS, evaluate=False) - parse_expr(rhs, transformations=_TRANSFORMATIONS, evaluate=False))
        else: expr = parse_expr(candidate, transformations=_TRANSFORMATIONS, evaluate=False)
        if not isinstance(expr, sp.Expr) or expr.has(sp.Integral, sp.Derivative, sp.Sum, sp.Product): return "standard"
        symbols = expr.free_symbols
        if len(symbols) > 3 or sp.count_ops(expr) > 40 or expr.is_polynomial(*symbols) is False: return "standard"
        degree = sp.Poly(expr, *symbols).total_degree() if symbols else 0
        return "trivial" if degree <= 2 else "standard"
    except Exception: return "standard"
def prefilter_node(state: dict[str, Any]) -> dict[str, Any]:
    result = check_input(_text_from_state(state))
    return {"prefilter_passed": result["safe"], "guardrail_passed": result["safe"], "guardrail_reason": result["block_reason"], "final_response": result["student_message"] if not result["safe"] else state.get("final_response"), "cleaned_problem": result["cleaned_text"], "complexity": result["complexity"], "agent_payload_log": state.get("agent_payload_log") or []}
