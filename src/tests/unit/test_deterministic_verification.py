from backend.agents.verification import deterministic_equation_check
import importlib.util
from pathlib import Path

_VERIFY_PATH = Path(__file__).resolve().parents[2] / "backend" / "agents" / "nodes" / "verify_math.py"
_VERIFY_SPEC = importlib.util.spec_from_file_location("verify_math_module", _VERIFY_PATH)
_verify_module = importlib.util.module_from_spec(_VERIFY_SPEC)
assert _VERIFY_SPEC and _VERIFY_SPEC.loader
_VERIFY_SPEC.loader.exec_module(_verify_module)
verify_math = _verify_module.verify_math


def test_deterministic_verifier_accepts_unique_correct_linear_solution():
    result = deterministic_equation_check("Solve x + 1 = 2", "x = 1")
    assert result is not None
    assert result["status"] == "correct"
    assert result["verification_method"] == "sympy_unique_numeric_equation"


def test_deterministic_verifier_rejects_wrong_claimed_solution():
    result = deterministic_equation_check("Solve x + 1 = 2", "x = 4")
    assert result is not None
    assert result["status"] == "incorrect"


def test_deterministic_verifier_declines_unsupported_or_ambiguous_input():
    assert deterministic_equation_check("Explain the quadratic formula", "x = 1") is None
    assert deterministic_equation_check("x^2 = 4", "x = 2") is None
    assert deterministic_equation_check("x + 1 = 2; y + 2 = 4", "x = 1") is None
    assert deterministic_equation_check("x^999999 = 1", "x = 1") is None



def test_verify_math_uses_symbolic_equation_check():
    result = verify_math("Solve x + 1 = 2", "x = 1")
    assert result is not None
    assert result["status"] == "correct"
    assert result["verification_method"] == "symbolic"


def test_verify_math_checks_integral_by_differentiation():
    result = verify_math("Integrate x^2 dx", "x^3/3 + C")
    assert result is not None
    assert result["status"] == "correct"
    assert result["verification_method"] == "symbolic"


def test_verify_math_declines_unrecognized_problem():
    assert verify_math("Explain why calculus matters", "x^3/3") is None
