from backend.agents.verification import deterministic_equation_check


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
