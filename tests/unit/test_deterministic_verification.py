from backend.v2.verification import deterministic_math_checks


def test_valid_symbolic_answer_passes():
    result = deterministic_math_checks("x = 2 + 3", "∴ Final Answer: 5", "5")
    assert result.status == "passed"


def test_non_finite_answer_fails():
    result = deterministic_math_checks("x = 2", "∴ Final Answer: zoo", "zoo")
    assert result.status == "failed"


def test_unparseable_answer_is_inconclusive():
    result = deterministic_math_checks("solve x", "Final Answer: definitely five", "definitely five")
    assert result.status == "inconclusive"
