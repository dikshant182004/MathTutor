from evals.runner import exact_answer_accuracy


def test_exact_answer_accuracy():
    cases = [{"answer": "4"}, {"answer": "96"}]
    assert exact_answer_accuracy(cases, lambda case: case["answer"]) == 1.0


def test_router_regression():
    from evals.runner import load_cases, router_regression
    result = router_regression(load_cases("evals/datasets/router.jsonl"))
    assert result["accuracy"] == 1.0


def test_socratic_regression():
    from evals.runner import load_cases, socratic_regression
    result = socratic_regression(load_cases("evals/datasets/socratic.jsonl"))
    assert result["accuracy"] == 1.0
