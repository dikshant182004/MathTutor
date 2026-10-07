from evals.runner import exact_answer_accuracy


def test_exact_answer_accuracy():
    cases = [{"answer": "4"}, {"answer": "96"}]
    assert exact_answer_accuracy(cases, lambda case: case["answer"]) == 1.0
