from backend.v2.routing import ExecutionTier, classify_problem


def test_short_definition_is_direct():
    plan = classify_problem("What is a derivative?")
    assert plan.tier == ExecutionTier.DIRECT
    assert not plan.llm_verify


def test_complex_problem_gets_deep_verification():
    plan = classify_problem("Prove that the integral of x squared from zero to one equals one third.")
    assert plan.tier == ExecutionTier.DEEP
    assert plan.deterministic_verify
    assert plan.llm_verify


def test_notes_enable_rag_only_when_useful():
    assert classify_problem("Explain Bayes theorem", has_student_notes=True).use_rag
    assert not classify_problem("What is pi?", has_student_notes=True).use_rag
