from backend.v2.socratic import next_socratic_step


def test_socratic_mode_does_not_immediately_reveal_solution():
    step = next_socratic_step(problem="2x+1=5", student_attempt="", skill="linear equations")
    assert step.kind == "diagnose"
    assert step.reveal_level == 0
