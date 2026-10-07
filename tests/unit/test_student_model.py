from backend.v2.student_model import StudentModel


def test_mastery_moves_up_after_success():
    model = StudentModel()
    before = model.skills.get("algebra")
    model.observe("algebra", correct=True)
    assert model.skills["algebra"].mastery > (before.mastery if before else 0.5)


def test_weakest_skill_is_sorted():
    model = StudentModel()
    model.observe("algebra", correct=False)
    model.observe("calculus", correct=True)
    assert model.weakest_skills(1)[0].skill == "algebra"


def test_difficulty_adapts():
    model = StudentModel()
    for _ in range(5):
        model.observe("algebra", correct=False)
    assert model.next_difficulty("algebra") == "easy"
