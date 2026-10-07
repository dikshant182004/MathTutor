from pathlib import Path
from backend.v2.student_memory import StudentMemoryStore

def test_student_memory_isolated(tmp_path: Path):
    store=StudentMemoryStore(str(tmp_path/"memory.json"))
    store.record_attempt("alice","algebra",correct=True)
    store.record_attempt("bob","algebra",correct=False,error="sign error")
    assert store.snapshot("alice")["skills"][0]["correct"]==1
    assert store.mistakes("alice")==[]
    assert store.mistakes("bob")[0]["count"]==1

def test_mastery_moves_in_expected_direction(tmp_path: Path):
    store=StudentMemoryStore(str(tmp_path/"memory.json"))
    before=store.get_skill("s","calculus").mastery
    store.record_attempt("s","calculus",correct=True)
    after=store.get_skill("s","calculus").mastery
    store.record_attempt("s","calculus",correct=False,error="chain rule")
    assert after>before and store.get_skill("s","calculus").mastery<after


def test_memory_namespace_isolation(tmp_path):
    store=StudentMemoryStore(str(tmp_path/"memory.json"))
    store.remember("a","semantic","method","factorization",0.9)
    store.remember("b","procedural","strategy","check units",0.8)
    assert store.recall("a","semantic")[0]["value"]=="factorization"
    assert store.recall("a","procedural")==[]
    assert store.recall("b","semantic")==[]


def test_next_problem_reinforces_prerequisite(tmp_path):
    store=StudentMemoryStore(str(tmp_path/"memory.json"))
    store.record_attempt("s","calculus",correct=False,error="needs prerequisite")
    result=store.next_problem("s")
    assert result["skill"] in {"algebra","functions"}
    assert result["target_skill"]=="calculus"


def test_retention_decay_is_bounded(tmp_path):
    store = StudentMemoryStore(str(tmp_path / "memory.json"))
    state = store.record_attempt("s", "algebra", correct=True)
    assert 0.0 <= store._effective_mastery(state.__dict__) <= 1.0


def test_mistake_pattern_aggregation(tmp_path):
    store = StudentMemoryStore(str(tmp_path / "memory.json"))
    store.record_attempt("s", "algebra", correct=False, error="sign error")
    store.record_attempt("s", "algebra", correct=False, error="sign error")
    assert store.mistakes("s", 1)[0]["count"] == 2
