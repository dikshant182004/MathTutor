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
