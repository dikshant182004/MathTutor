import json
from pathlib import Path

from backend.v2.student_memory import StudentMemoryStore


def test_student_memory_isolated(tmp_path: Path):
    store = StudentMemoryStore(str(tmp_path / "memory.json"))
    store.record_attempt("alice", "algebra", correct=True)
    store.record_attempt("bob", "algebra", correct=False, error="sign error")

    assert store.snapshot("alice")["skills"][0]["correct"] == 1
    assert store.snapshot("bob")["skills"][0]["correct"] == 0
    assert store.mistakes("alice") == []
    assert store.mistakes("bob")[0]["count"] == 1


def test_mastery_moves_in_expected_direction(tmp_path: Path):
    store = StudentMemoryStore(str(tmp_path / "memory.json"))
    before = store.get_skill("s", "calculus").mastery
    store.record_attempt("s", "calculus", correct=True, difficulty=0.5)
    after_correct = store.get_skill("s", "calculus").mastery
    store.record_attempt("s", "calculus", correct=False, difficulty=0.5, error="chain rule")
    after_wrong = store.get_skill("s", "calculus").mastery

    assert after_correct > before
    assert after_wrong < after_correct


def test_graph_contains_skill_and_mistake_edge(tmp_path: Path):
    store = StudentMemoryStore(str(tmp_path / "memory.json"))
    store.record_attempt("s", "probability", correct=False, error="forgot conditional denominator")
    graph = store.graph("s")
    assert "skill:probability" in graph["nodes"]
    assert any(edge["target"] == "skill:probability" for edge in graph["edges"])
