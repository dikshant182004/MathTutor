import importlib.util
from pathlib import Path

import pytest


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


SRC = Path(__file__).resolve().parents[2]
mm  = _load(
    SRC / "backend" / "agents" / "nodes" / "memory" / "memory_manager.py",
    "memory_manager_module",
)


def _store_state(**overrides) -> dict:
    base = {
        "ltm_mode":      "store",
        "student_id":    "s1",
        "thread_id":     "s1:1",
        "parsed_data":   {"topic": "calculus", "problem_text": "Integrate x^2"},
        "solution_plan": {
            "intent_type":     "solve",
            "solver_strategy": "power rule",
            "difficulty":      "easy",
        },
        "solver_output":   {"final_answer": "x^3/3 + C"},
        "verifier_output": {"status": "correct"},
        "solve_iterations": 1,
    }
    base.update(overrides)
    return base


def _patch_all_stores(monkeypatch) -> dict:
    calls = {k: 0 for k in ("episodic", "semantic", "procedural", "increment", "thread")}
    monkeypatch.setattr(mm, "store_episodic_memory",   lambda **_: calls.__setitem__("episodic",   calls["episodic"]   + 1))
    monkeypatch.setattr(mm, "update_semantic_memory",  lambda **_: calls.__setitem__("semantic",   calls["semantic"]   + 1))
    monkeypatch.setattr(mm, "update_procedural_memory",lambda **_: calls.__setitem__("procedural", calls["procedural"] + 1))
    monkeypatch.setattr(mm, "increment_problems_solved",lambda *_: calls.__setitem__("increment",  calls["increment"]  + 1))
    monkeypatch.setattr(mm, "update_thread_meta",      lambda **_: calls.__setitem__("thread",     calls["thread"]     + 1))
    return calls


# ── retrieve mode ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_retrieve_mode_calls_retrieve_ltm(monkeypatch):
    captured = {}

    def _fake_retrieve(student_id, problem_text, topic):
        captured["args"] = (student_id, problem_text, topic)
        return {"best_strategy": "substitution"}

    monkeypatch.setattr(mm, "retrieve_ltm", _fake_retrieve)

    out = mm.memory_manager_node({
        "ltm_mode":   "retrieve",
        "student_id": "s1",
        "parsed_data": {"problem_text": "Integrate x^2", "topic": "calculus"},
        "raw_text":   "fallback",
    })

    assert out["ltm_context"]["best_strategy"] == "substitution"
    assert captured["args"] == ("s1", "Integrate x^2", "calculus")


@pytest.mark.unit
def test_retrieve_mode_falls_back_to_raw_text_when_parsed_data_missing(monkeypatch):
    """If parsed_data is absent, the node should not crash."""
    monkeypatch.setattr(mm, "retrieve_ltm", lambda **_: {})

    out = mm.memory_manager_node({
        "ltm_mode":   "retrieve",
        "student_id": "s1",
        "parsed_data": None,
        "raw_text":   "Integrate x^2",
    })
    assert "ltm_context" in out


# ── store mode — explain intent ───────────────────────────────────────────────

@pytest.mark.unit
def test_store_explain_skips_semantic_procedural_and_increment(monkeypatch):
    calls = _patch_all_stores(monkeypatch)

    mm.memory_manager_node(_store_state(
        solution_plan={
            "intent_type":     "explain",
            "solver_strategy": "n/a",
            "difficulty":      "easy",
        },
        solver_output={"final_answer": ""},
        verifier_output={},
    ))

    assert calls["episodic"] == 1
    assert calls["thread"]   == 1
    assert calls["semantic"]   == 0
    assert calls["procedural"] == 0
    assert calls["increment"]  == 0


# ── store mode — solve intent (correct) ──────────────────────────────────────

@pytest.mark.unit
def test_store_solve_correct_calls_all_stores(monkeypatch):
    """A correctly solved problem must update episodic, semantic, procedural,
    increment the counter, and update thread meta."""
    calls = _patch_all_stores(monkeypatch)

    out = mm.memory_manager_node(_store_state(
        verifier_output={"status": "correct"},
    ))

    assert out["ltm_stored"]    is True
    assert calls["episodic"]    == 1
    assert calls["semantic"]    == 1
    assert calls["procedural"]  == 1
    assert calls["increment"]   == 1
    assert calls["thread"]      == 1


@pytest.mark.unit
def test_store_solve_incorrect_skips_increment(monkeypatch):
    """An incorrect solve must NOT increment the problems-solved counter."""
    calls = _patch_all_stores(monkeypatch)

    mm.memory_manager_node(_store_state(
        verifier_output={"status": "incorrect"},
    ))

    assert calls["increment"] == 0


# ── store mode — research intent ─────────────────────────────────────────────

@pytest.mark.unit
def test_store_research_intent_stores_episodic_only(monkeypatch):
    calls = _patch_all_stores(monkeypatch)

    mm.memory_manager_node(_store_state(
        solution_plan={
            "intent_type":     "research",
            "solver_strategy": "web search",
            "difficulty":      "medium",
        },
        solver_output={"final_answer": ""},
        verifier_output={},
    ))

    assert calls["episodic"]   == 1
    assert calls["semantic"]   == 0
    assert calls["procedural"] == 0
    assert calls["increment"]  == 0


# ── ltm_stored flag ───────────────────────────────────────────────────────────

@pytest.mark.unit
def test_store_mode_sets_ltm_stored_true(monkeypatch):
    _patch_all_stores(monkeypatch)
    out = mm.memory_manager_node(_store_state())
    assert out["ltm_stored"] is True