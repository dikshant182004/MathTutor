import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "backend" / "agents" / "nodes" / "hitl.py"
)
_SPEC = importlib.util.spec_from_file_location("hitl_module", _MODULE_PATH)
hitl  = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(hitl)

_process_clarification_response = hitl._process_clarification_response
_process_satisfaction_response  = hitl._process_satisfaction_response


# ── clarification ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_clarification_merges_original_and_new_info():
    state = {
        "parsed_data": {"problem_text": "Solve x + 2 = 5"},
        "raw_text":    "Solve x + 2 = 5",
        "ltm_context": {"weak_topics": {"algebra": 1}},
    }
    out = _process_clarification_response({"corrected_text": "also x is an integer"}, state)

    assert out["user_corrected_text"].startswith("Solve x + 2 = 5")
    assert "Student clarification" in out["user_corrected_text"]
    assert out["ltm_context"]  == state["ltm_context"]
    assert out["solution_plan"] is None


@pytest.mark.unit
def test_clarification_uses_retyped_question_as_is():
    state = {
        "parsed_data": {"problem_text": "Solve x^2 - 1 = 0"},
        "raw_text":    "Solve x^2 - 1 = 0",
        "ltm_context": None,
    }
    out = _process_clarification_response(
        {"corrected_text": "Solve x^2 - 1 = 0 for real x"}, state
    )
    assert out["user_corrected_text"] == "Solve x^2 - 1 = 0 for real x"


@pytest.mark.unit
def test_clarification_with_empty_corrected_text_falls_back_to_original():
    """Blank corrected_text must not replace the original problem."""
    state = {
        "parsed_data": {"problem_text": "Solve x + 2 = 5"},
        "raw_text":    "Solve x + 2 = 5",
        "ltm_context": None,
    }
    out = _process_clarification_response({"corrected_text": ""}, state)
    # Must still produce a non-empty corrected text rooted in the original
    assert out["user_corrected_text"]
    assert "Solve x + 2 = 5" in out["user_corrected_text"]


@pytest.mark.unit
def test_clarification_with_whitespace_only_corrected_text(monkeypatch):
    """Whitespace-only corrected_text should be treated the same as empty."""
    state = {
        "parsed_data": {"problem_text": "Find dy/dx"},
        "raw_text":    "Find dy/dx",
        "ltm_context": None,
    }
    out = _process_clarification_response({"corrected_text": "   "}, state)
    assert out["user_corrected_text"]
    assert "Find dy/dx" in out["user_corrected_text"]


# ── satisfaction ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_satisfaction_follow_up_injected_into_problem_context():
    state = {
        "parsed_data": {"problem_text": "Explain chain rule", "topic": "calculus"},
        "raw_text":    "Explain chain rule",
    }
    out = _process_satisfaction_response(
        {"satisfied": False, "follow_up": "Give geometric intuition too"}, state
    )

    assert out["student_satisfied"]     is False
    assert out["follow_up_question"]    == "Give geometric intuition too"
    assert "Student follow-up question" in out["user_corrected_text"]
    assert "Give geometric intuition too" in out["parsed_data"]["problem_text"]


@pytest.mark.unit
def test_satisfaction_positive_has_no_follow_up_injection():
    state = {"parsed_data": {"problem_text": "Explain limits"}}
    out   = _process_satisfaction_response({"satisfied": True, "follow_up": ""}, state)

    assert out["student_satisfied"]  is True
    assert out["follow_up_question"] is None
    assert "user_corrected_text" not in out


@pytest.mark.unit
def test_satisfaction_negative_with_empty_follow_up():
    """Dissatisfied but no follow-up text — should still mark not satisfied."""
    state = {
        "parsed_data": {"problem_text": "Explain limits"},
        "raw_text":    "Explain limits",
    }
    out = _process_satisfaction_response({"satisfied": False, "follow_up": ""}, state)
    assert out["student_satisfied"] is False