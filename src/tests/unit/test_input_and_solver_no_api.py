import importlib.util
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


SRC        = Path(__file__).resolve().parents[2]
input_mod  = _load(SRC / "backend" / "agents" / "nodes" / "input.py",  "input_module")
solver_mod = _load(SRC / "backend" / "agents" / "nodes" / "solver.py", "solver_module")


# ── helpers ───────────────────────────────────────────────────────────────────

def _full_state(**overrides) -> dict:
    base = {
        "raw_text":                 "Solve x+1=2",
        "image_path":               None,
        "audio_path":               None,
        "messages":                 [],
        "solve_iterations":         3,
        "hitl_required":            True,
        "hitl_type":                "clarification",
        "hitl_reason":              "old",
        "solver_output":            {"x": 1},
        "verifier_output":          {"status": "correct"},
        "explainer_output":         {"y": 2},
        "solution_plan":            {"intent_type": "solve"},
        "parsed_data":              {"topic": "algebra"},
        "safety_passed":            True,
        "safety_reason":            "ok",
        "student_satisfied":        True,
        "follow_up_question":       "q",
        "human_feedback":           "hf",
        "user_corrected_text":      "u",
        "agent_payload_log":        [{"a": 1}],
        "direct_response_tool_calls": [{"name": "web_search_tool"}],
    }
    base.update(overrides)
    return base


class _BoundLLM:
    def invoke(self, _messages):
        return AIMessage(content="Step 1...\n∴ Final Answer: x = 1")


class _BoundRagCallerLLM:
    def invoke(self, _messages):
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "rag_tool",
                    "args": {"query": "power rule"},
                    "id": "rag_call_1",
                    "type": "tool_call",
                }
            ],
        )


class _FakeReserveLLM:
    def bind_tools(self, _tools, tool_choice="auto"):
        # When the solver forces rag_tool, it passes a dict tool_choice with
        # function name "rag_tool". Simulate that call.
        if isinstance(tool_choice, dict):
            fn = (tool_choice.get("function") or {}).get("name")
            if fn == "rag_tool":
                return _BoundRagCallerLLM()
        return _BoundLLM()


def _patched_solver(monkeypatch) -> "solver_mod.SolverAgent":
    monkeypatch.setattr(solver_mod, "has_store",              lambda _tid: False)
    monkeypatch.setattr(
        solver_mod,
        "trim_messages_if_needed",
        lambda messages, thread_id=None, llm=None, **_: messages,
    )
    monkeypatch.setattr(solver_mod, "format_ltm_for_solver",   lambda *_: "")
    monkeypatch.setattr(solver_mod, "payload",                 lambda *a, **k: None)
    monkeypatch.setattr(
        solver_mod, "_make_scoped_rag",
        lambda _tid: (_ for _ in ()).throw(RuntimeError("RAG must not run")),
    )
    agent             = solver_mod.SolverAgent.__new__(solver_mod.SolverAgent)
    agent.reserve_llm = _FakeReserveLLM()
    return agent


# ── detect_input_type ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_detect_input_type_text_mode_resets_state(monkeypatch):
    out = input_mod.detect_input_type(_full_state())
    assert out["input_mode"]               == "text"
    assert out["solve_iterations"]         == 0
    assert out["hitl_required"]            is False
    assert out["solution_plan"]            is None
    assert out["parsed_data"]              is None
    assert out["direct_response_tool_calls"] is None


@pytest.mark.unit
def test_detect_input_type_clears_stale_hitl_fields(monkeypatch):
    out = input_mod.detect_input_type(_full_state())
    assert out["hitl_type"]   is None or out.get("hitl_type") in (None, "")
    assert out["human_feedback"]      in (None, "")
    assert out["user_corrected_text"] in (None, "")


@pytest.mark.unit
def test_detect_input_type_image_mode(monkeypatch):
    out = input_mod.detect_input_type(
        _full_state(raw_text=None, image_path="/tmp/img.png")
    )
    assert out["input_mode"] == "image"
    assert out["solve_iterations"] == 0


@pytest.mark.unit
def test_detect_input_type_audio_mode(monkeypatch):
    out = input_mod.detect_input_type(
        _full_state(raw_text=None, audio_path="/tmp/audio.wav")
    )
    assert out["input_mode"] == "audio"
    assert out["solve_iterations"] == 0


# ── SolverAgent — no-RAG path ─────────────────────────────────────────────────

@pytest.mark.unit
def test_solver_contract_no_api(monkeypatch):
    agent = _patched_solver(monkeypatch)
    state = {
        "parsed_data":    {"problem_text": "Solve x+1=2", "topic": "algebra"},
        "solution_plan":  {"intent_type": "solve", "difficulty": "easy",
                           "solver_strategy": "isolate x"},
        "solve_iterations": 0,
        "thread_id":      "t1",
        "messages":       [],
        "ltm_context":    {},
        "verifier_output": {},
        "human_feedback": "",
        "agent_payload_log": [],
    }
    out = agent.solver_agent(state)
    assert out["solve_iterations"]           == 1
    assert out["solver_output"]["final_answer"] == "x = 1"
    assert out["solver_output"]["rag_context_used"] is False


@pytest.mark.unit
def test_solver_increments_iteration_counter(monkeypatch):
    agent = _patched_solver(monkeypatch)
    state = {
        "parsed_data":    {"problem_text": "Solve 2x=4", "topic": "algebra"},
        "solution_plan":  {"intent_type": "solve", "difficulty": "easy",
                           "solver_strategy": "divide"},
        "solve_iterations": 2,        # already iterated twice
        "thread_id":      "t2",
        "messages":       [],
        "ltm_context":    {},
        "verifier_output": {},
        "human_feedback": "",
        "agent_payload_log": [],
    }
    out = agent.solver_agent(state)
    assert out["solve_iterations"] == 3


# ── SolverAgent — RAG path ────────────────────────────────────────────────────

@pytest.mark.unit
def test_solver_uses_rag_when_store_exists(monkeypatch):
    """When has_store returns True, RAG context must be fetched and flagged."""
    monkeypatch.setattr(solver_mod, "has_store", lambda _tid: True)
    monkeypatch.setattr(
        solver_mod,
        "trim_messages_if_needed",
        lambda messages, thread_id=None, llm=None, **_: messages,
    )
    monkeypatch.setattr(solver_mod, "format_ltm_for_solver",   lambda *_: "")
    monkeypatch.setattr(solver_mod, "payload",                 lambda *a, **k: None)

    fake_rag_context = "RAG: chain rule derivation steps"

    def _fake_scoped_rag(_tid):
        class _R:
            def invoke(self, _payload):
                return fake_rag_context
        return _R()

    monkeypatch.setattr(solver_mod, "_make_scoped_rag", _fake_scoped_rag)

    agent             = solver_mod.SolverAgent.__new__(solver_mod.SolverAgent)
    agent.reserve_llm = _FakeReserveLLM()

    state = {
        "parsed_data":    {"problem_text": "Differentiate x^2", "topic": "calculus"},
        "solution_plan":  {"intent_type": "solve", "difficulty": "medium",
                           "solver_strategy": "power rule"},
        "solve_iterations": 0,
        "thread_id":      "rag-thread",
        "messages":       [],
        "ltm_context":    {},
        "verifier_output": {},
        "human_feedback": "",
        "agent_payload_log": [],
    }
    out = agent.solver_agent(state)
    assert out["solver_output"]["rag_context_used"] is True