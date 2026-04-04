import importlib.util
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


SRC          = Path(__file__).resolve().parents[2]
solver_mod   = _load(SRC / "backend" / "agents" / "nodes" / "solver.py",                          "int_solver_mod")
verifier_mod = _load(SRC / "backend" / "agents" / "nodes" / "verifier.py",                        "int_verifier_mod")
mm           = _load(SRC / "backend" / "agents" / "nodes" / "memory" / "memory_manager.py",        "int_mm_mod")
artifacts_mod = _load(SRC / "backend" / "agents" / "utils" / "artifacts.py",                       "int_artifacts_mod")

verifier_mod.VerifierOutput = getattr(verifier_mod, "VerifierOutput", artifacts_mod.VerifierOutput)


# ── fake LLMs ─────────────────────────────────────────────────────────────────

class _BoundSolverLLM:
    def invoke(self, _):
        return AIMessage(content="Step 1: x+1=2 → x=1\n∴ Final Answer: x = 1")


class _FakeSolverReserveLLM:
    def bind_tools(self, _tools, tool_choice="auto"):
        return _BoundSolverLLM()


class _FakeVerifierLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _):
        class R:
            status = "correct"
            verdict = "x = 1 is verified."
            suggested_fix = ""
            confidence = 1.0
            hitl_reason = None

            def model_dump(self):
                return {
                    "status": self.status,
                    "verdict": self.verdict,
                    "suggested_fix": self.suggested_fix,
                    "confidence": self.confidence,
                    "hitl_reason": self.hitl_reason,
                }

        return R()


# ── test ──────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_full_solve_verify_store_pipeline(monkeypatch):
    # ── patch solver dependencies ──────────────────────────────────────────
    monkeypatch.setattr(solver_mod, "has_store",               lambda _: False)
    monkeypatch.setattr(
        solver_mod,
        "trim_messages_if_needed",
        lambda messages, thread_id=None, llm=None, **_: messages,
    )
    monkeypatch.setattr(solver_mod, "format_ltm_for_solver",   lambda *_: "")
    monkeypatch.setattr(solver_mod, "payload",                 lambda *a, **k: None)
    monkeypatch.setattr(
        solver_mod, "_make_scoped_rag",
        lambda _: (_ for _ in ()).throw(RuntimeError("RAG must not run")),
    )

    solver_agent             = solver_mod.SolverAgent.__new__(solver_mod.SolverAgent)
    solver_agent.reserve_llm = _FakeSolverReserveLLM()

    # ── step 1: solve ──────────────────────────────────────────────────────
    solver_state = {
        "parsed_data":    {"problem_text": "Solve x+1=2", "topic": "algebra"},
        "solution_plan":  {"intent_type": "solve", "difficulty": "easy",
                           "solver_strategy": "isolate x"},
        "solve_iterations": 0,
        "thread_id":      "int-thread-1",
        "messages":       [],
        "ltm_context":    {},
        "verifier_output": {},
        "human_feedback": "",
        "agent_payload_log": [],
    }
    solver_out = solver_agent.solver_agent(solver_state)
    assert solver_out["solver_output"]["final_answer"] == "x = 1"

    # ── step 2: verify ─────────────────────────────────────────────────────
    verifier_agent     = verifier_mod.VerifierAgent.__new__(verifier_mod.VerifierAgent)
    verifier_agent.llm = _FakeVerifierLLM()

    verifier_out = verifier_agent.verifier_agent({
        "parsed_data":   {"problem_text": "Solve x+1=2"},
        "solver_output": solver_out["solver_output"],
        "solve_iterations": solver_out["solve_iterations"],
    })
    assert verifier_out["verifier_output"]["status"] == "correct"

    # ── step 3: memory store ───────────────────────────────────────────────
    calls = {k: 0 for k in ("episodic", "semantic", "procedural", "increment", "thread")}
    monkeypatch.setattr(mm, "store_episodic_memory",    lambda **_: calls.__setitem__("episodic",   calls["episodic"]   + 1))
    monkeypatch.setattr(mm, "update_semantic_memory",   lambda **_: calls.__setitem__("semantic",   calls["semantic"]   + 1))
    monkeypatch.setattr(mm, "update_procedural_memory", lambda **_: calls.__setitem__("procedural", calls["procedural"] + 1))
    monkeypatch.setattr(mm, "increment_problems_solved",lambda *_:  calls.__setitem__("increment",  calls["increment"]  + 1))
    monkeypatch.setattr(mm, "update_thread_meta",       lambda **_: calls.__setitem__("thread",     calls["thread"]     + 1))

    mm_out = mm.memory_manager_node({
        "ltm_mode":       "store",
        "student_id":     "s_int",
        "thread_id":      "int-thread-1",
        "parsed_data":    solver_state["parsed_data"],
        "solution_plan":  solver_state["solution_plan"],
        "solver_output":  solver_out["solver_output"],
        "verifier_output": verifier_out["verifier_output"],
        "solve_iterations": solver_out["solve_iterations"],
    })

    assert mm_out["ltm_stored"]    is True
    assert calls["episodic"]       == 1
    assert calls["semantic"]       == 1
    assert calls["procedural"]     == 1
    assert calls["increment"]      == 1
    assert calls["thread"]         == 1