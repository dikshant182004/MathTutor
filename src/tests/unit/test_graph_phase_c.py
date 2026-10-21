from __future__ import annotations

import importlib.util
from pathlib import Path


def test_phase_c_graph_compiles_without_legacy_satisfaction_or_explainer(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")

    from langgraph.checkpoint.memory import InMemorySaver
    import backend.agents.utils.db_utils as db_utils

    monkeypatch.setattr(db_utils, "build_stm_checkpointer", lambda: InMemorySaver())

    graph_path = Path(__file__).resolve().parents[2] / "backend" / "agents" / "graph.py"
    spec = importlib.util.spec_from_file_location("phase_c_graph_module", graph_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)

    nodes = set(module.workflow.app.get_graph().nodes)
    assert "compose_node" in nodes
    assert "record_learning" in nodes
    assert "explainer_agent" not in nodes
    assert "store_ltm" not in nodes
