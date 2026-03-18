from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agents import Agent_Exception, logger, sys
from agents.base import BaseAgent
from agents.state import AgentState

from agents.nodes.input import asr_node, detect_input_type, ocr_node
from agents.nodes.guardrail import GuardrailAgent
from agents.nodes.parser import ParserAgent
from agents.nodes.router import IntentRouterAgent
from agents.nodes.solver import SolverAgent
from agents.nodes.verifier import VerifierAgent
from agents.nodes.safety import SafetyAgent
from agents.nodes.explainer import ExplainerAgent
from agents.nodes.manim_node import manim_node
from agents.nodes.hitl import HITLAgent
from agents.nodes.memory.memory_manager import build_stm_checkpointer, memory_manager_node
from agents.nodes.tools.tools import rag_tool, web_search_tool, calculator_tool


SOLVER_TOOLS = [rag_tool, calculator_tool, web_search_tool]


def _build_checkpointer():
    """
    Prefer Redis-backed STM checkpointer (durable, supports HITL resume).
    Fall back to in-memory checkpointer if Redis is unavailable.
    """
    try:
        return build_stm_checkpointer()
    except Exception as exc:
        logger.warning(f"[STM] Falling back to InMemorySaver (Redis unavailable): {exc}")
        return InMemorySaver()


checkpointer = _build_checkpointer()


def _route_after_detect(state: AgentState) -> str:
    """detect_input -> [ocr|asr|guardrail|hitl]"""
    if state.get("hitl_required"):
        return "hitl_node"
    mode = state.get("input_mode") or ""
    if mode == "image":
        return "ocr_node"
    if mode == "audio":
        return "asr_node"
    return "guardrail_agent"


def _route_after_guardrail(state: AgentState) -> str:
    """guardrail_agent -> [retrieve_ltm|END]"""
    if state.get("guardrail_passed") is False:
        return "END"
    return "retrieve_ltm"


def _route_after_parser(state: AgentState) -> str:
    """parser_agent -> [hitl|intent_router]"""
    return "hitl_node" if state.get("hitl_required") else "intent_router"


def _route_solver_or_tools(state: AgentState) -> str:
    """solver_agent -> [tool_node|verifier_agent] (ReAct loop)"""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    if last and getattr(last, "tool_calls", None):
        return "tool_node"
    return "verifier_agent"


def _route_after_verifier(state: AgentState) -> str:
    """verifier_agent -> [safety|solver(retry)|hitl]"""
    verifier = state.get("verifier_output") or {}
    status = verifier.get("status") or "incorrect"
    if status == "correct":
        return "safety_agent"
    if status == "needs_human":
        return "hitl_node"
    return "solver_agent"


def _route_after_safety(state: AgentState) -> str:
    """safety_agent -> [explainer|END]"""
    if state.get("safety_passed") is False:
        return "END"
    return "explainer_agent"


def _route_after_hitl(state: AgentState) -> str:
    """
    hitl_node -> next step, based on hitl_type.

    - bad_input      -> detect_input (re-run input routing after new upload/text)
    - clarification  -> guardrail_agent (re-run guardrail on corrected text)
    - verification   -> safety_agent if marked correct, else solver_agent
    - satisfaction   -> store_ltm if satisfied, else explainer_agent
    """
    hitl_type = state.get("hitl_type") or ""

    if hitl_type == "bad_input":
        return "detect_input"

    if hitl_type == "clarification":
        return "guardrail_agent"

    if hitl_type == "verification":
        verifier = state.get("verifier_output") or {}
        if (verifier.get("status") or "") == "correct":
            return "safety_agent"
        return "solver_agent"

    if hitl_type == "satisfaction":
        if state.get("student_satisfied") is True:
            return "store_ltm"
        return "explainer_agent"

    # Fallback: safest restart point.
    return "guardrail_agent"


def _retrieve_ltm_node(state: AgentState) -> dict:
    try:
        state = dict(state)
        # If HITL provided corrected text, prefer it for retrieval.
        if state.get("user_corrected_text") and not state.get("raw_text"):
            state["raw_text"] = state["user_corrected_text"]
        state["ltm_mode"] = "retrieve"
        out = memory_manager_node(state)
        return {"ltm_mode": "retrieve", **out}
    except Exception as e:
        raise Agent_Exception(e, sys)


def _store_ltm_node(state: AgentState) -> dict:
    try:
        state = dict(state)
        state["ltm_mode"] = "store"
        out = memory_manager_node(state)
        return {"ltm_mode": "store", **out}
    except Exception as e:
        raise Agent_Exception(e, sys)


class MathTutorWorkflow(
    BaseAgent,
    GuardrailAgent,
    ParserAgent,
    IntentRouterAgent,
    SolverAgent,
    VerifierAgent,
    SafetyAgent,
    ExplainerAgent,
    HITLAgent,
):
    def _create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        graph.add_node("detect_input", detect_input_type)
        graph.add_node("ocr_node", ocr_node)
        graph.add_node("asr_node", asr_node)
        graph.add_node("guardrail_agent", self.guardrail_agent)
        graph.add_node("retrieve_ltm", _retrieve_ltm_node)
        graph.add_node("parser_agent", self.parser_agent)
        graph.add_node("intent_router", self.intent_router_agent)
        graph.add_node("solver_agent", self.solver_agent)
        graph.add_node("tool_node", ToolNode(SOLVER_TOOLS))
        graph.add_node("verifier_agent", self.verifier_agent)
        graph.add_node("safety_agent", self.safety_agent)
        graph.add_node("explainer_agent", self.explainer_agent)
        graph.add_node("manim_node", manim_node)
        graph.add_node("hitl_node", self.hitl_node)
        graph.add_node("store_ltm", _store_ltm_node)

        # ── Entry ──────────────────────────────────────────────────────────────
        graph.set_entry_point("detect_input")

        # detect_input -> [ocr | asr | guardrail | hitl]
        graph.add_conditional_edges(
            "detect_input",
            _route_after_detect,
            {
                "ocr_node": "ocr_node",
                "asr_node": "asr_node",
                "guardrail_agent": "guardrail_agent",
                "hitl_node": "hitl_node",
            },
        )

        # ocr/asr -> guardrail
        graph.add_edge("ocr_node", "guardrail_agent")
        graph.add_edge("asr_node", "guardrail_agent")

        # guardrail -> [retrieve_ltm | END]
        graph.add_conditional_edges(
            "guardrail_agent",
            _route_after_guardrail,
            {
                "retrieve_ltm": "retrieve_ltm",
                "END": END,
            },
        )

        # retrieve_ltm -> parser
        graph.add_edge("retrieve_ltm", "parser_agent")

        # parser -> [hitl | intent_router]
        graph.add_conditional_edges(
            "parser_agent",
            _route_after_parser,
            {
                "hitl_node": "hitl_node",
                "intent_router": "intent_router",
            },
        )

        # intent_router -> solver (kick off ReAct loop)
        graph.add_edge("intent_router", "solver_agent")

        # ReAct loop: solver <-> tools; final -> verifier
        graph.add_conditional_edges(
            "solver_agent",
            _route_solver_or_tools,
            {
                "tool_node": "tool_node",
                "verifier_agent": "verifier_agent",
            },
        )
        graph.add_edge("tool_node", "solver_agent")

        # verifier -> [safety | solver(retry) | hitl]
        graph.add_conditional_edges(
            "verifier_agent",
            _route_after_verifier,
            {
                "safety_agent": "safety_agent",
                "solver_agent": "solver_agent",
                "hitl_node": "hitl_node",
            },
        )

        # safety -> [explainer | END]
        graph.add_conditional_edges(
            "safety_agent",
            _route_after_safety,
            {
                "explainer_agent": "explainer_agent",
                "END": END,
            },
        )

        # explainer -> manim -> hitl (satisfaction)
        graph.add_edge("explainer_agent", "manim_node")
        graph.add_edge("manim_node", "hitl_node")

        # hitl -> [detect_input | guardrail | solver | safety | explainer | store_ltm]
        graph.add_conditional_edges(
            "hitl_node",
            _route_after_hitl,
            {
                "detect_input": "detect_input",
                "guardrail_agent": "guardrail_agent",
                "solver_agent": "solver_agent",
                "safety_agent": "safety_agent",
                "explainer_agent": "explainer_agent",
                "store_ltm": "store_ltm",
            },
        )

        # store_ltm -> END
        graph.add_edge("store_ltm", END)

        return graph


workflow = MathTutorWorkflow()
chatbot = workflow.app