from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.agents import Agent_Exception, logger, sys, _MEDIA_CONF_THRESHOLD
from backend.agents.state import AgentState

from backend.agents.nodes.input import asr_node, detect_input_type, ocr_node
from backend.agents.nodes.guardrail import GuardrailAgent
from backend.agents.nodes.parser import ParserAgent
from backend.agents.nodes.router import IntentRouterAgent
from backend.agents.nodes.solver import SolverAgent
from backend.agents.nodes.verifier import VerifierAgent
from backend.agents.nodes.safety import SafetyAgent
from backend.agents.nodes.explainer import ExplainerAgent
from backend.agents.nodes.manim_node import manim_node
from backend.agents.nodes.hitl import HITLAgent
from backend.agents.nodes.memory.memory_manager import memory_manager_node
from backend.agents.utils.db_utils import build_stm_checkpointer
from backend.agents.nodes.tools.tools import rag_tool, web_search_tool, calculator_tool

SOLVER_TOOLS = [rag_tool, calculator_tool, web_search_tool]

def _build_checkpointer():
    try:
        return build_stm_checkpointer()
    except Exception as exc:
        logger.warning(f"[STM] Falling back to InMemorySaver (Redis unavailable): {exc}")
        return InMemorySaver()


# ── Routing functions ──────────────────────────────────────────────────────────

def _route_after_detect(state: AgentState) -> str:
    """detect_input -> [ocr | asr | guardrail | hitl]"""
    if state.get("hitl_required"):
        return "hitl_node"
    mode = state.get("input_mode") or ""
    if mode == "image":
        return "ocr_node"
    if mode == "audio":
        return "asr_node"
    return "guardrail_agent"


# Issue 7 fix — route after OCR based on confidence
def _route_after_ocr(state: AgentState) -> str:
    """
    ocr_node -> [guardrail_agent | hitl_node]
    Triggers HITL when confidence is below threshold or text is empty.
    """
    if state.get("hitl_required"):
        return "hitl_node"
    conf = state.get("ocr_confidence") or 0.0
    text = (state.get("ocr_text") or "").strip()
    if conf < _MEDIA_CONF_THRESHOLD or not text:
        return "hitl_node"
    return "guardrail_agent"


# Issue 7 fix — route after ASR based on confidence
def _route_after_asr(state: AgentState) -> str:
    """
    asr_node -> [guardrail_agent | hitl_node]
    Triggers HITL when confidence is below threshold or transcript is empty.
    """
    if state.get("hitl_required"):
        return "hitl_node"
    conf = state.get("asr_confidence") or 0.0
    text = (state.get("transcript") or "").strip()
    if conf < _MEDIA_CONF_THRESHOLD or not text:
        return "hitl_node"
    return "guardrail_agent"


def _route_after_guardrail(state: AgentState) -> str:
    """guardrail_agent -> [retrieve_ltm | END]"""
    if state.get("guardrail_passed") is False:
        return "END"
    return "retrieve_ltm"


def _route_after_parser(state: AgentState) -> str:
    """parser_agent -> [hitl | intent_router]"""
    return "hitl_node" if state.get("hitl_required") else "intent_router"


def _route_after_intent_router(state: AgentState) -> str:
    """
    intent_router -> [solver_agent | explainer_agent | hitl_node]

    solve / hint / formula_lookup → solver_agent  (solver handles all; verifier is
        lenient for hint/formula — see _route_after_verifier)
    explain → explainer_agent directly  (no new calculation needed)
    """
    plan        = state.get("solution_plan") or {}
    intent_type = plan.get("intent_type", "solve")

    if intent_type == "explain":
        # Skip solve+verify entirely; explainer will use LTM + problem text directly
        logger.info("[Router] intent=explain — routing directly to explainer_agent")
        return "explainer_agent"

    # solve | hint | formula_lookup all go through solver
    return "solver_agent"


def _route_solver_or_tools(state: AgentState) -> str:
    """solver_agent -> [tool_node | verifier_agent] (ReAct loop)"""
    messages = state.get("messages") or []
    last     = messages[-1] if messages else None
    if last and getattr(last, "tool_calls", None):
        return "tool_node"
    return "verifier_agent"


def _route_after_verifier(state: AgentState) -> str:
    """
    verifier_agent -> [safety | solver(retry) | hitl]

    5a fix — for hint / formula_lookup intents, treat partially_correct as correct
    so we don't force a full retry just because a hint is incomplete.
    """
    verifier = state.get("verifier_output") or {}
    status   = verifier.get("status") or "incorrect"
    plan     = state.get("solution_plan") or {}
    intent   = plan.get("intent_type", "solve")

    # Lightweight intents: partial result is good enough → skip to safety
    if intent in ("hint", "formula_lookup") and status in ("correct", "partially_correct"):
        return "safety_agent"

    if status == "correct":
        return "safety_agent"
    if status == "needs_human":
        return "hitl_node"

    iterations = state.get("solve_iterations", 0)
    if iterations >= 3:
        return "hitl_node"
    return "solver_agent"


def _route_after_safety(state: AgentState) -> str:
    """safety_agent -> [explainer | END]"""
    if state.get("safety_passed") is False:
        return "END"
    return "explainer_agent"


def _route_after_hitl(state: AgentState) -> str:
    """hitl_node -> next step, based on hitl_type."""
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
    return "guardrail_agent"


def _retrieve_ltm_node(state: AgentState) -> dict:
    try:
        state = dict(state)
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


# Issue 7 fix — inject HITL fields when confidence is too low
# These are set here so hitl_node sees the right hitl_type/reason
def _ocr_node_with_confidence_gate(state: AgentState) -> AgentState:
    """Wraps ocr_node; sets hitl fields when confidence fails."""
    state = ocr_node(state)
    conf  = state.get("ocr_confidence") or 0.0
    text  = (state.get("ocr_text") or "").strip()
    if conf < _MEDIA_CONF_THRESHOLD or not text:
        state["hitl_required"] = True
        state["hitl_type"]     = "bad_input"
        state["hitl_reason"]   = (
            f"OCR confidence {conf:.0%} is below threshold. "
            "Please upload a clearer image or type the problem."
        )
        logger.warning(f"[OCR] Low confidence gate triggered: conf={conf:.2f}")
    return state


def _asr_node_with_confidence_gate(state: AgentState) -> AgentState:
    """Wraps asr_node; sets hitl fields when confidence fails."""
    state = asr_node(state)
    conf  = state.get("asr_confidence") or 0.0
    text  = (state.get("transcript") or "").strip()
    if conf < _MEDIA_CONF_THRESHOLD or not text:
        state["hitl_required"] = True
        state["hitl_type"]     = "bad_input"
        state["hitl_reason"]   = (
            f"ASR confidence {conf:.0%} is below threshold. "
            "Please re-record in a quieter environment or type the problem."
        )
        logger.warning(f"[ASR] Low confidence gate triggered: conf={conf:.2f}")
    return state


# ── Workflow class ─────────────────────────────────────────────────────────────

class MathTutorWorkflow(
    GuardrailAgent,
    ParserAgent,
    IntentRouterAgent,
    SolverAgent,
    VerifierAgent,
    SafetyAgent,
    ExplainerAgent,
    HITLAgent,
):
    def __init__(self):
        super().__init__()
        self.checkpointer = _build_checkpointer()
        graph = self._create_workflow()
        self.app = graph.compile(checkpointer=self.checkpointer)
        logger.info("[Graph] MathTutorWorkflow compiled successfully")

    def _create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        graph.add_node("detect_input",    detect_input_type)
        # Issue 7 fix — use wrapped versions that set hitl fields on low confidence
        graph.add_node("ocr_node",        _ocr_node_with_confidence_gate)
        graph.add_node("asr_node",        _asr_node_with_confidence_gate)
        graph.add_node("guardrail_agent", self.guardrail_agent)
        graph.add_node("retrieve_ltm",    _retrieve_ltm_node)
        graph.add_node("parser_agent",    self.parser_agent)
        graph.add_node("intent_router",   self.intent_router_agent)
        graph.add_node("solver_agent",    self.solver_agent)
        graph.add_node("tool_node",       ToolNode(SOLVER_TOOLS))
        graph.add_node("verifier_agent",  self.verifier_agent)
        graph.add_node("safety_agent",    self.safety_agent)
        graph.add_node("explainer_agent", self.explainer_agent)
        graph.add_node("manim_node",      manim_node)
        graph.add_node("hitl_node",       self.hitl_node)
        graph.add_node("store_ltm",       _store_ltm_node)

        # ── Entry ──────────────────────────────────────────────────────────────
        graph.set_entry_point("detect_input")

        # detect_input -> [ocr | asr | guardrail | hitl]
        graph.add_conditional_edges(
            "detect_input",
            _route_after_detect,
            {"ocr_node": "ocr_node", "asr_node": "asr_node",
             "guardrail_agent": "guardrail_agent", "hitl_node": "hitl_node"},
        )

        # Issue 7 fix — conditional edges after ocr/asr (not plain edges)
        graph.add_conditional_edges(
            "ocr_node",
            _route_after_ocr,
            {"guardrail_agent": "guardrail_agent", "hitl_node": "hitl_node"},
        )
        graph.add_conditional_edges(
            "asr_node",
            _route_after_asr,
            {"guardrail_agent": "guardrail_agent", "hitl_node": "hitl_node"},
        )

        # guardrail -> [retrieve_ltm | END]
        graph.add_conditional_edges(
            "guardrail_agent",
            _route_after_guardrail,
            {"retrieve_ltm": "retrieve_ltm", "END": END},
        )

        # retrieve_ltm -> parser
        graph.add_edge("retrieve_ltm", "parser_agent")

        # parser -> [hitl | intent_router]
        graph.add_conditional_edges(
            "parser_agent",
            _route_after_parser,
            {"hitl_node": "hitl_node", "intent_router": "intent_router"},
        )

        # 5a fix — intent_router -> [solver_agent | explainer_agent]
        graph.add_conditional_edges(
            "intent_router",
            _route_after_intent_router,
            {"solver_agent": "solver_agent", "explainer_agent": "explainer_agent"},
        )

        # ReAct loop: solver <-> tools; final -> verifier
        graph.add_conditional_edges(
            "solver_agent",
            _route_solver_or_tools,
            {"tool_node": "tool_node", "verifier_agent": "verifier_agent"},
        )
        graph.add_edge("tool_node", "solver_agent")

        # verifier -> [safety | solver(retry) | hitl]
        graph.add_conditional_edges(
            "verifier_agent",
            _route_after_verifier,
            {"safety_agent":  "safety_agent",
             "solver_agent":  "solver_agent",
             "hitl_node":     "hitl_node"},
        )

        # safety -> [explainer | END]
        graph.add_conditional_edges(
            "safety_agent",
            _route_after_safety,
            {"explainer_agent": "explainer_agent", "END": END},
        )

        # explainer -> manim -> hitl (satisfaction)
        graph.add_edge("explainer_agent", "manim_node")
        graph.add_edge("manim_node",      "hitl_node")

        # hitl -> [detect_input | guardrail | solver | safety | explainer | store_ltm]
        graph.add_conditional_edges(
            "hitl_node",
            _route_after_hitl,
            {
                "detect_input":    "detect_input",
                "guardrail_agent": "guardrail_agent",
                "solver_agent":    "solver_agent",
                "safety_agent":    "safety_agent",
                "explainer_agent": "explainer_agent",
                "store_ltm":       "store_ltm",
            },
        )

        # store_ltm -> END
        graph.add_edge("store_ltm", END)

        return graph


workflow     = MathTutorWorkflow()
chatbot      = workflow.app
checkpointer = workflow.checkpointer