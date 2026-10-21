from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from concurrent.futures import ThreadPoolExecutor
from backend.agents.runtime_telemetry import timed_node, budget_limits, learning_evidence

from backend.agents import Agent_Exception, logger, sys, _MEDIA_CONF_THRESHOLD
from backend.agents.state import AgentState

from backend.agents.nodes.input import asr_node, detect_input_type, ocr_node
from backend.agents.nodes.prefilter import prefilter_node
from backend.agents.nodes.intake import IntakeAgent
from backend.agents.routing_policy import route_with_reason
from backend.agents.nodes.solver import SolverAgent
from backend.agents.nodes.verifier import VerifierAgent
from backend.agents.nodes.safety import SafetyAgent
from backend.agents.nodes.compose import compose_node
from backend.agents.nodes.direct_response import DirectResponseAgent   
from backend.agents.nodes.hitl import HITLAgent
from backend.agents.utils.helper import _log_payload as payload
from backend.agents.nodes.memory.memory_manager import memory_manager_node
from backend.agents.utils.db_utils import build_stm_checkpointer
from backend.agents.nodes.tools.tools import rag_tool, web_search_tool, calculator_tool, has_store

SOLVER_TOOLS = [calculator_tool, web_search_tool]

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
    return "prefilter"


def _route_after_ocr(state: AgentState) -> str:
    if state.get("hitl_required"):
        return "hitl_node"
    conf = state.get("ocr_confidence") or 0.0
    text = (state.get("ocr_text") or "").strip()
    if conf < _MEDIA_CONF_THRESHOLD or not text:
        return "hitl_node"
    return "prefilter"


def _route_after_asr(state: AgentState) -> str:
    if state.get("hitl_required"):
        return "hitl_node"
    conf = state.get("asr_confidence") or 0.0
    text = (state.get("transcript") or "").strip()
    if conf < _MEDIA_CONF_THRESHOLD or not text:
        return "hitl_node"
    return "prefilter"


def _route_after_guardrail(state: AgentState) -> str:
    """guardrail_agent -> [parser_agent | END]"""
    if state.get("guardrail_passed") is False:
        return "END"
    return "parser_agent"


def _route_after_parser(state: AgentState) -> str:
    """parser_agent -> [hitl | retrieve_ltm]"""
    return "hitl_node" if state.get("hitl_required") else "retrieve_ltm"


def _route_after_intent_router(state: AgentState) -> str:
    """
    intent_router -> [solver_agent | direct_response_node | hitl_node]
    """
    plan        = state.get("solution_plan") or {}
    intent_type = plan.get("intent_type", "solve")

    # Non-solve intents: handle directly without solve pipeline
    if intent_type in ("explain", "research", "generate"):
        logger.info(f"[Router] intent={intent_type} — routing to direct_response_node")
        return "direct_response_node"
    logger.info(f"[Router] intent={intent_type} — routing to solver_agent")
    return "solver_agent"


def _route_solver_or_tools(state: AgentState) -> str:
    """Bounded solver/tool loop; stop when the request budget is exhausted."""
    if state.get("budget_exhausted"):
        return "END"
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    calls = getattr(last, "tool_calls", None) or []
    if calls:
        _, max_tools = budget_limits(state)
        used = int(state.get("tool_calls_used") or 0)
        if used + len(calls) > max_tools:
            return "budget_stop_node"
        return "tool_node"
    return "verifier_agent"


def _route_after_verifier(state: AgentState) -> str:
    if state.get("budget_exhausted"):
        return "END"
    """
    verifier_agent -> [safety | solver(retry) | hitl]

    For hint / formula_lookup intents, treat partially_correct as correct
    so we don't force a full retry just because a hint is incomplete.
    """
    verifier = state.get("verifier_output") or {}
    status   = verifier.get("status") or "incorrect"
    plan     = state.get("solution_plan") or {}
    intent   = plan.get("intent_type", "solve")

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


def _route_after_direct_response(state: AgentState) -> str:
    return "END" if state.get("budget_exhausted") else "safety_agent"


def _route_after_explainer(state: AgentState) -> str:
    return "END" if state.get("budget_exhausted") else "hitl_node"


def _run_tool_node(state: AgentState) -> dict:
    """Execute only the number of tool calls allowed by this request's budget."""
    messages = state.get("messages") or []
    last = messages[-1] if messages else None
    calls = getattr(last, "tool_calls", None) or []
    _, max_tools = budget_limits(state)
    used = int(state.get("tool_calls_used") or 0)
    if used + len(calls) > max_tools:
        return {
            "budget_exhausted": True,
            "final_response": "This request reached its configured tool-call budget. Please narrow the question or start a new request.",
        }
    result = ToolNode(SOLVER_TOOLS)(state)
    return {**result, "tool_calls_used": used + len(calls)}


def _route_after_safety(state: AgentState) -> str:
    """Route safe output to deterministic composition or directly to learning."""
    if state.get("budget_exhausted") or state.get("safety_passed") is False:
        return "END"
    intent = (state.get("solution_plan") or {}).get("intent_type", "solve")
    if intent in ("explain", "research", "generate", "formula_lookup"):
        return "record_learning"
    return "compose_node"


def _route_after_hitl(state: AgentState) -> str:
    """The graph suspends only for input, clarification, or verification."""
    hitl_type = state.get("hitl_type") or ""
    if hitl_type == "bad_input":
        return "detect_input"
    if hitl_type == "clarification":
        return "prefilter"
    if hitl_type == "verification":
        verifier = state.get("verifier_output") or {}
        return "safety_agent" if (verifier.get("status") or "") == "correct" else "solver_agent"
    return "prefilter"


def _retrieve_ltm_node(state: AgentState) -> dict:
    try:
        state = dict(state)
        if state.get("user_corrected_text") and not state.get("raw_text"):
            state["raw_text"] = state["user_corrected_text"]
        state["ltm_mode"] = "retrieve"
        plan = state.get("solution_plan") or {}
        parsed = state.get("parsed_data") or {}
        needs_web = bool(plan.get("needs_web"))
        query = str(parsed.get("problem_text") or plan.get("topic") or "").strip()[:500]
        _, max_tools = budget_limits(state)
        tool_calls_used = int(state.get("tool_calls_used") or 0)
        prefetch_web = bool(needs_web and query and tool_calls_used < max_tools)
        # Student memory retrieval and optional web retrieval are independent; run concurrently.
        thread_id = str(state.get("thread_id") or "")
        prefetch_notes = bool(thread_id and has_store(thread_id) and query)
        workers = 1 + int(prefetch_web) + int(prefetch_notes)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            memory_future = pool.submit(memory_manager_node, state)
            web_future = pool.submit(web_search_tool.invoke, {"query": query}) if prefetch_web else None
            notes_future = pool.submit(rag_tool.invoke, {"query": query, "thread_id": thread_id}) if prefetch_notes else None
            out = memory_future.result()
            try:
                web_context = web_future.result() if web_future else ""
            except Exception as exc:
                logger.warning("[Retrieval] Optional web prefetch failed: %s", type(exc).__name__)
                web_context = ""
            try:
                notes_context = notes_future.result() if notes_future else ""
            except Exception as exc:
                logger.warning("[Retrieval] Optional student-notes retrieval failed: %s", type(exc).__name__)
                notes_context = ""
        state["retrieved_context"] = "\n\n".join(part for part in (str(notes_context or ""), str(web_context or "")) if part)[:12000]
        if prefetch_web:
            state["tool_calls_used"] = tool_calls_used + 1

        # ── Build payload for activity panel ──────────────────────────────
        ltm = out.get("ltm_context") or {}
        eps = ltm.get("similar_episodes") or []
        weak = ltm.get("weak_topics") or {}
        strong = ltm.get("strong_topics") or {}
        patterns = ltm.get("mistake_patterns") or []
        best_strat = ltm.get("best_strategy")
        avg_att = ltm.get("avg_attempts")

        # Summarise episodes
        ep_lines = []
        for ep in eps[:3]:
            ep_lines.append(
                f"{ep.get('topic','?')} ({ep.get('difficulty','?')}) "
                f"→ {ep.get('outcome','?')} | ans: {ep.get('final_answer','?')}"
            )

        # Summarise weak/strong topics
        weak_str = ", ".join(
            f"{t}({c})" for t, c in weak.items() if c > 0
        ) or "none"
        strong_str = ", ".join(
            f"{t}({c})" for t, c in strong.items() if c > 0
        ) or "none"

        # Mistake patterns (top 2)
        pat_str = "; ".join(
            p.get("pattern", "")[:60] for p in patterns[:2]
        ) or "none"

        summary_parts = []
        if eps:
            summary_parts.append(f"{len(eps)} similar episode(s) found")
        if best_strat:
            summary_parts.append(f"best strategy: {best_strat}")
        if any(c > 0 for c in weak.values()):
            top_weak = max(weak, key=weak.get)
            summary_parts.append(f"weak: {top_weak}")
        summary = " | ".join(summary_parts) if summary_parts else "no prior history"

        payload(
            state, "retrieve_ltm",
            summary=summary,
            fields={
                "Similar episodes":  "\n".join(ep_lines) if ep_lines else "none",
                "Weak topics":       weak_str,
                "Strong topics":     strong_str,
                "Mistake patterns":  pat_str,
                "Best strategy":     f"{best_strat} (avg {avg_att:.1f} attempts)" if best_strat and avg_att else best_strat or "none",
                "Episodes retrieved": str(len(eps)),
            },
        )

        return {
            "ltm_mode": "retrieve",
            **out,
            "retrieved_context": state.get("retrieved_context") or "",
            "retrieved_chunks": {"student_notes": str(notes_context or "")[:6000], "web": str(web_context or "")[:6000]},
            "tool_calls_used": state.get("tool_calls_used") or 0,
            "agent_payload_log": state.get("agent_payload_log") or [],
        }
    except Exception as e:
        raise Agent_Exception(e, sys)


def _record_learning_node(state: AgentState) -> dict:
    try:
        state = dict(state)
        state["ltm_mode"] = "store"
        out = memory_manager_node(state)

        # ── Build payload for activity panel ──────────────────────────────
        parsed       = state.get("parsed_data") or {}
        plan         = state.get("solution_plan") or {}
        verifier_out = state.get("verifier_output") or {}
        solver_out   = state.get("solver_output") or {}
        evidence = learning_evidence(state)

        topic      = parsed.get("topic") or "—"
        difficulty = plan.get("difficulty") or "—"
        outcome    = evidence["outcome"]
        answer     = solver_out.get("final_answer") or "—"
        attempts   = state.get("solve_iterations") or 1

        payload(
            state, "store_ltm",
            summary=f"Stored | topic={topic} | outcome={outcome}",
            fields={
                "Topic":      topic,
                "Difficulty": difficulty,
                "Outcome":    outcome,
                "Answer":     answer[:80],
                "Attempts":   str(attempts),
                "Episodic":   "✓ saved",
                "Semantic":   "✓ updated",
                "Procedural": "✓ updated",
            },
        )

        return {
            "ltm_mode": "store",
            **out,
            "agent_payload_log": state.get("agent_payload_log") or [],
        }
    
    except Exception as e:
        raise Agent_Exception(e, sys)


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
    IntakeAgent,
    SolverAgent,
    VerifierAgent,
    SafetyAgent,

    DirectResponseAgent,   
    HITLAgent,
):
    def __init__(self):
        super().__init__()
        self.checkpointer = _build_checkpointer()
        graph = self._create_workflow()
        self.app = graph.compile(checkpointer=self.checkpointer)
        logger.info("[Graph] MathTutorWorkflow compiled successfully")

    def _route_intake(self, state: AgentState) -> str:
        if state.get("budget_exhausted"):
            return "END"
        target, reason = route_with_reason(state)
        logger.info("[RoutingPolicy] target=%s reason=%s", target, reason)
        return target

    def _create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        # Instrument actual wall-clock time for each node; timings are cumulative per request.
        graph.add_node("detect_input", timed_node("detect_input", detect_input_type))
        graph.add_node("ocr_node", timed_node("ocr_node", _ocr_node_with_confidence_gate))
        graph.add_node("asr_node", timed_node("asr_node", _asr_node_with_confidence_gate))
        graph.add_node("prefilter", timed_node("prefilter", prefilter_node))
        graph.add_node("intake", timed_node("intake", self.intake_node, model_node=True))
        graph.add_node("retrieve_ltm", timed_node("retrieve_ltm", _retrieve_ltm_node))

        graph.add_node("solver_agent", timed_node("solver_agent", self.solver_agent, model_node=True))
        graph.add_node("tool_node", timed_node("tool_node", _run_tool_node))
        graph.add_node("verifier_agent", timed_node("verifier_agent", self.verifier_agent, model_node=True))
        graph.add_node("safety_agent", timed_node("safety_agent", self.safety_agent, model_node=True))
        graph.add_node("compose_node", timed_node("compose_node", compose_node))
        graph.add_node("record_learning", timed_node("record_learning", _record_learning_node))
        graph.add_node("direct_response_node", timed_node("direct_response_node", self.direct_response_agent, model_node=True))
        graph.add_node("hitl_node", timed_node("hitl_node", self.hitl_node))
        graph.add_node("budget_stop_node", lambda state: {"budget_exhausted": True, "final_response": "This request reached its configured tool-call budget. Please narrow the question or start a new request."})
        graph.add_edge("budget_stop_node", END)

        # ── Entry ──────────────────────────────────────────────────────────────
        graph.set_entry_point("detect_input")

        # detect_input -> [ocr | asr | guardrail | hitl]
        graph.add_conditional_edges(
            "detect_input",
            _route_after_detect,
            {"ocr_node": "ocr_node", "asr_node": "asr_node",
             "prefilter": "prefilter", "hitl_node": "hitl_node"},
        )

        graph.add_conditional_edges(
            "ocr_node",
            _route_after_ocr,
            {"prefilter": "prefilter", "hitl_node": "hitl_node"},
        )
        graph.add_conditional_edges(
            "asr_node",
            _route_after_asr,
            {"prefilter": "prefilter", "hitl_node": "hitl_node"},
        )

        # Deterministic prefilter -> one structured intake call -> pure routing policy.
        graph.add_conditional_edges(
            "prefilter",
            lambda state: "intake" if state.get("prefilter_passed") is not False else "END",
            {"intake": "intake", "END": END},
        )
        graph.add_conditional_edges(
            "intake",
            self._route_intake,
            {"END": END, "hitl_node": "hitl_node", "direct_response_node": "direct_response_node",
             "solver_agent": "solver_agent", "retrieve_ltm": "retrieve_ltm"},
        )
        graph.add_edge("retrieve_ltm", "solver_agent")

        # direct_response_node -> safety_agent (same safety check as solve path)
        graph.add_conditional_edges("direct_response_node", _route_after_direct_response, {"safety_agent": "safety_agent", "END": END})

        # ReAct loop: solver <-> tools; final -> verifier
        graph.add_conditional_edges(
            "solver_agent",
            _route_solver_or_tools,
            {"tool_node": "tool_node", "verifier_agent": "verifier_agent", "budget_stop_node": "budget_stop_node", "END": END},
        )
        graph.add_conditional_edges("tool_node", lambda state: "END" if state.get("budget_exhausted") else "solver_agent", {"solver_agent": "solver_agent", "END": END})

        # verifier -> [safety | solver(retry) | hitl]
        graph.add_conditional_edges(
            "verifier_agent",
            _route_after_verifier,
            {"safety_agent":  "safety_agent",
             "solver_agent":  "solver_agent",
             "hitl_node":     "hitl_node",
             "END": END},
        )

        # Safety is deterministic-first; safe solve output is composed without another LLM.
        graph.add_conditional_edges(
            "safety_agent",
            _route_after_safety,
            {"compose_node": "compose_node", "record_learning": "record_learning", "END": END},
        )
        graph.add_edge("compose_node", "record_learning")
        graph.add_edge("record_learning", END)
     
        graph.add_conditional_edges(
            "hitl_node",
            _route_after_hitl,
            {
                "detect_input":         "detect_input",
                "prefilter":             "prefilter",
                "solver_agent":         "solver_agent",
                "safety_agent":         "safety_agent",
                "direct_response_node": "direct_response_node",
            },
        )


        return graph


workflow     = MathTutorWorkflow()
chatbot      = workflow.app
checkpointer = workflow.checkpointer