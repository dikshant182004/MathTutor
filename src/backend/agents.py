from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Annotated, Any, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool as lc_tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from backend.exceptions import Agent_Exception
from backend.logger import get_logger
from backend.tools.tools import rag_tool, web_search_tool, has_store
from backend.utils.artifacts import (
    ExplainerOutput,
    IntentRouterOutput,
    ParserOutput,
    SolverOutput,
    VerifierOutput,
)
from backend.utils.helper import MediaProcessor, python_calculator

load_dotenv()
logger = get_logger(__name__)
checkpointer = InMemorySaver()


def _get_secret(key: str, default: str = "") -> str:
    """
    Read a secret from st.secrets (Streamlit Cloud) first,
    then fall back to os.getenv / .env (local development).
    This lets the same code run unchanged in both environments.
    """
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


def python_calculator_tool(expression: str) -> dict:
    """
    Evaluate a Python math expression safely (no imports, no side effects).
    Use for numerical computation: arithmetic, trig, logarithms, factorials etc.
    Returns {expression, result} or {expression, error}.
    """
    return python_calculator(expression)


SOLVER_TOOLS = [rag_tool, web_search_tool, python_calculator_tool]
MAX_SOLVE_ITERATIONS = 3

class AgentState(TypedDict):
    input_mode:  str
    raw_text:    Optional[str]
    image_path:  Optional[str]
    audio_path:  Optional[str]
    thread_id:   Optional[str]

    ocr_text:        Optional[str]
    ocr_confidence:  Optional[float]
    transcript:      Optional[str]
    asr_confidence:  Optional[float]

    user_corrected_text: Optional[str]

    parsed_data:        Optional[dict]           # ParserOutput
    solution_plan:      Optional[dict]           # IntentRouterOutput
    retrieved_context:  Optional[str]
    solver_output:      Optional[dict]           # SolverOutput
    verifier_output:    Optional[dict]           # VerifierOutput
    explainer_output:   Optional[dict]           # ExplainerOutput
    solve_iterations:   int                      # solver↔verifier loop counter
    manim_video_path:   Optional[str]            # path returned by MCP server

    agent_payload_log: Optional[List[dict]]

    # ── Conversation log (for app.py history reload) ──────────────────────────
    # Stores prefixed HITL banner strings so they survive thread switching.
    # Written by app.py, read by _load_history. Never used by agent nodes.
    conversation_log: Optional[List[str]]

    # ── Final explainer response (plain markdown string) ──────────────────────
    # Written by explainer_agent. Read by app.py from the "updates" stream event
    # and yielded directly to the chat bubble.
    final_response: Optional[str]

    hitl_required: bool
    hitl_reason:   Optional[str]
    human_feedback: Optional[str]                # answer injected after interrupt

    messages: Annotated[list[BaseMessage], add_messages]


def _log_payload(state: dict, node: str, summary: str, fields: dict) -> None:
    """
    Append a structured payload summary to state["agent_payload_log"].
    Called by every agent after it writes its result so the activity panel
    in app.py can display what the agent decided / produced.

    fields: a flat dict of key → value to show — keep values short strings.
    """
    log: list = state.get("agent_payload_log") or []
    log.append({
        "node":    node,
        "summary": summary,
        "fields":  {k: str(v)[:200] for k, v in fields.items() if v not in (None, "", [], {})},
    })
    state["agent_payload_log"] = log

def retrieve_all_threads() -> list[str]:
    all_threads: set[str] = set()
    for checkpoint in checkpointer.list(None):
        tid = checkpoint.config["configurable"].get("thread_id")
        if tid:
            all_threads.add(tid)
    return list(all_threads)

def detect_input_type(state: AgentState) -> AgentState:
    try:
        if state.get("raw_text"):
            state["input_mode"] = "text"
        elif state.get("image_path"):
            state["input_mode"] = "image"
        elif state.get("audio_path"):
            state["input_mode"] = "audio"
        else:
            state["hitl_required"] = True
            state["hitl_reason"]   = "No valid input detected — please provide text, image, or audio."
        state.setdefault("solve_iterations", 0)
        return state
    except Exception as e:
        raise Agent_Exception(e, sys)


def ocr_node(state: AgentState) -> AgentState:
    try:
        processor = MediaProcessor()
        path = state.get("image_path")
        if not path:
            return state
        text, conf             = processor.process_image(path)
        state["ocr_text"]      = text
        state["ocr_confidence"] = conf
        logger.info(f"[OCR] conf={conf:.2f} text_len={len(text)}")
        return state
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        raise Agent_Exception(e, sys)


def asr_node(state: AgentState) -> AgentState:
    try:
        processor = MediaProcessor()
        path = state.get("audio_path")
        if not path:
            return state
        transcript, conf        = processor.process_audio(path)
        state["transcript"]     = transcript
        state["asr_confidence"] = conf
        logger.info(f"[ASR] conf={conf:.2f} transcript_len={len(transcript)}")
        return state
    except Exception as e:
        logger.error(f"ASR failed: {e}")
        raise Agent_Exception(e, sys)


def hitl_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop interrupt node.
    Builds a specific, context-rich question for the user based on WHY the
    interrupt was triggered (parser ambiguity, verifier uncertainty, iteration cap).
    """
    raw_reason   = state.get("hitl_reason") or ""   # guard against explicit None
    parsed       = state.get("parsed_data") or {}
    problem_text = parsed.get("problem_text") or ""
    verifier_out = state.get("verifier_output") or {}
    iteration    = state.get("solve_iterations", 0)

    # ── Build a specific, readable question depending on the context ──────────
    if "ambiguous" in raw_reason.lower() or "clarif" in raw_reason.lower():
        # Parser triggered — problem statement is unclear
        question = (
            f"I need a clarification before I can solve this.\n\n"
            f"**Problem I received:** {problem_text[:200] or 'Could not extract problem text.'}\n\n"
            f"**What I need from you:** {raw_reason}"
        )
    elif "iteration" in raw_reason.lower() or "attempt" in raw_reason.lower():
        # Solver hit max iterations without a correct answer
        last_fix = verifier_out.get("suggested_fix", "")
        question = (
            f"I tried solving this {iteration} times but could not produce a verified answer.\n\n"
            f"**Last verifier feedback:** {verifier_out.get('correctness_notes', 'Unknown issue')[:200]}\n\n"
            f"**Suggested fix I am stuck on:** {last_fix[:200] if last_fix else 'None'}\n\n"
            f"Could you guide me — is there additional context, a specific method you want used, "
            f"or a correction to the problem statement?"
        )
    elif verifier_out.get("status") == "needs_human":
        # Verifier flagged it for human review
        question = (
            f"I solved the problem but the verifier is not confident in the answer.\n\n"
            f"**Verifier assessment:** {verifier_out.get('correctness_notes', '')[:250]}\n\n"
            f"**Why human review is needed:** {verifier_out.get('hitl_reason', raw_reason)}\n\n"
            f"Could you verify if this approach is correct, or point out what is wrong?"
        )
    else:
        # Generic fallback — still include all available context
        question = (
            f"{raw_reason}\n\n"
            f"**Problem:** {problem_text[:200] or 'See above.'}\n\n"
            f"Please provide any additional information or correction."
        )

    logger.info(f"[HITL] Interrupting — question built for context: {raw_reason[:60]}")

    # ── If iteration cap triggered this HITL, reset solve pipeline NOW ────────
    # This must happen inside the node (not in the router) so LangGraph persists
    # the reset to the checkpoint before the graph pauses.
    if "iteration" in raw_reason.lower() or "attempt" in raw_reason.lower():
        state["solver_output"]    = None
        state["verifier_output"]  = None
        state["solve_iterations"] = 0
        state["messages"]         = []
        logger.info("[HITL] Iteration cap — solve state reset before interrupt")

    # interrupt() pauses the graph here. When resumed via Command(resume=value),
    # the return value of interrupt() IS the resume value directly.
    human_feedback = interrupt({
        "reason":   raw_reason,
        "question": question,
        "context": {
            "problem":    problem_text[:300],
            "iteration":  iteration,
            "verifier":   verifier_out.get("status", ""),
        },
    })

    state["human_feedback"] = str(human_feedback)
    state["hitl_required"]  = False
    state["hitl_reason"]    = None   # clear so _route_after_hitl gets clean state

    # IMPORTANT: append the clarification as a new HumanMessage rather than
    # replacing the whole messages list.  Replacing loses all prior tool call
    # history which the solver needs to continue reasoning after a mid-solve HITL.
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=f"[Human clarification]: {human_feedback}")
    ]
    return state


def satisfaction_hitl_node(state: AgentState) -> AgentState:
    """
    Simple post-solution satisfaction check.

    Pauses the graph with interrupt(). Resumed via Command(resume=value) from
    app.py — the return value of interrupt() IS the user's answer directly.

    Routing (handled by _route_after_satisfaction):
      "yes" / "y" / satisfied words → END
      anything else                 → parser_agent (re-explain with feedback)
    """
    logger.info("[SatisfactionHITL] Asking user for satisfaction")

    # interrupt() pauses here; Command(resume=value) sends value back as return
    human_feedback = interrupt({
        "reason":   "satisfaction_check",
        "question": "Are you satisfied with this explanation?\n\nClick **✅ Yes** to ask your next question, or **🔄 No** to re-explain.",
    })

    state["human_feedback"] = str(human_feedback).strip()
    return state


class BaseAgent:

    def __init__(self):

        key = _get_secret("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets.")
        self.llm = ChatGroq(
            api_key=key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
        )
        self.media_processor = MediaProcessor()
        self.workflow = self._create_workflow()
        # NOTE: Do NOT use interrupt_before here.
        # Both hitl_node and satisfaction_hitl_node call interrupt() internally.
        # interrupt_before + interrupt() on the same node causes an infinite
        # re-interrupt loop because the node never gets to clear hitl_required.
        self.app = self.workflow.compile(checkpointer=checkpointer)


class ParserAgent(BaseAgent):

    def parser_agent(self, state: AgentState) -> AgentState:
        try:
            raw_text = (
                state.get("user_corrected_text")
                or state.get("ocr_text")
                or state.get("transcript")
                or state.get("raw_text")
            )

            if not raw_text:
                state["hitl_required"] = True
                state["hitl_reason"]   = "No question text could be extracted from the input."
                return state

            prompt = f"""You are a math problem parser for a JEE-level math tutor.
                Tasks:
                1. Clean OCR/ASR noise and artefacts from the input text.
                2. Normalise mathematical notation (fractions, exponents, integrals, etc.).
                3. Identify all variables and constraints explicitly stated.
                4. Set needs_clarification=true if the problem is genuinely ambiguous or incomplete.
                5. List any OCR corrections applied.

                Allowed topics: algebra | probability | calculus | linear_algebra |
                                geometry | trigonometry | statistics | number_theory

                Input text:
                {raw_text}"""

            structured_llm = self.llm.with_structured_output(ParserOutput)
            parsed: ParserOutput = structured_llm.invoke([HumanMessage(content=prompt)])

            state["parsed_data"] = parsed.model_dump()

            if parsed.needs_clarification:
                state["hitl_required"] = True
                state["hitl_reason"]   = parsed.clarification_reason or "Problem is ambiguous."

            _log_payload(state, "parser_agent",
                summary=f"Topic: {parsed.topic or '?'} | Confidence: {parsed.confidence_score:.0%}",
                fields={
                    "Problem":        parsed.problem_text[:120],
                    "Topic":          parsed.topic,
                    "Variables":      ", ".join(parsed.variables) if parsed.variables else None,
                    "Constraints":    ", ".join(parsed.constraints) if parsed.constraints else None,
                    "Clarification?": parsed.clarification_reason if parsed.needs_clarification else None,
                    "OCR fixes":      ", ".join(parsed.ocr_corrections) if parsed.ocr_corrections else None,
                    "Confidence":     f"{parsed.confidence_score:.0%}",
                },
            )

            logger.info(f"[Parser] topic={parsed.topic} needs_clarification={parsed.needs_clarification}")
            return state

        except Exception as e:
            logger.error(f"[Parser] failed: {e}")
            raise Agent_Exception(e, sys)

class IntentRouterAgent(BaseAgent):

    def intent_router_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            context_snippet = (state.get("retrieved_context") or "")[:500]

            prompt = f"""You are an intent router for a JEE math mentor system.

                Classify the problem below and decide how to solve it.
                                
                IMPORTANT: needs_visualization and requires_calculator MUST be JSON booleans
                (the literal values true or false, never the strings "true" or "false").

                Fields to return:
                topic            — algebra | probability | calculus | linear_algebra | geometry | trigonometry | statistics | number_theory
                difficulty       — easy | medium | hard
                solver_strategy  — 1-2 sentence description of the optimal solution approach
                needs_visualization — Set true if ANY of the following apply:
                    • 3D geometry: spheres, cones, cylinders, planes, lines in space, distance/angle in 3D
                    • 2D coordinate geometry: circles, parabolas, ellipses, hyperbolas, tangent/normal lines
                    • Graph sketching: functions, curves, regions of integration
                    • Vectors: cross product, angle between vectors, projection
                    • Transformations: rotations, reflections, shear in 2D/3D
                    • Trigonometry: unit circle, sinusoidal graphs, phase shift
                    • Calculus concepts: area under curve, solids of revolution, limits graphically
                    • Probability: tree diagrams, Venn diagrams, geometric probability
                    Set false for pure algebra, number theory, or problems solved entirely with equations.
                visualization_hint  — Concise description of what the Manim animation should show (required when needs_visualization=true, else null).
                    Examples: "Draw sphere with inscribed cone, label r and h, animate volume formula"
                             "Plot parabola y=x² and tangent at x=2, shade region"
                             "Show unit circle, animate angle θ, trace sin/cos values"
                requires_calculator — true if numerical computation is needed

                Context (first 500 chars):
                {context_snippet}

                Problem:
                {problem_text}"""

            structured_llm = self.llm.with_structured_output(IntentRouterOutput)
            result: IntentRouterOutput = structured_llm.invoke([HumanMessage(content=prompt)])

            state["solution_plan"] = result.model_dump()
            _log_payload(state, "intent_router",
                summary=f"{result.topic.title()} | {result.difficulty.title()} | {'Viz needed' if result.needs_visualization else 'No viz'}",
                fields={
                    "Topic":      result.topic,
                    "Difficulty": result.difficulty,
                    "Strategy":   result.solver_strategy[:150],
                    "Needs viz":  str(result.needs_visualization),
                    "Needs calc": str(result.requires_calculator),
                    "Viz hint":   result.visualization_hint,
                },
            )
            logger.info(f"[Router] topic={result.topic} difficulty={result.difficulty} viz={result.needs_visualization}")
            return state

        except Exception as e:
            logger.error(f"[Router] failed: {e}")
            raise Agent_Exception(e, sys)

class SolverAgent(BaseAgent):
    
    def _build_solver_llm(self):
        # tool_choice="auto" lets the model decide when to call tools.
        # Without it, Groq sometimes ignores tool schema and writes inline
        # <function=...> tags which cause 400 tool_use_failed errors.
        return self.llm.bind_tools(SOLVER_TOOLS, tool_choice="auto")
    
    def solver_agent(self, state: AgentState) -> AgentState:
        """
        Single turn of the ReAct loop.
        Appends the LLM response (which may contain tool_calls) to state['messages'].
        """
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            plan  = state.get("solution_plan") or {}
            strategy = plan.get("solver_strategy", "")
            prev_verifier = state.get("verifier_output") or {}
            human_fb = state.get("human_feedback") or ""
            iteration = state.get("solve_iterations", 0)
            thread_id = state.get("thread_id") or ""

            feedback_block = ""
            if prev_verifier.get("suggested_fix"):
                feedback_block = f"\n\nPrevious attempt was WRONG. Verifier feedback:\n{prev_verifier['suggested_fix']}"

            existing_messages = state.get("messages") or []

            # Include human_feedback in the system prompt ONLY on a fresh start
            # (no existing messages). When hitl_node appended a HumanMessage with
            # the clarification, it's already in existing_messages — adding it to
            # the system prompt too would duplicate the feedback for the LLM.
            if human_fb and not existing_messages:
                feedback_block += f"\n\nHuman tutor feedback:\n{human_fb}"

            # Tell the LLM exactly which tools are available based on what
            # is actually indexed — avoids wasted rag_tool calls when no PDF
            # was uploaded, and avoids skipping rag_tool when one was.
            doc_uploaded = has_store(thread_id)

            if doc_uploaded:
                rag_instruction = (
                    f"- rag_tool(query, thread_id): searches the student's uploaded study material.\n"
                    f"  * A PDF IS indexed for this session — call this FIRST for any topic that\n"
                    f"    could appear in the student's notes or textbook.\n"
                    f"  * Always pass thread_id=\"{thread_id}\".\n"
                    f"  * Use a focused query (e.g. \"Bayes theorem formula\") not the full problem.\n"
                    f"  * If it returns empty context, fall back to web_search_tool."
                )
            else:
                rag_instruction = (
                    "- rag_tool: NOT available — no PDF has been uploaded for this session.\n"
                    "  * Do NOT call rag_tool at all."
                )

            system_prompt = f"""You are an expert JEE mathematics solver (attempt {iteration + 1}).
Solving strategy: {strategy}
{feedback_block}

CRITICAL TOOL-USE RULES — read carefully:
1. Each response must be EITHER a tool call OR written reasoning. NEVER mix them.
2. If you need a tool: output ONLY the tool call, nothing else — no prose before or after.
3. If you are writing the solution: output ONLY text — do NOT embed tool calls inside text.
4. NEVER write <function=...> tags in your text. Use the structured tool_calls mechanism only.
5. Decide at the start of each turn: do I need a tool? If yes → call it. If no → write solution.

TOOLS available:
{rag_instruction}
- web_search_tool(query)             : search the web for formulas, theory, or examples.
                                       {"Use if rag_tool found nothing." if doc_uploaded else "Use when you need a formula or theorem."}
- python_calculator_tool(expression) : evaluate any Python math expression numerically.
                                       Use for ALL arithmetic — never compute mentally.

WORKFLOW:
{"Step 1 — call rag_tool with a focused query (e.g. 'cross product formula')." if doc_uploaded else "Step 1 — call web_search_tool if you need a formula or theorem."}
{"Step 2 — if rag_tool empty, call web_search_tool." if doc_uploaded else "Step 2 — call python_calculator_tool for all numerical computation."}
Step {"3" if doc_uploaded else "3"} — call python_calculator_tool for all numerical computation.
Step {"4" if doc_uploaded else "4"} — write the complete step-by-step solution with every formula and substitution shown.
Step {"5" if doc_uploaded else "4"} — end with a clearly labelled FINAL ANSWER on its own line."""

            # Build a clean message list for the solver.
            if not existing_messages:
                # First call: fresh problem
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
                ]
            else:
                # Subsequent calls (tool results / retry): keep full history but
                # ensure the system prompt is always at position 0.
                if isinstance(existing_messages[0], SystemMessage):
                    messages = [SystemMessage(content=system_prompt)] + list(existing_messages[1:])
                else:
                    messages = [SystemMessage(content=system_prompt)] + list(existing_messages)

            solver_llm = self._build_solver_llm()

            # Retry once on Groq 400 tool_use_failed — this happens when the model
            # outputs a malformed inline tool call tag. On retry we strip the last
            # assistant message (the malformed one) and re-invoke.
            try:
                response = solver_llm.invoke(messages)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if "tool_use_failed" in err_str or "400" in err_str:
                    logger.warning(f"[Solver] tool_use_failed on first attempt, retrying with fresh message list | err={err_str[:120]}")
                    # Retry with only system + problem — drop any prior context that
                    # confused the model into mixing prose and tool calls.
                    retry_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=f"Solve this problem step by step:\n\n{problem_text}"),
                    ]
                    response = solver_llm.invoke(retry_messages)
                else:
                    raise

            # Increment iteration counter only when the LLM gives a final answer
            # (i.e. no tool_calls — it has finished reasoning)
            updates: dict = {"messages": [response]}
            
            has_tool_calls = bool(getattr(response, "tool_calls", None))

            if not has_tool_calls:
                # ── Final answer turn — no more tools needed ───────────────────
                solution_text = response.content or ""

                # Parse FINAL ANSWER section if present
                if "FINAL ANSWER" in solution_text.upper():
                    final_answer = solution_text.upper().split("FINAL ANSWER")[-1]
                    # Strip the label itself (handles "FINAL ANSWER:" etc.)
                    final_answer = solution_text.split(
                        solution_text[solution_text.upper().find("FINAL ANSWER"):].split()[0]
                        + " "
                    )[-1].strip()
                else:
                    # Fall back to last non-empty line
                    lines = [l.strip() for l in solution_text.splitlines() if l.strip()]
                    final_answer = lines[-1] if lines else solution_text

                calc_used = any(
                    isinstance(m, ToolMessage) and "calculator" in (m.name or "").lower()
                    for m in messages
                )
                rag_used = any(
                    isinstance(m, ToolMessage) and "rag" in (m.name or "").lower()
                    for m in messages
                )

                updates["solve_iterations"] = iteration + 1
                updates["solver_output"] = {
                    "solution":         solution_text,
                    "steps":            [],
                    "final_answer":     final_answer,
                    "formulas_used":    [],
                    "rag_context_used": rag_used,
                    "calculator_used":  calc_used,
                    "confidence_score": 0.85,
                }
                _log_payload(state, "solver_agent",
                    summary=f"Answer produced | iter {iteration + 1} | calc={calc_used}",
                    fields={
                        "Final answer": final_answer[:120] if final_answer else solution_text[:120],
                        "RAG used":     str(rag_used),
                        "Calc used":    str(calc_used),
                        "Iteration":    str(iteration + 1),
                    },
                )
                # Also store in agent_payload_log via state update
                updates["agent_payload_log"] = state.get("agent_payload_log") or []
                logger.info(f"[Solver] Final answer produced | iter={iteration+1}")
            else:
                logger.info(f"[Solver] Tool calls requested: {[tc['name'] for tc in response.tool_calls]}")

            return updates

        except Exception as e:
            logger.error(f"[Solver] failed: {e}")
            raise Agent_Exception(e, sys)

class VerifierAgent(BaseAgent):

    def verifier_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            solver_out = state.get("solver_output") or {}
            solution = solver_out.get("solution", "")
            final_answer = solver_out.get("final_answer", "")
            iteration = state.get("solve_iterations", 1)

            prompt = f"""You are a strict mathematical verifier and critic for JEE-level problems.

                Your job:
                1. Check EVERY step of the solution for correctness.
                2. Verify units and domain / range validity.
                3. Check for missed edge cases (division by zero, undefined log, etc.).
                4. If the solution is wrong, provide a concrete suggested_fix.
                5. Set status = "needs_human" ONLY if you yourself are genuinely unsure (low confidence).
                6. Set status = "correct" only when fully satisfied.

                Return:
                status             — correct | incorrect | partially_correct | needs_human
                is_correct         — bool
                correctness_notes  — detailed assessment
                unit_domain_check  — units and domain/range validity
                edge_case_notes    — edge cases considered or missed
                suggested_fix      — concrete fix (null if correct)
                hitl_reason        — why human review needed (null otherwise)
                confidence_score   — your confidence 0.0–1.0

                Problem:
                {problem_text}

                Solution to verify (iteration {iteration}):
                {solution}

                Final answer claimed: {final_answer}"""

            structured_llm = self.llm.with_structured_output(VerifierOutput)
            result: VerifierOutput = structured_llm.invoke([HumanMessage(content=prompt)])

            state["verifier_output"] = result.model_dump()

            if result.status == "needs_human":
                state["hitl_required"] = True
                state["hitl_reason"]   = result.hitl_reason or "Verifier confidence too low — human review needed."

            _log_payload(state, "verifier_agent",
                summary=f"Status: {result.status.upper()} | Confidence: {result.confidence_score:.0%}",
                fields={
                    "Status":      result.status,
                    "Correct?":    str(result.is_correct),
                    "Assessment":  result.correctness_notes[:150],
                    "Domain check":result.unit_domain_check[:100],
                    "Edge cases":  result.edge_case_notes[:100],
                    "Fix needed":  result.suggested_fix[:120] if result.suggested_fix else None,
                    "HITL reason": result.hitl_reason,
                    "Confidence":  f"{result.confidence_score:.0%}",
                },
            )

            logger.info(f"[Verifier] status={result.status} confidence={result.confidence_score:.2f} iter={iteration}")
            return state

        except Exception as e:
            logger.error(f"[Verifier] failed: {e}")
            raise Agent_Exception(e, sys)

class ExplainerAgent(BaseAgent):

    def explainer_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            solver_out = state.get("solver_output") or {}
            solution = solver_out.get("solution", "")
            plan = state.get("solution_plan") or {}
            needs_viz = plan.get("needs_visualization", False)
            viz_hint = plan.get("visualization_hint", "")

            # ── Step 1: Structured explanation (NO manim code here) ───────────
            # Manim code is generated in a separate plain call below.
            # Reason: Groq's function-calling schema validation fails with
            # 'tool_use_failed' (Error 400) when manim_scene_code contains
            # long multi-line Python strings inside structured output.
            prompt = f"""You are a patient, encouraging JEE math tutor explaining to a student.

                Your job — produce a COMPLETE, mathematically rigorous explanation:

                1. step_by_step: A list of numbered steps. EACH step MUST contain:
                   - A plain-English description of what we are doing and WHY
                   - The actual mathematical working: formulas, substitutions, calculations, algebra
                   - The intermediate result obtained at that step
                   Example of a good step: "Apply Bayes' theorem: P(A|B) = P(B|A)·P(A)/P(B) = (3/4·1/3)/(1/2) = 1/2"
                   DO NOT just write "Calculate the probability" — show the numbers and algebra.

                2. explanation: A one-paragraph conceptual summary of the approach used.

                3. key_concepts: 3–5 core mathematical concepts/theorems applied.

                4. common_mistakes: 2–3 mistakes students typically make on this problem type.

                5. difficulty_rating: easy | medium | hard

                Problem:
                {problem_text}

                Full verified solution (use this as your source of truth for all numbers):
                {solution}"""

            structured_llm = self.llm.with_structured_output(ExplainerOutput)
            result: ExplainerOutput = structured_llm.invoke([HumanMessage(content=prompt)])

            # ── Step 2: Generate Manim code separately (plain text call) ──────
            # Only requested when needs_viz=True. A plain non-structured call
            # avoids the Groq 400 'tool_use_failed' error caused by large code
            # strings being jammed into a function-calling schema.
            manim_scene_code = None
            manim_scene_description = None
            if needs_viz:
                manim_prompt = f"""Generate complete, runnable Manim Community Edition Python code for a Scene that visualises the following math solution.

The animation should: {viz_hint or "illustrate the key mathematical concept step by step"}.

Requirements:
- Import from manim (not manimlib): from manim import *
- Define exactly ONE class that extends Scene
- Be fully self-contained (no external files, no user input)
- Use MathTex for all equations
- Use ThreeDScene if the problem involves 3D geometry

Problem: {problem_text}

Solution summary: {solution[:800]}

Reply with ONLY the Python code. No explanation, no markdown fences, no preamble."""

                plain_llm = self.llm  # plain ChatGroq, no bind_tools/structured_output
                manim_response = plain_llm.invoke([HumanMessage(content=manim_prompt)])
                raw_code = (manim_response.content or "").strip()
                # Strip accidental markdown fences if model adds them
                if raw_code.startswith("```"):
                    raw_code = "\n".join(
                        line for line in raw_code.splitlines()
                        if not line.strip().startswith("```")
                    ).strip()
                if raw_code and "class" in raw_code and "Scene" in raw_code:
                    manim_scene_code = raw_code
                    manim_scene_description = (
                        f"Animation visualising: {viz_hint or problem_text[:120]}"
                    )
                    logger.info(f"[Explainer] Manim code generated ({len(raw_code)} chars)")
                else:
                    logger.warning("[Explainer] Manim code generation returned unusable output")

            # Merge manim fields into the explainer output dict
            explainer_dict = result.model_dump()
            explainer_dict["manim_scene_code"] = manim_scene_code
            explainer_dict["manim_scene_description"] = manim_scene_description
            state["explainer_output"] = explainer_dict

            md_parts: list[str] = []
            md_parts.append(f"## 📘 Solution")
            md_parts.append(f"**Problem:** {problem_text}\n")
            md_parts.append("---")

            # ── Conceptual overview ───────────────────────────────────────────
            if result.explanation:
                md_parts.append(f"**Overview:** {result.explanation}\n")

            # ── Step-by-step with full mathematical working ───────────────────
            if result.step_by_step:
                md_parts.append("### Step-by-Step Working\n")
                for i, step in enumerate(result.step_by_step, 1):
                    md_parts.append(f"**Step {i}.** {step}\n")

            # ── Final answer ──────────────────────────────────────────────────
            final = solver_out.get("final_answer", "")
            if final:
                md_parts.append("---")
                md_parts.append(f"### ✅ Final Answer\n\n> {final}\n")

            # ── Key concepts ──────────────────────────────────────────────────
            if result.key_concepts:
                md_parts.append("---")
                md_parts.append("### 💡 Key Concepts\n")
                for c in result.key_concepts:
                    md_parts.append(f"- {c}")
                md_parts.append("")

            # ── Common mistakes ───────────────────────────────────────────────
            if result.common_mistakes:
                md_parts.append("### ⚠️ Common Mistakes to Avoid\n")
                for m in result.common_mistakes:
                    md_parts.append(f"- {m}")
                md_parts.append("")

            rich_md = "\n".join(md_parts)

            # Store ONLY in final_response — NOT in state["messages"].
            # app.py streams final_response directly to the chat bubble.
            # Writing to state["messages"] as well causes _load_history to
            # show the same content twice (once from messages, once from
            # final_response).
            state["final_response"] = rich_md
            # Keep messages clean — store a minimal marker so history reload
            # knows a solution was produced, but _load_history skips it
            # because it starts with "## 📘 Solution" (filtered in _load_history).
            from langchain_core.messages import AIMessage as _AIMsg
            state["messages"] = [_AIMsg(content=rich_md)]

            _log_payload(state, "explainer_agent",
                summary=f"{len(result.step_by_step)} steps | difficulty: {result.difficulty_rating}",
                fields={
                    "Steps":           str(len(result.step_by_step)),
                    "Key concepts":    ", ".join(result.key_concepts[:3]) if result.key_concepts else None,
                    "Common mistakes": ", ".join(result.common_mistakes[:2]) if result.common_mistakes else None,
                    "Manim code":      "Yes — scene generated" if manim_scene_code else None,
                    "Difficulty":      result.difficulty_rating,
                },
            )
            logger.info(f"[Explainer] steps={len(result.step_by_step)} manim={bool(manim_scene_code)}")
            return state

        except Exception as e:
            logger.error(f"[Explainer] failed: {e}")
            raise Agent_Exception(e, sys)

async def _call_manim_mcp(scene_code: str, scene_class: str) -> Optional[str]:
    """
    Calls the FastMCP Manim server via streamable-http transport.
    Server must be running:  python manim_mcp_server.py
    Env vars:
        MANIM_MCP_SERVER_URL  — default http://localhost:8765/mcp
    Returns absolute path to rendered video, or None on failure.

    FastMCP client.call_tool() returns a CallToolResult with:
      .data    — hydrated Python dict (preferred, populated when server returns a dict)
      .content — list of TextContent / ImageContent blocks (fallback)
      .is_error — bool
    """
    try:
        from fastmcp import Client as FastMCPClient
        import json as _json

        server_url = _get_secret("MANIM_MCP_SERVER_URL", "http://localhost:8765/mcp")
        logger.info(f"[Manim MCP] Connecting to {server_url}")

        async with FastMCPClient(server_url) as client:
            result = await client.call_tool(
                "render_manim_scene",
                {
                    "scene_code":  scene_code,
                    "scene_class": scene_class,
                    "quality":     "medium_quality",
                    "fmt":         "mp4",
                },
            )

            if getattr(result, "is_error", False):
                err = getattr(result, "data", None) or {}
                logger.error(f"[Manim MCP] Tool returned error: {err.get('error', 'unknown')}")
                return None

            # Path 1: .data is already a hydrated dict (FastMCP auto-deserialises dicts)
            payload = getattr(result, "data", None)
            if isinstance(payload, dict):
                video_path = payload.get("video_path") or payload.get("output_path")
                if video_path:
                    logger.info(f"[Manim MCP] Rendered (via .data) → {video_path}")
                    return str(video_path)

            # Path 2: .content is a list of TextContent blocks — parse JSON from .text
            for item in (getattr(result, "content", None) or []):
                text = getattr(item, "text", None)
                if not text:
                    continue
                try:
                    payload = _json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    video_path = payload.get("video_path") or payload.get("output_path")
                    if video_path:
                        logger.info(f"[Manim MCP] Rendered (via .content) → {video_path}")
                        return str(video_path)

        logger.warning("[Manim MCP] No video_path found in response")
        return None

    except ImportError:
        logger.warning("[Manim MCP] `fastmcp` not installed — pip install fastmcp")
        return None
    except Exception as exc:
        logger.error(f"[Manim MCP] Error: {exc}")
        return None

def _extract_scene_class(code: str) -> str:
    """Extract the first Scene subclass name from generated Manim code."""
    import re
    match = re.search(r"class\s+(\w+)\s*\(.*Scene.*\)", code)
    return match.group(1) if match else "MathScene"


def manim_node(state: AgentState) -> AgentState:
    """
    Reads manim_scene_code from explainer_output.
    Calls the Manim MCP server to render it.
    Stores the output path in state['manim_video_path'].
    Gracefully skips if no code was generated or MCP is unavailable.
    """
    explainer = state.get("explainer_output") or {}
    scene_code = explainer.get("manim_scene_code")

    if not scene_code:
        logger.info("[Manim] No scene code — skipping.")
        return state

    scene_class = _extract_scene_class(scene_code)
    logger.info(f"[Manim] Rendering scene class: {scene_class}")

    # asyncio.run() raises RuntimeError if an event loop is already running
    # (which Streamlit does). nest_asyncio patches the loop to allow nesting.
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        logger.warning("[Manim] nest_asyncio not installed — pip install nest_asyncio")

    try:
        loop = asyncio.get_event_loop()
        video_path = loop.run_until_complete(_call_manim_mcp(scene_code, scene_class))
    except RuntimeError:
        # Last resort: new loop
        loop = asyncio.new_event_loop()
        try:
            video_path = loop.run_until_complete(_call_manim_mcp(scene_code, scene_class))
        finally:
            loop.close()

    state["manim_video_path"] = video_path

    if video_path:
        logger.info(f"[Manim] Video saved at: {video_path}")
    else:
        logger.warning("[Manim] Render failed or MCP unavailable — continuing without visualization.")

    return state

def _route_after_detect(state: AgentState) -> str:
    """Route to OCR / ASR / parser based on input mode."""
    mode = state.get("input_mode") or ""
    if state.get("hitl_required"):
        return "hitl_node"
    return {"image": "ocr_node", "audio": "asr_node"}.get(mode, "parser_agent")


def _route_after_parser(state: AgentState) -> str:
    """After parsing: HITL if clarification needed, else intent_router."""
    if state.get("hitl_required"):
        return "hitl_node"
    return "intent_router"


def _route_solver_or_tools(state: AgentState) -> str:
    """
    Core ReAct routing after solver_agent runs.
    - If the LLM emitted tool_calls  → tool_node  (execute the tools)
    - If no tool_calls               → verifier_agent (solution is complete)
    """
    messages = state.get("messages") or []
    last_msg = messages[-1] if messages else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        tool_names = [tc["name"] for tc in last_msg.tool_calls]
        logger.info(f"[Router] Solver requested tools: {tool_names} -> tool_node")
        return "tool_node"
    logger.info("[Router] Solver finished reasoning -> verifier_agent")
    return "verifier_agent"


def _route_after_verifier(state: AgentState) -> str:
    """
    After verification:
      correct           -> explainer_agent
      incorrect/partial -> solver_agent (ReAct loop restarts)
      needs_human       -> hitl_node
      iteration cap     -> hitl_node
    """
    verifier  = state.get("verifier_output") or {}
    status    = verifier.get("status", "incorrect")
    iteration = state.get("solve_iterations", 0)

    if status == "correct":
        # Clear any stale HITL flags before moving to explainer
        state["hitl_required"] = False
        state["hitl_reason"]   = None
        return "explainer_agent"

    if status == "needs_human":
        state["hitl_required"] = True
        state["hitl_reason"]   = (
            verifier.get("hitl_reason")
            or verifier.get("correctness_notes")
            or "Verifier confidence too low — human review needed."
        )
        return "hitl_node"

    if iteration >= MAX_SOLVE_ITERATIONS:
        state["hitl_required"] = True
        state["hitl_reason"]   = (
            f"Solution not verified after {MAX_SOLVE_ITERATIONS} attempts. "
            "Human tutor review required."
        )
        return "hitl_node"

    # incorrect / partially_correct — retry solver.
    # Clear any stale hitl_required so the next verifier pass starts clean.
    state["hitl_required"] = False
    state["hitl_reason"]   = None
    return "solver_agent"



def _route_after_hitl(state: AgentState) -> str:
    """
    After clarification HITL resumes.

    Decision tree:
    1. No solver output yet (parser triggered HITL)     → parser_agent
    2. Iteration cap hit (solver exhausted retries)     → parser_agent
       NOTE: state reset (solver_output, messages etc.) was already done
             inside hitl_node before the interrupt() call for this case.
    3. Verifier sent us here (mid-solve, needs_human)   → solver_agent
    4. Fallback                                         → parser_agent

    IMPORTANT: This is a LangGraph conditional edge function.
    Do NOT mutate state here — mutations are silently discarded by LangGraph.
    All state writes must happen inside node functions.
    """
    solver_out  = state.get("solver_output")
    verifier    = state.get("verifier_output") or {}
    hitl_reason = state.get("hitl_reason") or ""

    # Case 1: parser triggered — no solving has happened yet
    if not solver_out:
        logger.info("[HITL route] No solver output → parser_agent")
        return "parser_agent"

    # Case 2: iteration cap — go back to parser for a fresh start.
    # hitl_node already cleared solver_output/messages/iterations in state
    # (returned as part of its node output) before calling interrupt().
    if "iteration" in hitl_reason.lower() or "attempt" in hitl_reason.lower():
        logger.info("[HITL route] Iteration cap → parser_agent (fresh start)")
        return "parser_agent"

    # Case 3: verifier flagged for human review — resume solver with feedback
    if verifier.get("status") in ("incorrect", "partially_correct", "needs_human"):
        logger.info("[HITL route] Verifier sent here → solver_agent (resume)")
        return "solver_agent"

    # Case 4: fallback
    logger.info("[HITL route] Fallback → parser_agent")
    return "parser_agent"


# Words that mean the user is satisfied — checked as startswith on lowercased input
_SATISFIED_WORDS = ("yes", "y ", "yep", "yup", "satisfied", "ok", "okay",
                    "sure", "done", "correct", "great", "good", "fine", "next",
                    "perfect", "understood", "got it", "thanks", "thank")


def _is_satisfied(feedback: str) -> bool:
    """Return True if the feedback string means the user is satisfied."""
    fb = feedback.strip().lower()
    # Exact single-word matches
    if fb in {"yes", "y", "ok", "sure", "done", "next", "fine", "great",
               "good", "perfect", "yep", "yup"}:
        return True
    # Startswith for multi-word phrases
    return any(fb.startswith(w) for w in _SATISFIED_WORDS)


def _route_after_satisfaction(state: AgentState) -> str:
    """
    After the satisfaction check:
      - satisfied (yes/y/ok/…) → END
      - anything else           → parser_agent (re-parse / re-explain with feedback)
    Pipeline fields are reset in both branches so the next turn starts clean.
    """
    feedback = (state.get("human_feedback") or "").strip()
    _reset = {
        "parsed_data": None, "solution_plan": None,
        "solver_output": None, "verifier_output": None,
        "explainer_output": None, "final_response": None,
        "solve_iterations": 0, "hitl_required": False,
        "hitl_reason": None, "human_feedback": None,
        "retrieved_context": None, "manim_video_path": None,
        "messages": [],
    }
    if _is_satisfied(feedback):
        logger.info("[SatisfactionHITL] User satisfied → END")
        state.update(_reset)
        return "END"
    logger.info(f"[SatisfactionHITL] Not satisfied ({feedback!r}) → parser_agent")
    # Keep raw_text pointing at the original question so parser re-parses it
    # with the new feedback injected as human_feedback (already in state).
    state.update(_reset)
    return "parser_agent"


class MathTutorWorkflow(
    ParserAgent,
    IntentRouterAgent,
    SolverAgent,
    VerifierAgent,
    ExplainerAgent,
):
    def _create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        graph.add_node("detect_input",    detect_input_type)
        graph.add_node("ocr_node",        ocr_node)
        graph.add_node("asr_node",        asr_node)
        graph.add_node("parser_agent",    self.parser_agent)
        graph.add_node("intent_router",   self.intent_router_agent)
        graph.add_node("solver_agent",    self.solver_agent)
        graph.add_node("tool_node",       ToolNode(SOLVER_TOOLS))  # <-- ReAct tool executor
        graph.add_node("verifier_agent",  self.verifier_agent)
        graph.add_node("hitl_node",       hitl_node)
        graph.add_node("explainer_agent", self.explainer_agent)
        graph.add_node("manim_node",      manim_node)
        graph.add_node("satisfaction_hitl",   satisfaction_hitl_node)

        # ── Entry ──────────────────────────────────────────────────────────────
        graph.set_entry_point("detect_input")

        # ── detect_input -> [ocr | asr | parser | hitl] ───────────────────────
        graph.add_conditional_edges(
            "detect_input",
            _route_after_detect,
            {
                "ocr_node":     "ocr_node",
                "asr_node":     "asr_node",
                "parser_agent": "parser_agent",
                "hitl_node":    "hitl_node",
            },
        )

        # ── ocr / asr -> parser ───────────────────────────────────────────────
        graph.add_edge("ocr_node", "parser_agent")
        graph.add_edge("asr_node", "parser_agent")

        # ── parser -> [hitl | intent_router] ─────────────────────────────────
        graph.add_conditional_edges(
            "parser_agent",
            _route_after_parser,
            {
                "hitl_node":    "hitl_node",
                "intent_router": "intent_router",
            },
        )

        # ── intent_router -> solver_agent (kick off ReAct loop) ───────────────
        graph.add_edge("intent_router", "solver_agent")

        # ── ReAct loop: solver_agent <-> tool_node ────────────────────────────
        #    solver emits tool_calls  -> tool_node executes them -> back to solver
        #    solver emits final answer (no tool_calls) -> verifier_agent
        graph.add_conditional_edges(
            "solver_agent",
            _route_solver_or_tools,
            {
                "tool_node":      "tool_node",
                "verifier_agent": "verifier_agent",
            },
        )
        # tool_node always returns to solver so it can reason over results
        graph.add_edge("tool_node", "solver_agent")

        # ── verifier -> [explainer | solver(retry) | hitl] ───────────────────
        graph.add_conditional_edges(
            "verifier_agent",
            _route_after_verifier,
            {
                "explainer_agent": "explainer_agent",
                "solver_agent":    "solver_agent",
                "hitl_node":       "hitl_node",
            },
        )

        # ── hitl -> [parser | solver] ─────────────────────────────────────────
        graph.add_conditional_edges(
            "hitl_node",
            _route_after_hitl,
            {
                "parser_agent": "parser_agent",
                "solver_agent": "solver_agent",
            },
        )

        # ── explainer -> manim -> END ─────────────────────────────────────────
        graph.add_edge("explainer_agent", "manim_node")
        graph.add_edge("manim_node",      "satisfaction_hitl")
        graph.add_conditional_edges(
            "satisfaction_hitl",
            _route_after_satisfaction,
            {
                "END":          END,
                "parser_agent": "parser_agent",
            },
        )

        return graph


# ── Module-level singletons exposed to app.py ─────────────────────────────────
workflow = MathTutorWorkflow()
chatbot  = workflow.app