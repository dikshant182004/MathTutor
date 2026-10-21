from backend.agents import *
from backend.agents.nodes import *
from backend.agents.runtime_telemetry import budget_limits
from backend.agents.nodes.memory.memory_manager import (
    trim_messages_if_needed,
    format_ltm_for_solver,
)

_TOOLS_NO_RAG = [calculator_tool, web_search_tool]


class SolverAgent(BaseAgent):

    _SYSTEM_PROMPT = """\
        You are an expert JEE mathematics solver.
        Write solutions exactly as they appear on a JEE answer sheet.

        STRICT TOOL RULE
        Each response must be EITHER a structured tool call OR written working.
        Never both in the same response. Never write <function=...> tags in text.

        TOOLS
        {tool_guide}

        SOLUTION FORMAT
        Given:    [restate given info with the problem's exact variable names]
        To find:  [what is asked]

        Step 1 — [Heading e.g. "Integration by parts"]:
            [line-by-line working, each line ending with the new expression]
            ∴  [result of this step]

        Step N — ...

        ∴ Final Answer: [exact result — same notation as working — with unit if needed]

        Strategy: {strategy}
        Attempt {attempt} of {max_attempts}.{feedback_block}{ltm_block}"""

    _TOOL_GUIDE_NO_RAG = """\
        - symbolic_calculator → ONLY for large factorials, high-precision decimals,
                                or large matrix operations. NOT for basic probability,
                                simple integrals, or routine algebra.
        - web_search_tool     → formulae or theory you need to look up."""

    # Removing rag tool from second iteration reasoning
    _TOOL_GUIDE_RETRY = """\
        RAG was already used in the previous attempt — do NOT call rag_tool again.
        Use the verifier feedback above to correct your approach directly.
        - symbolic_calculator → ONLY for large factorials, high-precision decimals,
                                or large matrix operations. NOT for basic probability,
                                simple integrals, or routine algebra.
        - web_search_tool     → formulae or theory you need to look up."""

    def _build_system(
        self,
        strategy:      str,
        attempt:       int,
        max_attempts:  int,
        rag_available: bool,
        is_retry:      bool,
        feedback:      str,
        ltm_hint:      str,
    ) -> SystemMessage:
        feedback_block = (
            f"\n\nPrevious attempt was INCORRECT.\nVerifier feedback: {feedback}"
            if feedback else ""
        )
        ltm_block = (
            f"\n\nSTUDENT CONTEXT (from past sessions):\n{ltm_hint}"
            if ltm_hint else ""
        )

        if is_retry:
            tool_guide = self._TOOL_GUIDE_RETRY
        elif rag_available:
            tool_guide = self._TOOL_GUIDE_WITH_RAG
        else:
            tool_guide = self._TOOL_GUIDE_NO_RAG

        return SystemMessage(content=self._SYSTEM_PROMPT.format(
            tool_guide     = tool_guide,
            strategy       = strategy or "choose the most direct method",
            attempt        = attempt,
            max_attempts   = max_attempts,
            feedback_block = feedback_block,
            ltm_block      = ltm_block,
        ))

    def _bind_tools(self, rag_available: bool = False, thread_id: str = "",
                    rag_already_called: bool = False, is_retry: bool = False):
        """Bind only calculation and web tools; retrieval runs as a graph node."""
        return self.reserve_llm.bind_tools(_TOOLS_NO_RAG, tool_choice="auto")

    def _extract_final_answer(self, text: str) -> str:
        for marker in ("∴ Final Answer:", "Final Answer:", "FINAL ANSWER:"):
            if marker in text:
                return text.split(marker)[-1].strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return lines[-1] if lines else text

    def solver_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            plan          = state.get("solution_plan") or {}
            strategy      = plan.get("solver_strategy", "")
            intent_type   = plan.get("intent_type", "solve")
            difficulty    = plan.get("difficulty", "medium")
            topic         = parsed.get("topic") or ""
            iteration     = state.get("solve_iterations", 0)
            llm_calls_made = 0
            thread_id     = state.get("thread_id") or ""
            human_fb      = state.get("human_feedback") or ""
            prev_verifier = state.get("verifier_output") or {}
            ltm_context   = state.get("ltm_context") or {}
            existing_msgs = state.get("messages") or []

            if state.get("solve_iterations", 0) == 0 and not any(
                isinstance(m, ToolMessage) for m in existing_msgs
            ):
                existing_msgs = []

            messages_before_trim = list(existing_msgs)
            existing_msgs = trim_messages_if_needed(
                messages=list(existing_msgs),
                thread_id=thread_id,
                llm=self.reserve_llm,
            )
            # Trimming performs one real summarization LLM call only when it changes the messages.
            if existing_msgs != messages_before_trim:
                llm_calls_made += 1

            ltm_hint = format_ltm_for_solver(ltm_context, topic) if ltm_context else ""

            feedback = prev_verifier.get("suggested_fix") or ""
            if human_fb and not existing_msgs:
                feedback = f"{feedback}\nHuman feedback: {human_fb}".strip()

            rag_available = False
            is_retry = iteration > 0

            rag_already_called = False
            rag_query = ""
            rag_result = ""

system = self._build_system(
                strategy      = strategy,
                attempt       = iteration + 1,
                max_attempts  = 3,
                rag_available = rag_available,
                is_retry      = is_retry,
                feedback      = feedback,
                ltm_hint      = ltm_hint,
            )

            if not existing_msgs:
                retrieval_context = str(state.get("retrieved_context") or "").strip()
                user_prompt = f"Solve this problem:\n\n{problem_text}"
                if retrieval_context:
                    user_prompt += (
                        "\n\nOptional pre-fetched web context (use only if relevant; "
                        "do not treat snippets as proof):\n" + retrieval_context[:12000]
                    )
                messages = [system, HumanMessage(content=user_prompt)]
            else:
                messages = (
                    [system] + list(existing_msgs[1:])
                    if isinstance(existing_msgs[0], SystemMessage)
                    else [system] + list(existing_msgs)
                )

            if int(state.get("llm_calls_used") or 0) + llm_calls_made >= budget_limits(state)[0]:
                return {
                    "llm_calls_made": llm_calls_made,
                    "budget_exhausted": True,
                    "final_response": (
                        "This request reached its configured model-call budget. "
                        "Please narrow the question or start a new request."
                    ),
                }
            response = self._bind_tools(
                rag_available, thread_id, rag_already_called, is_retry
            ).invoke(messages)
            llm_calls_made += 1

            updates: dict = {"messages": [response]}

            if getattr(response, "tool_calls", None):
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info("[Solver] Tool calls: %s", tool_names)
                return {
                    "llm_calls_made": llm_calls_made,
                    **updates,
                    "agent_payload_log": state.get("agent_payload_log") or [],
                }

            solution_text = response.content or ""
            if not solution_text.strip():
                logger.warning("[Solver] Empty solution text after tool loop — forcing retry signal")
                return {
                        "llm_calls_made": llm_calls_made,
                    "solve_iterations": iteration + 1,
                    "agent_payload_log": state.get("agent_payload_log") or [],
                    "solver_output": {
                        "solution":         "",
                        "final_answer":     "",
                        "rag_context_used": False,
                        "calculator_used":  False,
                        "web_search_used":  False,
                    },
                }

            all_msgs = list(existing_msgs) + [response]

            calc_used = any(
                isinstance(m, ToolMessage) and "calculator" in (m.name or "").lower()
                for m in all_msgs
            )
            rag_used = False
            web_used = any(
                isinstance(m, ToolMessage) and "web_search" in (m.name or "").lower()
                for m in all_msgs
            )

            final_answer = self._extract_final_answer(solution_text)
            payload(
                state, "solver_agent",
                summary=(
                    f"Solution produced | attempt {iteration + 1} | "
                    f"topic={topic} | difficulty={difficulty}"
                ),
                fields={
                    "Topic":         topic,
                    "Intent":        intent_type,
                    "Difficulty":    difficulty,
                    "Attempt":       str(iteration + 1),
                    "RAG used":      str(rag_used),
                    "RAG query":     rag_query,
                    "RAG preview":   rag_result[:500] if isinstance(rag_result, str) else str(rag_result)[:500],
                    "RAG source":    "student's uploaded PDF" if rag_query else "",
                    "Calc used":     str(calc_used),
                    "Web used":      str(web_used),
                    "RAG indexed":   str(rag_available),
                    "LTM hint":      ltm_hint[:80] if ltm_hint else "none",
                    "Final answer":  final_answer[:80] if final_answer else "unknown",
                    "Preview":       solution_text[:120],
                },
            )

            logger.info(
                f"[Solver] Working done | iter={iteration + 1} "
                f"| calc={calc_used} | rag={rag_used} | web={web_used}"
            )
            return {
                        "llm_calls_made": llm_calls_made,
                "messages":         [response],
                "solve_iterations": iteration + 1,
                "agent_payload_log": state.get("agent_payload_log") or [],
                "solver_output": {
                    "solution":         solution_text,
                    "final_answer":     final_answer,
                    "rag_context_used": rag_used,
                    "calculator_used":  calc_used,
                    "web_search_used":  web_used,
                },
            }

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e) or "tokens" in str(e).lower():
                logger.warning(f"[Solver] Token/rate limit hit: {e}")
                return {
                    "solve_iterations": iteration + 1,
                    "hitl_required":    True,
                    "hitl_type":        "verification",
                    "hitl_reason":      "Token limit reached. Please try again in a moment.",
                    "agent_payload_log": state.get("agent_payload_log") or [],                                              
                    "solver_output": {
                        "solution":         "",
                        "final_answer":     "",
                        "rag_context_used": False,
                        "calculator_used":  False,
                        "web_search_used":  False,
                    }
                }
            logger.error(f"[Solver] failed: {e}")
            raise Agent_Exception(e, sys)