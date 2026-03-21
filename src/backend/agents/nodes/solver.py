from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes.tools.tools import has_store, rag_tool as _rag_tool_base
from backend.agents.nodes.memory.memory_manager import (
    trim_messages_if_needed,
    format_ltm_for_solver,
)
from langchain_core.tools import tool as lc_tool


def _make_scoped_rag(thread_id: str):
    # just scoping it to provide the right index to our rag_tool with the correct thread_id 
    @lc_tool
    def rag_tool(query: str) -> str:
        """
        Search the uploaded PDF document for passages relevant to the query.

        Call this FIRST for any problem that could be covered by the student's
        uploaded notes or textbook. Use a focused query like
        "integration by parts formula" — not the full problem text.
        If no document is indexed or no relevant passages are found,
        fall back to web_search_tool.
        """
        if not has_store(thread_id):
            return (
                "CRAG: No document is indexed for this session. "
                "Do NOT call rag_tool again — use web_search_tool instead."
            )
        return _rag_tool_base.invoke({"query": query, "thread_id": thread_id})

    return rag_tool


_TOOLS_NO_RAG   = [calculator_tool, web_search_tool]


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

    _TOOL_GUIDE_WITH_RAG = """\
        - rag_tool            → search the uploaded document FIRST for relevant theorems/formulae.
                                Use a focused query like "integration by parts formula", not the full problem.
                                If it returns nothing useful, fall back to web_search_tool.
        - symbolic_calculator → ONLY for large factorials, high-precision decimals,
                                or large matrix operations. NOT for basic probability,
                                simple integrals, or routine algebra.
        - web_search_tool     → external formulae or theory when rag_tool finds nothing."""

    _TOOL_GUIDE_NO_RAG = """\
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
        return SystemMessage(content=self._SYSTEM_PROMPT.format(
            tool_guide     = self._TOOL_GUIDE_WITH_RAG if rag_available else self._TOOL_GUIDE_NO_RAG,
            strategy       = strategy or "choose the most direct method",
            attempt        = attempt,
            max_attempts   = max_attempts,
            feedback_block = feedback_block,
            ltm_block      = ltm_block,
        ))

    def _bind_tools(self, rag_available: bool, thread_id: str):
        """
        when RAG is available, use the scoped wrapper so the LLM
        never sees thread_id as a required argument in the tool schema.
        """
        if rag_available:
            scoped_rag = _make_scoped_rag(thread_id)
            tools = [scoped_rag, calculator_tool, web_search_tool]
        else:
            tools = _TOOLS_NO_RAG
        return self.llm.bind_tools(tools, tool_choice="auto")

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
            topic         = parsed.get("topic") or ""
            iteration     = state.get("solve_iterations", 0)
            thread_id     = state.get("thread_id") or ""
            human_fb      = state.get("human_feedback") or ""
            prev_verifier = state.get("verifier_output") or {}
            ltm_context   = state.get("ltm_context") or {}
            existing_msgs = state.get("messages") or []

            existing_msgs = trim_messages_if_needed(
                messages  = list(existing_msgs),
                thread_id = thread_id,
                llm       = self.llm,
            )

            ltm_hint = format_ltm_for_solver(ltm_context, topic) if ltm_context else ""

            feedback = prev_verifier.get("suggested_fix") or ""
            if human_fb and not existing_msgs:
                feedback = f"{feedback}\nHuman feedback: {human_fb}".strip()

            rag_available = has_store(thread_id)
            system = self._build_system(
                strategy      = strategy,
                attempt       = iteration + 1,
                max_attempts  = 3,
                rag_available = rag_available,
                feedback      = feedback,
                ltm_hint      = ltm_hint,
            )

            if not existing_msgs:
                messages = [
                    system,
                    HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
                ]
            else:
                messages = (
                    [system] + list(existing_msgs[1:])
                    if isinstance(existing_msgs[0], SystemMessage)
                    else [system] + list(existing_msgs)
                )

            response = self._bind_tools(rag_available, thread_id).invoke(messages)
            updates: dict = {"messages": [response]}

            if getattr(response, "tool_calls", None):
                logger.info(
                    f"[Solver] Tool calls: {[tc['name'] for tc in response.tool_calls]}"
                )
                return updates

            solution_text = response.content or ""
            if not solution_text.strip():
                logger.warning("[Solver] Empty solution text after tool loop — forcing retry signal")
                return {
                    "solve_iterations": iteration + 1,
                    "solver_output": {
                        "solution":         "",
                        "final_answer":     "",
                        "rag_context_used": False,
                        "calculator_used":  False,
                    },
                }

            all_msgs = list(state.get("messages") or []) + [response]
            calc_used = any(
                isinstance(m, ToolMessage) and "calculator" in (m.name or "").lower()
                for m in all_msgs
            )
            rag_used = any(
                isinstance(m, ToolMessage) and "rag" in (m.name or "").lower()
                for m in all_msgs
            )

            payload(
                state, "solver_agent",
                summary = f"Working produced | attempt {iteration + 1}",
                fields  = {
                    "Preview": solution_text[:120],
                    "Calc":    str(calc_used),
                    "RAG":     str(rag_used),
                    "Attempt": str(iteration + 1),
                },
            )

            logger.info(
                f"[Solver] Working done | iter={iteration + 1} "
                f"| calc={calc_used} | rag={rag_used}"
            )
            return {
                "messages": [response],
                "solve_iterations": iteration + 1,
                "solver_output": {
                    "solution":         solution_text,
                    "final_answer":     self._extract_final_answer(solution_text),
                    "rag_context_used": rag_used,
                    "calculator_used":  calc_used,
                },
            }

        except Exception as e:
            logger.error(f"[Solver] failed: {e}")
            raise Agent_Exception(e, sys)