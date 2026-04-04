from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes.tools.tools import has_store, rag_tool as _rag_tool_base
from backend.agents.nodes.memory.memory_manager import (
    trim_messages_if_needed,
    format_ltm_for_solver,
)
from langchain_core.tools import tool as lc_tool

_SCOPED_RAG_CACHE: dict = {}

def _make_scoped_rag(thread_id: str):
    # just scoping it to provide the right index to our rag_tool with the correct thread_id
    if thread_id in _SCOPED_RAG_CACHE:
        return _SCOPED_RAG_CACHE[thread_id]
    @lc_tool
    def rag_tool(query: str) -> str:
        """
        Search the uploaded PDF document for passages relevant to the query.

        MANDATORY: This tool is called FIRST on every new problem when a PDF is present.

        CRITICAL QUERY RULE (this is the fix you asked for):
        The query MUST target CONCEPT / TOPIC / THEOREM / FORMULA similarity,
        NOT problem-text similarity.

        Goal: Even if the problem wording is completely different from the student's notes,
        we still want to retrieve the exact formula or explanation the student wrote.

        Examples:
          Problem: "A bag contains 3 red and 2 blue balls. Two balls are drawn..."
          Good query → "Bayes theorem"
          Good query → "Bayes theorem formula"
          Good query → "Bayes theorem conditional probability"

          Problem: "Evaluate ∫ x² sin(x) dx"
          Good query → "integration by parts formula"
          Good query → "integration by parts"

        Bad queries (do NOT use):
          - Full problem statement
          - "find the probability that both balls are red"
          - Anything containing specific numbers or the exact question text

        Keep the query SHORT (3-8 words max) and focused only on the mathematical concept.
        If the tool returns "no relevant passages found", do NOT call it again.
        """
        if not has_store(thread_id):
            return (
                "CRAG: No document is indexed for this session. "
                "Do NOT call rag_tool again — use web_search_tool instead."
            )
        return _rag_tool_base.invoke({"query": query, "thread_id": thread_id})

    _SCOPED_RAG_CACHE[thread_id] = rag_tool
    return rag_tool

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

    _TOOL_GUIDE_WITH_RAG = """\
        MANDATORY FIRST STEP: call rag_tool before writing any solution.

        QUERY STRATEGY (CRITICAL):
        You are NOT searching for a problem that looks similar to the current question.
        You are searching for the exact CONCEPT / THEOREM / FORMULA / TECHNIQUE
        that the student wrote in their notes.

        Step-by-step how to create the query:
        1. Read the problem and identify the SINGLE core mathematical idea needed.
        2. Name that idea as a short concept (theorem name, formula name, method name).
        3. Use that as the query.

        Good examples:
          - "Bayes theorem"
          - "Bayes theorem formula"
          - "Bayes theorem conditional probability"
          - "integration by parts formula"
          - "L'Hôpital's rule"
          - "matrix diagonalization method"

        Bad examples (never do this):
          - The full problem statement
          - Any sentence containing numbers or specific wording from the question

        This guarantees that even if the student's notes only contain the plain formula
        ("P(A|B) = ...") without any example problem, it will still be retrieved.

        Call rag_tool EXACTLY ONCE with this short concept query.
        After you receive the result, write the complete solution using the returned
        passages + your own knowledge. Do NOT call rag_tool again."""

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

    def _bind_tools(self, rag_available: bool, thread_id: str,
                    rag_already_called: bool = False, is_retry: bool = False):
        """
        Tool binding per phase:

        - Retry (iteration > 0)            : [calc, web], tool_choice="auto"
                                             RAG context already injected as HumanMessage;
                                             LLM can use calc/web or just reason directly.
        - No PDF uploaded                  : [calc, web], tool_choice="auto"
        - Attempt 1, RAG not yet called    : [rag, calc, web], tool_choice={"type":"function","function":{"name":"rag_tool"}}
                                             Forces the FIRST call to be rag_tool specifically.
        - Attempt 1, RAG already returned  : [calc, web], tool_choice="auto"
                                             RAG is done; LLM writes solution, optionally
                                             using calc or web if it needs to.
        """
        if is_retry or not rag_available:
            return self.reserve_llm.bind_tools(_TOOLS_NO_RAG, tool_choice="auto")

        scoped_rag = _make_scoped_rag(thread_id)

        if rag_already_called:
            # RAG returned — drop it from the tool list, free the LLM to write
            return self.reserve_llm.bind_tools(_TOOLS_NO_RAG, tool_choice="auto")

        # First entry, RAG not yet called — force rag_tool as the next action
        return self.reserve_llm.bind_tools(
            [scoped_rag, calculator_tool, web_search_tool],
            tool_choice={"type": "function", "function": {"name": "rag_tool"}},
        )

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
            thread_id     = state.get("thread_id") or ""
            human_fb      = state.get("human_feedback") or ""
            prev_verifier = state.get("verifier_output") or {}
            ltm_context   = state.get("ltm_context") or {}
            existing_msgs = state.get("messages") or []

            if state.get("solve_iterations", 0) == 0 and not any(
                isinstance(m, ToolMessage) for m in existing_msgs
            ):
                existing_msgs = []

            existing_msgs = trim_messages_if_needed(   # trimming
                messages  = list(existing_msgs),   
                thread_id = thread_id,
                llm       = self.reserve_llm,
            )

            ltm_hint = format_ltm_for_solver(ltm_context, topic) if ltm_context else ""

            feedback = prev_verifier.get("suggested_fix") or ""
            if human_fb and not existing_msgs:
                feedback = f"{feedback}\nHuman feedback: {human_fb}".strip()

            rag_available = has_store(thread_id)
            is_retry = iteration > 0

            _RAG_SENTINEL = "[RAG context retrieved from student's notes]"
            rag_already_called = any(
                (isinstance(m, ToolMessage) and "rag" in (m.name or "").lower())
                or (isinstance(m, HumanMessage) and _RAG_SENTINEL in (m.content or ""))
                for m in existing_msgs
            )

            # Tracks whether RAG was executed inline THIS invocation.
            # Needed because inline RAG never writes a ToolMessage to history,
            # so all_msgs inspection alone would miss it and report rag_used=False.
            _rag_ran_inline = False
            rag_query  = ""   
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

            response = self._bind_tools(
                rag_available, thread_id, rag_already_called, is_retry
            ).invoke(messages)

            updates: dict = {"messages": [response]}

            if getattr(response, "tool_calls", None):
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info(f"[Solver] Tool calls: {tool_names}")

                has_only_rag = all("rag" in n.lower() for n in tool_names)
                if not has_only_rag:
                    return {
                        **updates,
                        "agent_payload_log": state.get("agent_payload_log") or [],
                    }

                # ---- RAG was called: execute it directly here ---------------
                rag_call        = response.tool_calls[0]
                rag_query       = rag_call["args"].get("query", "")
                rag_result      = _make_scoped_rag(thread_id).invoke({"query": rag_query})
                _rag_ran_inline = True  # Bug 2 fix: flag so rag_used is correct below

                logger.info(f"[Solver] RAG executed inline for query: '{rag_query}'")
               
                rag_context_block = (    
                    f"[RAG context retrieved from student's notes]\n{rag_result}\n"
                    f"[End of RAG context]\n\n"
                    f"Now write the full solution to the problem using the context above "
                    f"plus your own knowledge."
                )

                messages_for_call2 = [
                    system,
                    HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
                    HumanMessage(content=rag_context_block),
                ]

                response2 = self._bind_tools(
                    rag_available=False,   # forces _TOOLS_NO_RAG regardless of PDF
                    thread_id=thread_id,
                    rag_already_called=True,
                    is_retry=False,
                ).invoke(messages_for_call2)

                if getattr(response2, "tool_calls", None):
                    tool_names2 = [tc["name"] for tc in response2.tool_calls]
                    logger.info(f"[Solver] LLM Call 2 tool calls: {tool_names2}")
                    return {
                        "messages": messages_for_call2 + [response2],
                        "agent_payload_log": state.get("agent_payload_log") or [],
                    }

                # Use response2 as our final response going forward
                response = response2

            solution_text = response.content or ""
            if not solution_text.strip():
                logger.warning("[Solver] Empty solution text after tool loop — forcing retry signal")
                return {
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
            rag_used = _rag_ran_inline or rag_already_called or any(
                isinstance(m, ToolMessage) and "rag" in (m.name or "").lower()
                for m in all_msgs
            )
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