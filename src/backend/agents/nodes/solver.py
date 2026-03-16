
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
1. Each response must be EITHER a tool call OR written reasoning. NEVER BOTH in the same response.
2. If you need a tool: output ONLY the structured tool call, NOTHING else — no prose before, no prose after, no explanation.
3. If you are writing the solution: output ONLY text — do NOT embed any tool calls inside text.
4. NEVER write <function=...> tags or <tool_call> tags anywhere in your text. The API handles tool calls — you must use the structured mechanism, not text tags.
5. Decide at the START of each turn: do I need a tool? If yes → ONLY the tool call. If no → ONLY the solution text.
6. If you are unsure whether to call a tool, skip it and write the solution directly.

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

            # Retry logic for Groq 400 tool_use_failed.
            # The model sometimes mixes prose + inline <function=...> tags.
            # Strategy: up to 2 retries, each with a cleaner/stricter prompt.
            # Retry 1 — same temperature, stripped context
            # Retry 2 — temperature=0, ultra-strict one-line prompt
            def _invoke_with_retries(messages: list, problem_text: str, system_prompt: str):
                try:
                    return solver_llm.invoke(messages)
                except Exception as err1:
                    err1_str = str(err1)
                    if "tool_use_failed" not in err1_str and "400" not in err1_str:
                        raise
                    logger.warning(f"[Solver] tool_use_failed attempt 1, retrying | {err1_str[:120]}")

                    # Retry 1 — clean context, same LLM
                    retry1_messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=(
                            "IMPORTANT: Your next response must be EITHER a single tool call "
                            "OR a written solution. Do NOT mix them. Do NOT write <function=...> tags.\n\n"
                            f"Solve this problem:\n\n{problem_text}"
                        )),
                    ]
                    try:
                        return solver_llm.invoke(retry1_messages)
                    except Exception as err2:
                        err2_str = str(err2)
                        if "tool_use_failed" not in err2_str and "400" not in err2_str:
                            raise
                        logger.warning(f"[Solver] tool_use_failed attempt 2, retrying with temp=0 | {err2_str[:120]}")

                        # Retry 2 — temperature=0, no tools, force direct answer
                        cold_llm = ChatGroq(
                            api_key=_get_secret("GROQ_API_KEY"),
                            model_name="llama-3.3-70b-versatile",
                            temperature=0,
                        )
                        retry2_messages = [
                            SystemMessage(content=(
                                "You are a JEE mathematics solver. "
                                "Write a complete step-by-step solution as plain text only. "
                                "Do NOT use any tools. Do NOT write <function=...> tags. "
                                "End with FINAL ANSWER on its own line."
                            )),
                            HumanMessage(content=f"Solve this problem:\n\n{problem_text}"),
                        ]
                        return cold_llm.invoke(retry2_messages)

            response = _invoke_with_retries(messages, problem_text, system_prompt)

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