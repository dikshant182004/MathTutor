
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