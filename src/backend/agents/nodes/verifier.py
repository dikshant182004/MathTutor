from backend.agents import *
from backend.agents.nodes import *

class VerifierAgent(BaseAgent):

    def verifier_agent(self, state: AgentState) -> AgentState:
        try:
            parsed       = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            solver_out   = state.get("solver_output") or {}
            iteration    = state.get("solve_iterations", 1)
            solution     = solver_out.get("solution", "")
            final_answer = solver_out.get("final_answer", "")

            # Add this right after reading solver_out, before building the prompt:
            if not solution.strip():
                logger.warning("[Verifier] Empty solution received — routing back to solver")
                return {
                    "verifier_output": {
                        "status":        "incorrect",
                        "verdict":       "No solution text was produced by the solver.",
                        "suggested_fix": "The solver must produce a complete written solution. Do not call tools without following up with written working.",
                        "confidence":    0.0,
                        "hitl_reason":   None,
                    }
                }

            _VERIFIER_PROMPT = """You are a strict mathematical verifier for JEE-level problems.
 
                Check the solution below against the problem on three criteria:
                1. Correctness   — is every algebraic step valid? Cite the step number if not.
                2. Units/domain  — is the final answer in the right domain/range/units?
                3. Edge cases    — division by zero, undefined log/sqrt, empty set, etc.
                
                Status rules:
                - 'correct'           → all three checks pass, fully satisfied.
                - 'partially_correct' → method is right but an arithmetic/sign error exists.
                - 'incorrect'         → the method itself is wrong.
                - 'needs_human'       → you genuinely cannot determine correctness.
                                        Use this sparingly — only when domain knowledge is missing.
                
                When not correct: suggested_fix must name the exact step and the exact mistake.
                Not 'check step 3' — write what is wrong and what to do instead.
                
                Problem:
                {problem_text}
                
                Solution (attempt {iteration}):
                {solution}
                
                Claimed final answer: {final_answer}"""
 
            result: VerifierOutput = self.llm.with_structured_output(VerifierOutput).invoke([
                HumanMessage(content=_VERIFIER_PROMPT.format(
                    problem_text      = problem_text,
                    iteration      = iteration,
                    solution     = solution,
                    final_answer = final_answer,
                ))
            ])
 
            updates: dict = {"verifier_output": result.model_dump()}
 
            if result.status == "needs_human":
                updates["hitl_required"] = True
                updates["hitl_type"]     = "verification"  
                updates["hitl_reason"]   = (
                    result.hitl_reason or "Verifier cannot determine correctness."
                )
 
            payload(
                state, "verifier_agent",
                summary = f"{result.status.upper()} | {result.confidence:.0%} confidence",
                fields  = {
                    "Status":  result.status,
                    "Verdict": result.verdict[:180],
                    "Fix":     result.suggested_fix[:120] if result.suggested_fix else None,
                },
            )
            logger.info(f"[Verifier] status={result.status} confidence={result.confidence:.2f}")
            return updates

        except Exception as e:
            logger.error(f"[Verifier] failed: {e}")
            raise Agent_Exception(e, sys)