from backend.agents import *
from backend.agents.nodes import *
from backend.v2.verification import deterministic_math_checks


class VerifierAgent(BaseAgent):

    def verifier_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            solver_out = state.get("solver_output") or {}
            iteration = state.get("solve_iterations", 1)
            solution = solver_out.get("solution", "")
            final_answer = solver_out.get("final_answer", "")

            if not solution.strip():
                logger.warning("[Verifier] Empty solution received — routing back to solver")
                return {
                    "verifier_output": {
                        "status": "incorrect",
                        "verdict": "No solution text was produced by the solver.",
                        "suggested_fix": "The solver must produce a complete written solution.",
                        "confidence": 0.0,
                    }
                }

            # Cheap deterministic gate first. A definite symbolic failure should
            # never spend another LLM call merely to rediscover the same error.
            deterministic = deterministic_math_checks(problem_text, solution, final_answer)
            if deterministic.status == "failed":
                verdict = "; ".join(deterministic.failures)
                result = {
                    "status": "incorrect",
                    "verdict": verdict,
                    "suggested_fix": "Correct the final answer so it satisfies the deterministic mathematical checks.",
                    "confidence": deterministic.confidence,
                }
                payload(
                    state, "verifier_agent",
                    summary=f"DETERMINISTIC FAILURE | {deterministic.confidence:.0%} confidence",
                    fields={"Status": "incorrect", "Verdict": verdict[:180]},
                )
                return {
                    "verifier_output": result,
                    "agent_payload_log": state.get("agent_payload_log") or [],
                }

            _VERIFIER_PROMPT = """You are a strict mathematical verifier for JEE-level problems.

Check the solution on:
1. Correctness — every algebraic/logical step.
2. Units/domain — final answer has the right domain/range/units.
3. Edge cases — division by zero, undefined log/sqrt, extraneous roots, etc.

Return correct, partially_correct, incorrect, or needs_human.
When incorrect, identify the exact step and mistake.

Problem:
{problem_text}

Solution (attempt {iteration}):
{solution}

Claimed final answer:
{final_answer}"""

            result: VerifierOutput = self.llm.with_structured_output(VerifierOutput).invoke([
                HumanMessage(content=_VERIFIER_PROMPT.format(
                    problem_text=problem_text,
                    iteration=iteration,
                    solution=solution,
                    final_answer=final_answer,
                ))
            ])

            updates: dict = {"verifier_output": result.model_dump()}

            if result.status == "needs_human":
                updates["hitl_required"] = True
                updates["hitl_type"] = "verification"
                updates["hitl_reason"] = (
                    result.hitl_reason or result.verdict or "Verifier cannot determine correctness."
                )

            payload(
                state, "verifier_agent",
                summary=f"{result.status.upper()} | {result.confidence:.0%} confidence",
                fields={
                    "Status": result.status,
                    "Deterministic checks": "; ".join(deterministic.checks)[:180],
                    "Verdict": result.verdict[:180],
                    "Fix": result.suggested_fix[:120] if result.suggested_fix else None,
                },
            )
            logger.info(
                f"[Verifier] status={result.status} confidence={result.confidence:.2f} "
                f"| deterministic={deterministic.status}"
            )
            return {**updates, "agent_payload_log": state.get("agent_payload_log") or []}

        except Exception as e:
            logger.error(f"[Verifier] failed: {e}")
            raise Agent_Exception(e, sys)
