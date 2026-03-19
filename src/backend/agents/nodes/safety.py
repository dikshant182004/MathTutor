import yaml
from functools import lru_cache
from pathlib import Path
from backend.agents import *
from backend.agents.nodes import *

_GUARDRAILS_DIR = Path(__file__).parent / "security_checks"

@lru_cache(maxsize=1)
def _load_output_policy() -> dict:
    return yaml.safe_load((_GUARDRAILS_DIR / "output_policy.yaml").read_text())

class SafetyAgent(BaseAgent):
    """
    Output safety agent — fires AFTER verifier confirms the answer is correct,
    BEFORE the explainer formats it for the student.

    Checks the solver's raw working text against output_policy.yaml.

    For a math tutor the violation risk is low but real edge cases exist:
    - A crafted problem embedding harmful synthesis steps in a "calculate" framing
    - Solver hallucinating real names / contact details in a probability example
    - Jailbroken input that slipped past guardrail and produced off-policy output

    Two-stage design (same as guardrail):
    1. Keyword fast path — catches obvious violations without an LLM call.
    2. LLM policy check — for anything that passes keyword check.

    State written:
        safety_passed  — bool, read by route_after_safety
        safety_reason  — set when passed=False
        final_response — student-facing block message when passed=False
    """

    def _keyword_check(self, text: str) -> tuple[bool, str]:
        """Returns (is_blocked, reason). Fast — no LLM."""
        lower = text.lower()
        for kw in _HARD_BLOCK_KEYWORDS:
            if kw in lower:
                return True, f"keyword match: '{kw}'"
        return False, ""

    def _build_prompt(self, solution_text: str, policy: dict) -> str:
        blocked  = "\n".join(f"- {p}" for p in policy.get("blocked_output_patterns", []))
        allowed  = "\n".join(f"- {p}" for p in policy.get("always_allowed", []))

        return f"""You are the output safety checker for a JEE mathematics tutor.

            Review the solver's response below and decide if it is safe to show to a student.
            BLOCKED patterns (any of these → passed=false):
            {blocked}

            ALWAYS ALLOWED (do not flag these):
            {allowed}

            Rules:
            - Mathematical content, LaTeX, step-by-step algebra, theorems — ALWAYS safe.
            - Only block when there is a clear policy violation.
            - When in doubt, PASS — false positives frustrate students.

            Solver output to review:
            {solution_text[:2000]}"""

    def safety_agent(self, state: AgentState) -> dict:
        try:
            solver_out = state.get("solver_output") or {}
            solution_text = solver_out.get("solution", "")

            if not solution_text:
                return {"safety_passed": True}

            # ── Stage 1: keyword fast path ────────────────────────────────────
            is_blocked, kw_reason = self._keyword_check(solution_text)

            if is_blocked:
                policy  = _load_output_policy()
                message = policy.get("on_violation", {}).get(
                    "replacement_message",
                    "The solution could not be displayed due to a policy violation.",
                )
                payload(
                    state, "safety_agent",
                    summary = f"BLOCKED — keyword: {kw_reason}",
                    fields  = {"Reason": kw_reason},
                )
                logger.warning(f"[Safety] Keyword block | {kw_reason}")
                return {
                    "safety_passed":  False,
                    "safety_reason":  kw_reason,
                    "final_response": message,
                }

            # ── Stage 2: LLM policy check ─────────────────────────────────────
            policy = _load_output_policy()
            prompt = self._build_prompt(solution_text, policy)

            result: SafetyOutput = self.llm.with_structured_output(SafetyOutput).invoke(
                [HumanMessage(content=prompt)]
            )

            updates: dict = {
                "safety_passed": result.passed,
                "safety_reason": result.reason,
            }

            if not result.passed:
                replacement = policy.get("on_violation", {}).get(
                    "replacement_message",
                    "The solution could not be displayed due to a policy violation.",
                )
                updates["final_response"] = replacement
                logger.warning(
                    f"[Safety] LLM blocked output | "
                    f"violation={result.violation_type} | reason={result.reason}"
                )

            payload(
                state, "safety_agent",
                summary = f"{'PASSED' if result.passed else 'BLOCKED'} | {result.violation_type or 'ok'}",
                fields  = {
                    "Passed":    str(result.passed),
                    "Violation": result.violation_type,
                    "Reason":    result.reason,
                },
            )
            logger.info(f"[Safety] passed={result.passed}")
            return updates

        except Exception as e:
            logger.error(f"[Safety] failed: {e}")
            raise Agent_Exception(e, sys)