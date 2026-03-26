import yaml
import re 
from backend.agents import *
from backend.agents.nodes import *
from functools import lru_cache
from pathlib import Path

_GUARDRAIL_POLICY_DIR = Path(__file__).resolve().parent / "security_checks"

@lru_cache(maxsize=1)
def _load_policies() -> dict:
    """Load all YAML policy files once and cache them."""

    topic_policy     = yaml.safe_load((_GUARDRAIL_POLICY_DIR / "topic_policy.yaml").read_text())
    injection_policy = yaml.safe_load((_GUARDRAIL_POLICY_DIR / "injection_patterns.yaml").read_text())
    return {
        "allowed_topics": topic_policy.get("allowed_topics", []),
        "blocked_categories": topic_policy.get("blocked_categories", []),
        "borderline_policy": topic_policy.get("borderline_policy", "allow_if_math_is_central"),
        "prompt_injection": injection_policy.get("prompt_injection", []),
        "extraction": injection_policy.get("extraction_attempts", []),
        "safe_patterns": injection_policy.get("safe_patterns", []),
    }


def _rule_based_check(text: str) -> tuple[bool, str, str]:
    """
    Fast rule-based check — runs before the LLM call.

    Returns (is_blocked, block_reason, message).
    is_blocked=False means the input passed all rules and can proceed to LLM check.
    """

    lower = text.lower().strip()
    policies = _load_policies()

    for safe in policies["safe_patterns"]:
        if safe.lower() in lower:
            return False, "", ""

    for pattern in policies["prompt_injection"] + policies["extraction"]:
        if pattern.lower() in lower:
            logger.warning(f"[Guardrail] Injection pattern matched: '{pattern}'")
            return (
                True,
                "prompt_injection",
                "I can only help with JEE mathematics problems.",
            )

    # ── PII — basic regex for email / phone / Aadhaar-like numbers ───────────
    pii_patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
        r"\b\d{10}\b",        # 10-digit phone
        r"\b\d{4}\s\d{4}\s\d{4}\b",  # Aadhaar format
    ]
    for p in pii_patterns:
        if re.search(p, text):
            logger.warning("[Guardrail] PII pattern detected")
            return (
                True,
                "pii",
                "Please do not share personal information. Rephrase your math problem.",
            )

    # if len(lower.split()) < 3:
    #     return (
    #         True,
    #         "off_topic",
    #         "Please provide a complete math problem.",
    #     )

    return False, "", ""


class GuardrailAgent(BaseAgent):
    
    def _build_guardrail_prompt(self, raw_input: str, policies: dict) -> str:
        allowed  = ", ".join(policies["allowed_topics"])
        blocked  = ", ".join(policies["blocked_categories"])
        borderline = policies["borderline_policy"]

        return f"""You are the input guardrail for a JEE mathematics tutor.

            Your job: decide if the student's input is an on-topic math problem or request.

            ALLOWED topics: {allowed}
            BLOCKED categories: {blocked}
            Borderline policy: {borderline}

            Rules:
            - Pass physics / engineering problems if the core task is mathematical.
            - Pass requests for math help, hints, concept explanations, formula lookups.
            - Block anything clearly outside mathematics: general knowledge, coding tasks,
            personal advice, essay writing, entertainment.
            - If you are even slightly unsure, PASS it — false positives are worse than
            letting a borderline question through.

            Student input:
            {raw_input}"""

    def guardrail_agent(self, state: AgentState) -> dict:
        try:
            raw_input = (
                state.get("user_corrected_text")
                or state.get("raw_text")
                or state.get("ocr_text")
                or state.get("transcript")
                or ""
            ).strip()

            # ── Stage 1: rule-based fast path ─────────────────────────────────
            is_blocked, block_reason, block_message = _rule_based_check(raw_input)

            if is_blocked:
                payload(
                    state, "guardrail_agent",
                    summary = f"BLOCKED ({block_reason}) — rule-based",
                    fields  = {"Reason": block_reason, "Input preview": raw_input[:80]},
                )
                logger.warning(f"[Guardrail] Rule blocked | reason={block_reason}")
                return {
                    "guardrail_passed": False,
                    "guardrail_reason": block_reason,
                    "final_response":   block_message,
                }

            # ── Stage 2: LLM topic relevance check ────────────────────────────
            policies = _load_policies()
            prompt   = self._build_guardrail_prompt(raw_input, policies)

            result: GuardrailOutput = self.llm.with_structured_output(GuardrailOutput).invoke(
                [HumanMessage(content=prompt)]
            )

            updates: dict = {
                "guardrail_passed": result.passed,
                "guardrail_reason": result.block_reason,
            }

            if not result.passed:
                updates["final_response"] = (
                    result.message or "I can only help with JEE mathematics problems."
                )

            payload(
                state, "guardrail_agent",
                summary = f"{'PASSED' if result.passed else 'BLOCKED'} | topic={result.topic or '?'}",
                fields  = {
                    "Passed":  str(result.passed),
                    "Topic":   result.topic,
                    "Blocked": result.block_reason,
                },
            )
            logger.info(f"[Guardrail] passed={result.passed} topic={result.topic}")
            return updates

        except Exception as e:
            logger.error(f"[Guardrail] failed: {e}")
            raise Agent_Exception(e, sys)