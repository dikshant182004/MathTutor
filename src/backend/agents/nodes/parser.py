from agents import *
from agents.nodes import *

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

