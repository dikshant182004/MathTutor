from backend.agents import *
from backend.agents.nodes import *

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
                return {
                    "hitl_required": True,
                    "hitl_type":     "bad_input",
                    "hitl_reason":   "No question text could be extracted from the input.",
                }

            _PARSER_PROMPT = """You are a math problem parser. Your job is to clean and structure the raw input.
                Steps:
                1. Fix any OCR/ASR noise and normalise math notation (fractions, exponents, integrals, Greek letters, etc.).
                2. Extract all variables and explicit constraints from the problem.
                3. Set needs_clarification=true ONLY when the problem is unsolvable without more information — \
                for example, a missing variable definition, a truncated sentence, or two contradictory conditions. \
                Do NOT set it for problems that are unusual or hard. If you can make a reasonable interpretation, solve it.
                4. When needs_clarification=true, write clarification_reason as a clear, specific, student-facing message
                explaining EXACTLY what information is missing or ambiguous. 
                Example: "The variable 'n' is used but never defined. Please specify what n represents."
                Example: "The problem appears to be cut off after 'find the value of'. Please provide the complete question."
                NOT: "Problem is ambiguous." — be specific.
                
                Input:
                {_raw_text}"""
 
            structured_llm = self.llm.with_structured_output(ParserOutput)
            parsed: ParserOutput = structured_llm.invoke(
                [HumanMessage(content=_PARSER_PROMPT.format(_raw_text=raw_text))])

            updates: dict = {"parsed_data": parsed.model_dump()}
 
            if parsed.needs_clarification:
                updates["hitl_required"] = True
                updates["hitl_type"]     = "clarification" 
                updates["hitl_reason"]  = parsed.clarification_reason or "Problem is ambiguous."

            payload(
                state, "parser_agent",
                summary = f"Topic: {parsed.topic or '?'}",
                fields  = {
                    "Problem": parsed.problem_text[:120],
                    "Topic": parsed.topic,
                    "Variables": ", ".join(parsed.variables) if parsed.variables else None,
                    "Constraints": ", ".join(parsed.constraints) if parsed.constraints else None,
                    "Clarification": parsed.clarification_reason if parsed.needs_clarification else None,
                },
            )
            logger.info(f"[Parser] topic={parsed.topic} needs_clarification={parsed.needs_clarification}")
            return updates
        
        except Exception as e:
            logger.error(f"[Parser] failed: {e}")
            raise Agent_Exception(e, sys)

