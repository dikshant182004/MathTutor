from backend.agents import *
from backend.agents.nodes import *


class IntentRouterAgent(BaseAgent):

    def intent_router_agent(self, state: AgentState) -> AgentState:
        try:
            parsed       = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""

            _ROUTER_PROMPT = """\
                You are classifying a JEE-level math problem to plan how to solve it.

                Return FOUR things:

                1. topic        — single best-fit math domain
                2. difficulty   — easy | medium | hard  (JEE standard)
                3. solver_strategy — the most direct solution path in 1-2 sentences.
                                     Name the specific theorem, technique, or formula.
                4. intent_type  — classify the student's intent:
                    solve          → they want a full worked solution  (DEFAULT — use this when unsure)
                    explain        → they want a concept or method explained, no new calculation
                    hint           → they want a small nudge/clue, not a complete solution
                    formula_lookup → they are only asking for a formula or theorem statement

                INTENT CLASSIFICATION RULES:
                - If the input contains a numeric problem to solve → solve
                - If the input starts with "explain", "what is", "why does", "how does" → explain
                - If the input contains "hint", "clue", "just tell me where to start" → hint
                - If the input contains "formula for", "state the theorem", "what is the formula" → formula_lookup
                - When in doubt → solve

                Problem:
                {problem_text}"""

            structured_llm = self.llm.with_structured_output(IntentRouterOutput)
            result: IntentRouterOutput = structured_llm.invoke(
                [HumanMessage(content=_ROUTER_PROMPT.format(problem_text=problem_text))]
            )

            payload(
                state, "intent_router",
                summary=f"{result.topic.title()} | {result.difficulty.title()} | {result.intent_type}",
                fields={
                    "Topic":       result.topic,
                    "Difficulty":  result.difficulty,
                    "Intent":      result.intent_type,   # 5a
                    "Strategy":    result.solver_strategy,
                },
            )

            logger.info(
                f"[Router] topic={result.topic} difficulty={result.difficulty} "
                f"intent={result.intent_type}"
            )
            return {"solution_plan": result.model_dump()}

        except Exception as e:
            logger.error(f"[Router] failed: {e}")
            raise Agent_Exception(e, sys)