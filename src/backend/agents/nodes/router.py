from backend.agents import *
from backend.agents.nodes import *


class IntentRouterAgent(BaseAgent):

    def intent_router_agent(self, state: AgentState) -> AgentState:
        try:
            parsed       = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""

            _ROUTER_PROMPT = """\
                You are classifying a JEE-level math student's input to decide how to respond.

                Return FOUR things:

                1. topic        — single best-fit math domain (algebra | probability | calculus |
                                  linear_algebra | geometry | trigonometry | statistics | number_theory)
                2. difficulty   — easy | medium | hard  (JEE standard; use "medium" for non-problem inputs)
                3. solver_strategy — the most direct solution path in 1-2 sentences.
                                     For non-solve intents, describe the explanation or generation approach.
                4. intent_type  — classify the student's intent using EXACTLY one of:

                    solve          → student wants a full worked solution to a specific numeric/algebraic problem
                                     DEFAULT — use when unsure and input contains a concrete problem
                    explain        → student wants a concept, theorem, or method explained
                                     triggers: "what is", "explain", "why does", "how does", "tell me about",
                                     "describe", "what are", "what's the difference between"
                    hint           → student wants a small nudge/clue for a problem they're working on
                                     triggers: "hint", "clue", "just tell me where to start", "am I on the right track"
                    formula_lookup → student wants a formula or theorem statement only
                                     triggers: "formula for", "state the theorem", "what is the formula for"
                    research       → student wants info about recent developments, history, or general math knowledge
                                     that is NOT a specific problem to solve and NOT a simple concept explanation
                                     triggers: "recent", "latest", "discovery", "history of", "who discovered",
                                     "applications of", "real world use", "tell me something interesting about"
                    generate       → student wants practice problems, exam questions, or examples created for them
                                     triggers: "give me", "create", "make", "generate", "some questions",
                                     "practice problems", "example problems", "test me", "quiz me"

                CRITICAL RULES:
                - If input contains a SPECIFIC number, equation, or expression to solve → ALWAYS "solve"
                - "What is Bayes theorem?" → "explain"  (NOT "solve")
                - "What is the formula for integration by parts?" → "formula_lookup"
                - "Give me 5 questions on probability for JEE Mains" → "generate"
                - "Tell me about recent discoveries in mathematics" → "research"
                - "Explain the AM-GM inequality and when to use it" → "explain"
                - When in doubt between explain and solve: if there is no specific numeric problem → "explain"

                Student input:
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
                    "Intent":      result.intent_type,
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