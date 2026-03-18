from agents import *
from agents.nodes import *

class IntentRouterAgent(BaseAgent):

    def intent_router_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""

            _ROUTER_PROMPT = """\
                You are classifying a JEE-level math problem to plan how to solve it.
                
                Return three things:
                - topic: the single best-fit math domain
                - difficulty: easy | medium | hard  (JEE standard)
                - solver_strategy: the most direct solution path in 1-2 sentences — name the \
                specific theorem, technique, or formula to use, not a generic description
                
                Problem:
                {problem_text}"""

            structured_llm = self.llm.with_structured_output(IntentRouterOutput)
            result: IntentRouterOutput = structured_llm.invoke(
                [HumanMessage(content=_ROUTER_PROMPT.format(problem_text=problem_text))])

            payload(state, "intent_router",
                summary=f"{result.topic.title()} | {result.difficulty.title()}",
                fields={
                    "Topic":      result.topic,
                    "Difficulty": result.difficulty,
                    "Strategy":   result.solver_strategy
                },
            )

            logger.info(f"[Router] topic={result.topic} difficulty={result.difficulty} ")
            return {"solution_plan": result.model_dump()}

        except Exception as e:
            logger.error(f"[Router] failed: {e}")
            raise Agent_Exception(e, sys)

