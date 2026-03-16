
class IntentRouterAgent(BaseAgent):

    def intent_router_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            context_snippet = (state.get("retrieved_context") or "")[:500]

            prompt = f"""You are an intent router for a JEE math mentor system.

                Classify the problem below and decide how to solve it.
                                
                IMPORTANT: needs_visualization and requires_calculator MUST be JSON booleans
                (the literal values true or false, never the strings "true" or "false").

                Fields to return:
                topic            — algebra | probability | calculus | linear_algebra | geometry | trigonometry | statistics | number_theory
                difficulty       — easy | medium | hard
                solver_strategy  — 1-2 sentence description of the optimal solution approach
                needs_visualization — Set true if ANY of the following apply:
                    • 3D geometry: spheres, cones, cylinders, planes, lines in space, distance/angle in 3D
                    • 2D coordinate geometry: circles, parabolas, ellipses, hyperbolas, tangent/normal lines
                    • Graph sketching: functions, curves, regions of integration
                    • Vectors: cross product, angle between vectors, projection
                    • Transformations: rotations, reflections, shear in 2D/3D
                    • Trigonometry: unit circle, sinusoidal graphs, phase shift
                    • Calculus concepts: area under curve, solids of revolution, limits graphically
                    • Probability: tree diagrams, Venn diagrams, geometric probability
                    Set false for pure algebra, number theory, or problems solved entirely with equations.
                visualization_hint  — Concise description of what the Manim animation should show (required when needs_visualization=true, else null).
                    Examples: "Draw sphere with inscribed cone, label r and h, animate volume formula"
                             "Plot parabola y=x² and tangent at x=2, shade region"
                             "Show unit circle, animate angle θ, trace sin/cos values"
                requires_calculator — true if numerical computation is needed

                Context (first 500 chars):
                {context_snippet}

                Problem:
                {problem_text}"""

            structured_llm = self.llm.with_structured_output(IntentRouterOutput)
            result: IntentRouterOutput = structured_llm.invoke([HumanMessage(content=prompt)])

            state["solution_plan"] = result.model_dump()
            _log_payload(state, "intent_router",
                summary=f"{result.topic.title()} | {result.difficulty.title()} | {'Viz needed' if result.needs_visualization else 'No viz'}",
                fields={
                    "Topic":      result.topic,
                    "Difficulty": result.difficulty,
                    "Strategy":   result.solver_strategy[:150],
                    "Needs viz":  str(result.needs_visualization),
                    "Needs calc": str(result.requires_calculator),
                    "Viz hint":   result.visualization_hint,
                },
            )
            logger.info(f"[Router] topic={result.topic} difficulty={result.difficulty} viz={result.needs_visualization}")
            return state

        except Exception as e:
            logger.error(f"[Router] failed: {e}")
            raise Agent_Exception(e, sys)

