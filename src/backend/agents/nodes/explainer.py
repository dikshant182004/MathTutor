from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes.memory.memory_manager import format_ltm_for_explainer   


class ExplainerAgent(BaseAgent):
    """
    Produces the final student-facing explanation after the verifier confirms
    the answer is correct.

    Two LLM calls — deliberately split:

    Call 1  →  with_structured_output(ExplainerOutput)
        Structures the verified raw working into SolutionStep objects,
        builds key_formulae, key_concepts, common_mistakes, approach_summary.
        Also sets needs_diagram and manim_hint.

    Call 2  →  plain text, only when needs_diagram=True
        Generates the Manim scene code from manim_hint.
        Kept separate because large code strings inside a JSON schema cause
        Groq 400 errors (tool_use_failed) — plain text call avoids this.

    State written:
        explainer_output  — ExplainerOutput dict
        final_response    — rich markdown string rendered in the Streamlit chat bubble
        manim_scene_code  — raw Python string for manim_node (None if not needed)
    """

    def _build_explanation_prompt(
        self,
        problem_text: str,
        solution_text: str,
        verifier_verdict: str,
        topic: str,
        difficulty: str,
        ltm_hint: str,         
    ) -> str:
        base = f"""You are an expert JEE mathematics teacher producing a model answer explanation.

        A student submitted this problem and the solver produced a verified correct solution.
        Your job is to structure the solution into a clear, rigorous explanation that a student
        who got this wrong can study to understand both the answer and the method.

        MATHEMATICAL NOTATION RULES (follow these exactly):
        - Use the EXACT variable names from the problem — never rename them.
        - Integrals   : write ∫ f(x) dx — always include the differential.
        - Fractions   : (numerator)/(denominator) — fully bracketed.
        - Powers      : use the form from the problem (x^2 or x²) consistently.
        - Roots       : √(expr) throughout — never mix with sqrt().
        - Exact form  : give answers as fractions/surds/π/e/ln, not decimals.
        - Limits      : lim_{{x → a}} with arrow.
        - Summations  : Σ_{{k=1}}^{{n}} with explicit bounds.
        - Vectors     : →a for vector, |→a| for magnitude.
        - final_answer: ALWAYS a non-empty string — write "0" not 0, never boolean.
          If the answer is zero write "0", if no such points exist write "0 points".

        STEP STRUCTURE:
        - Each step must show the complete algebraic working line-by-line.
        - Every manipulation on its own line ending in = the new expression.
        - step.result is the expression that step circles/underlines — math only.
        - step.why is ONLY for non-obvious moves (e.g. an unexpected substitution choice).
        - step.inline_diagram: include a small ASCII/Unicode diagram ONLY when it
        genuinely clarifies that specific step (number line, triangle, Venn diagram).
        Leave None for algebraic steps.

        DIAGRAMS:
        - Set needs_diagram=true if the problem involves geometry, coordinate geometry,
        vectors, trigonometric graphs, probability trees, or 3D figures.
        - Set needs_diagram=false for pure algebra, number theory, or arithmetic.
        - When needs_diagram=true, write manim_hint describing EXACTLY what to animate.

        Problem (topic: {topic} | difficulty: {difficulty}):
        {problem_text}

        Verified correct solution (use this as your source of truth — do not recompute):
        {solution_text}

        Verifier notes:
        {verifier_verdict}"""

        if ltm_hint:
            base += f"""

        STUDENT PERSONALISATION (use this to tailor your explanation):
        {ltm_hint}

        Based on the above, ensure common_mistakes directly addresses the student's known
        error patterns, and key_concepts front-loads the concepts they historically struggle with."""

        return base

    def _build_manim_prompt(
        self,
        problem_text: str,
        manim_hint: str,
        solution_text: str,
    ) -> str:
        return f"""Generate complete, runnable Manim Community Edition Python code for a Scene
        that visualises the following JEE mathematics solution.

        The animation should: {manim_hint}

        Requirements:
        - Import from manim: from manim import *
        - Define exactly ONE class that extends Scene (or ThreeDScene for 3D problems).
        - Fully self-contained — no external files, no user input.
        - Use MathTex for all equations — proper LaTeX notation.
        - Use ThreeDScene only if the problem involves 3D geometry.
        - Keep the animation under 30 seconds total.
        - Animate the KEY mathematical insight — not every step.

        Problem: {problem_text}

        Solution summary (for context):
        {solution_text[:600]}

        Reply with ONLY the Python code. No explanation, no markdown fences."""

    def explainer_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            topic         = parsed.get("topic") or "mathematics"

            plan          = state.get("solution_plan") or {}
            difficulty    = plan.get("difficulty") or "medium"

            solver_out    = state.get("solver_output") or {}
            solution_text = solver_out.get("solution", "")

            verifier_out  = state.get("verifier_output") or {}
            verdict       = verifier_out.get("verdict", "")

            # 4a — pull LTM context and format personalisation hint
            ltm_context = state.get("ltm_context") or {}
            ltm_hint    = format_ltm_for_explainer(ltm_context, topic)

            prompt = self._build_explanation_prompt(
                problem_text     = problem_text,
                solution_text    = solution_text,
                verifier_verdict = verdict,
                topic            = topic,
                difficulty       = difficulty,
                ltm_hint         = ltm_hint,    
            )

            result: ExplainerOutput = self.llm.with_structured_output(ExplainerOutput).invoke(
                [HumanMessage(content=prompt)]
            )

            manim_code: str | None = None
            if result.needs_diagram and result.manim_hint:
                logger.info(f"[Explainer] Generating Manim scene: {result.manim_hint[:80]}")
                manim_prompt = self._build_manim_prompt(
                    problem_text  = problem_text,
                    manim_hint    = result.manim_hint,
                    solution_text = solution_text,
                )
                raw  = self.llm.invoke([HumanMessage(content=manim_prompt)])
                code = (raw.content or "").strip()

                # Strip accidental markdown fences
                if code.startswith("```"):
                    code = "\n".join(
                        line for line in code.splitlines()
                        if not line.strip().startswith("```")
                    ).strip()
                if "class" in code and "Scene" in code:
                    manim_code = code
                    logger.info(f"[Explainer] Manim code generated ({len(code)} chars)")
                else:
                    logger.warning("[Explainer] Manim code unusable — skipping")

            final_md = render_md(result, problem_text)

            explainer_dict = result.model_dump()
            explainer_dict["manim_scene_code"] = manim_code

            payload(
                state, "explainer_agent",
                summary = (
                    f"{len(result.steps)} steps | "
                    f"{result.difficulty_rating} | "
                    f"diagram={'yes' if manim_code else 'no'} | "
                    f"personalised={'yes' if ltm_hint else 'no'}"   
                ),
                fields = {
                    "Steps":          str(len(result.steps)),
                    "Key formulae":   str(len(result.key_formulae)),
                    "Key concepts":   str(len(result.key_concepts)),
                    "Common mistakes":str(len(result.common_mistakes)),
                    "Diagram":        str(result.needs_diagram),
                    "Difficulty":     result.difficulty_rating,
                    "LTM hint":       ltm_hint[:80] if ltm_hint else "none", 
                },
            )
            logger.info(
                f"[Explainer] done | steps={len(result.steps)} "
                f"manim={bool(manim_code)} difficulty={result.difficulty_rating} "
                f"ltm_personalised={bool(ltm_hint)}"
            )

            return {
                "explainer_output": explainer_dict,
                "final_response":   final_md,
                "conversation_log":   [final_md],  
                "manim_scene_code": manim_code,
                "hitl_required":    True,
                "hitl_type":        "satisfaction",
                "hitl_reason":      "Explanation delivered — awaiting student feedback.",
                "follow_up_question": None,
                "student_satisfied":  None,
            }

        except Exception as e:
            logger.error(f"[Explainer] failed: {e}")
            raise Agent_Exception(e, sys)