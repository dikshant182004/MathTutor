from agents import *

class ExplainerAgent(BaseAgent):

    def explainer_agent(self, state: AgentState) -> AgentState:
        try:
            parsed = state.get("parsed_data") or {}
            problem_text = parsed.get("problem_text") or ""
            solver_out = state.get("solver_output") or {}
            solution = solver_out.get("solution", "")
            plan = state.get("solution_plan") or {}
            needs_viz = plan.get("needs_visualization", False)
            viz_hint = plan.get("visualization_hint", "")

            # ── Step 1: Structured explanation via plain JSON (NO with_structured_output) ──
            # Reason: Groq's function-calling (used by with_structured_output) produces
            # 'tool_use_failed' 400 errors when the schema contains long string fields.
            # Solution: ask for raw JSON as plain text, parse manually.
            prompt = f"""You are a patient, encouraging JEE math tutor explaining to a student.

Produce a JSON object with EXACTLY these keys (no extra keys, no markdown fences):

{{
  "step_by_step": [
    "Step description with full mathematical working, formulas, substitutions and the result"
  ],
  "explanation": "One-paragraph conceptual summary of the approach.",
  "key_concepts": ["Concept 1", "Concept 2", "Concept 3"],
  "common_mistakes": ["Mistake 1", "Mistake 2"],
  "difficulty_rating": "easy | medium | hard"
}}

Rules:
- step_by_step: each entry MUST show the actual math — formulas, numbers, algebra — not just a description.
- Return ONLY the JSON object. No preamble, no explanation, no markdown backticks.

Problem:
{problem_text}

Full verified solution (use this as your source of truth for all numbers):
{solution}"""

            plain_llm = self.llm   # plain ChatGroq — no bind_tools, no structured_output
            raw_response = plain_llm.invoke([HumanMessage(content=prompt)])
            raw_text = (raw_response.content or "").strip()

            # Strip markdown fences if model adds them anyway
            if raw_text.startswith("```"):
                raw_text = "\n".join(
                    line for line in raw_text.splitlines()
                    if not line.strip().startswith("```")
                ).strip()

            # Parse JSON — fall back to safe defaults if malformed
            try:
                import json as _json
                parsed_json = _json.loads(raw_text)
            except Exception as parse_err:
                logger.warning(f"[Explainer] JSON parse failed ({parse_err}), using fallback")
                parsed_json = {
                    "step_by_step": [solution],
                    "explanation": solution[:500],
                    "key_concepts": [],
                    "common_mistakes": [],
                    "difficulty_rating": "medium",
                }

            # Build a simple namespace so the rest of the code can use result.X
            class _ExplainerResult:
                def __init__(self, d: dict):
                    self.step_by_step      = d.get("step_by_step") or []
                    self.explanation       = d.get("explanation") or ""
                    self.key_concepts      = d.get("key_concepts") or []
                    self.common_mistakes   = d.get("common_mistakes") or []
                    self.difficulty_rating = d.get("difficulty_rating") or "medium"
                def model_dump(self):
                    return {
                        "step_by_step":      self.step_by_step,
                        "explanation":       self.explanation,
                        "key_concepts":      self.key_concepts,
                        "common_mistakes":   self.common_mistakes,
                        "difficulty_rating": self.difficulty_rating,
                    }

            result = _ExplainerResult(parsed_json)

            # ── Step 2: Generate Manim code separately (plain text call) ──────
            # Only requested when needs_viz=True. A plain non-structured call
            # avoids the Groq 400 'tool_use_failed' error caused by large code
            # strings being jammed into a function-calling schema.
            manim_scene_code = None
            manim_scene_description = None
            if needs_viz:
                manim_prompt = f"""Generate complete, runnable Manim Community Edition Python code for a Scene that visualises the following math solution.

The animation should: {viz_hint or "illustrate the key mathematical concept step by step"}.

Requirements:
- Import from manim (not manimlib): from manim import *
- Define exactly ONE class that extends Scene
- Be fully self-contained (no external files, no user input)
- Use MathTex for all equations
- Use ThreeDScene if the problem involves 3D geometry

Problem: {problem_text}

Solution summary: {solution[:800]}

Reply with ONLY the Python code. No explanation, no markdown fences, no preamble."""

                plain_llm = self.llm  # plain ChatGroq, no bind_tools/structured_output
                manim_response = plain_llm.invoke([HumanMessage(content=manim_prompt)])
                raw_code = (manim_response.content or "").strip()
                # Strip accidental markdown fences if model adds them
                if raw_code.startswith("```"):
                    raw_code = "\n".join(
                        line for line in raw_code.splitlines()
                        if not line.strip().startswith("```")
                    ).strip()
                if raw_code and "class" in raw_code and "Scene" in raw_code:
                    manim_scene_code = raw_code
                    manim_scene_description = (
                        f"Animation visualising: {viz_hint or problem_text[:120]}"
                    )
                    logger.info(f"[Explainer] Manim code generated ({len(raw_code)} chars)")
                else:
                    logger.warning("[Explainer] Manim code generation returned unusable output")

            # Merge manim fields into the explainer output dict
            explainer_dict = result.model_dump()
            explainer_dict["manim_scene_code"] = manim_scene_code
            explainer_dict["manim_scene_description"] = manim_scene_description
            state["explainer_output"] = explainer_dict

            md_parts: list[str] = []
            md_parts.append(f"## 📘 Solution")
            md_parts.append(f"**Problem:** {problem_text}\n")
            md_parts.append("---")

            # ── Conceptual overview ───────────────────────────────────────────
            if result.explanation:
                md_parts.append(f"**Overview:** {result.explanation}\n")

            # ── Step-by-step with full mathematical working ───────────────────
            if result.step_by_step:
                md_parts.append("### Step-by-Step Working\n")
                for i, step in enumerate(result.step_by_step, 1):
                    md_parts.append(f"**Step {i}.** {step}\n")

            # ── Final answer ──────────────────────────────────────────────────
            final = solver_out.get("final_answer", "")
            if final:
                md_parts.append("---")
                md_parts.append(f"### ✅ Final Answer\n\n> {final}\n")

            # ── Key concepts ──────────────────────────────────────────────────
            if result.key_concepts:
                md_parts.append("---")
                md_parts.append("### 💡 Key Concepts\n")
                for c in result.key_concepts:
                    md_parts.append(f"- {c}")
                md_parts.append("")

            # ── Common mistakes ───────────────────────────────────────────────
            if result.common_mistakes:
                md_parts.append("### ⚠️ Common Mistakes to Avoid\n")
                for m in result.common_mistakes:
                    md_parts.append(f"- {m}")
                md_parts.append("")

            rich_md = "\n".join(md_parts)

            # Store ONLY in final_response — NOT in state["messages"].
            # app.py streams final_response directly to the chat bubble.
            # Writing to state["messages"] as well causes _load_history to
            # show the same content twice (once from messages, once from
            # final_response).
            state["final_response"] = rich_md
            # Keep messages clean — store a minimal marker so history reload
            # knows a solution was produced, but _load_history skips it
            # because it starts with "## 📘 Solution" (filtered in _load_history).
            from langchain_core.messages import AIMessage as _AIMsg
            state["messages"] = [_AIMsg(content=rich_md)]

            _log_payload(state, "explainer_agent",
                summary=f"{len(result.step_by_step)} steps | difficulty: {result.difficulty_rating}",
                fields={
                    "Steps":           str(len(result.step_by_step)),
                    "Key concepts":    ", ".join(result.key_concepts[:3]) if result.key_concepts else None,
                    "Common mistakes": ", ".join(result.common_mistakes[:2]) if result.common_mistakes else None,
                    "Manim code":      "Yes — scene generated" if manim_scene_code else None,
                    "Difficulty":      result.difficulty_rating,
                },
            )
            logger.info(f"[Explainer] steps={len(result.step_by_step)} manim={bool(manim_scene_code)}")
            return state

        except Exception as e:
            logger.error(f"[Explainer] failed: {e}")
            raise Agent_Exception(e, sys)