"""
direct_response.py — DirectResponseAgent

Handles intents that do NOT need the solve → verify pipeline:
  explain        → concept/theorem explanations (e.g. "what is Bayes theorem?")
  hint           → nudges without full solution
  formula_lookup → formula/theorem statements
  research       → recent developments, general math knowledge (web-searched)
  generate       → practice problems, exam questions (web-searched)

WIRING FIXES vs first version:
  1. web_search_tool is a LangChain @tool whose underlying function takes
     query: str. Calling tool.invoke({"query": q}) or tool.invoke(q) both work,
     but the @tool decorator wraps it — we call the underlying function directly
     via tool.func(query) to skip LangChain overhead and avoid ToolException wrapping.

  2. AgentState keys: we write "manim_scene_code" as None explicitly so the
     manim_node doesn't find stale code from a previous solve in the same thread.

  3. payload() call now uses the correct node name "direct_response_agent"
     which matches AGENT_META in frontend/__init__.py.
"""
from __future__ import annotations

from backend.agents import *
from backend.agents.nodes import *
from backend.agents.nodes.memory.memory_manager import format_ltm_for_explainer
from backend.agents.nodes.tools.tools import web_search_tool as _ws_tool


def _web_search(query: str) -> str:
    """
    Calls web_search_tool's underlying Tavily function directly.
    This avoids LangChain ToolException wrapping and is safe to call
    synchronously from inside a LangGraph node.
    """
    try:
        # @tool wraps the function — .func gives us the raw callable
        result = _ws_tool.func(query)
        return result or ""
    except Exception as exc:
        logger.warning(f"[DirectResponse] web_search failed: {exc}")
        return ""


class DirectResponseAgent(BaseAgent):
    """
    Handles explain / hint / formula_lookup / research / generate intents.
    Produces a final_response without touching the solve → verify pipeline.
    """

    def _explain_prompt(self, problem_text: str, topic: str, ltm_hint: str) -> str:
        base = f"""You are an expert JEE mathematics teacher.

        A student has asked for an explanation — there is NO specific numeric problem to solve.
        Explain the concept, theorem, or method clearly and rigorously for a JEE student.

        STRUCTURE YOUR RESPONSE:
        1. **Core Idea** — what is it and why does it matter? (2-3 sentences)
        2. **Key Formula / Statement** — the precise mathematical statement using LaTeX ($...$)
        3. **Intuition** — a concrete analogy or geometric interpretation
        4. **When to Use** — what types of JEE problems trigger this concept?
        5. **Worked Mini-Example** — a short illustrative calculation
        6. **Common Pitfalls** — 2-3 mistakes students make

        Use $inline math$ and $$display math$$ for all expressions.

        Topic area: {topic}
        Student's question: {problem_text}"""

        if ltm_hint:
            base += f"\n\nPERSONALISATION (tailor your explanation):\n{ltm_hint}"
        return base

    def _hint_prompt(self, problem_text: str, topic: str) -> str:
        return f"""You are an expert JEE mathematics teacher giving a Socratic hint.

        The student wants a nudge — NOT a full solution.
        Give them:
        1. **Direction** — the right technique/approach (1-2 sentences, no answer)
        2. **Key Insight** — the one thing they need to notice
        3. **First Step** — describe the very first move only

        Do NOT solve the problem. Do NOT reveal the answer. Be encouraging.

        Topic: {topic}
        Problem: {problem_text}"""

    def _formula_prompt(self, problem_text: str, topic: str) -> str:
        return f"""You are an expert JEE mathematics teacher.

        The student wants a formula or theorem statement. Provide:
        1. **Formula** — precise statement in LaTeX ($$...$$)
        2. **Variables** — what each symbol means
        3. **Conditions** — when does this formula apply?
        4. **JEE Usage** — in what types of problems is this used?
        5. **Related Formulae** — 1-2 closely related results

        Topic: {topic}
        Request: {problem_text}"""

    def _research_prompt(self, problem_text: str, topic: str, web_context: str) -> str:
        ctx_block = (
            f"\n\nRECENT INFORMATION FROM THE WEB (use as primary source):\n{web_context}\n"
            if web_context
            else "\n\n(No web results available — use your training knowledge.)\n"
        )
        return f"""You are an expert mathematics educator.

        A JEE student asked about a mathematical topic beyond a standard textbook problem.
        Answer engagingly for a motivated student.
        {ctx_block}
        STRUCTURE:
        1. **Overview** — what is this about? (2-3 sentences)
        2. **Key Details** — the substantive content
        3. **Why It Matters** — relevance and real applications
        4. **Connection to JEE** — how does this connect to their studies?
        5. **Explore Next** — 1-2 topics to investigate further

        Use $inline$ or $$display$$ for any math.

        Topic: {topic}
        Question: {problem_text}"""

    def _generate_prompt(
        self,
        problem_text: str,
        topic: str,
        difficulty: str,
        web_context: str,
    ) -> str:
        ctx_block = (
            f"\n\nWEB CONTEXT (use for current exam patterns):\n{web_context}\n"
            if web_context else ""
        )
        return f"""You are an expert JEE mathematics teacher creating practice material.

        Generate high-quality practice content matching the student's request.
        {ctx_block}
        REQUIREMENTS:
        - Difficulty: {difficulty} (JEE standard)
        - Topic: {topic}
        - Each problem must be self-contained
        - For MCQ: provide 4 options (A/B/C/D), mark correct answer in Answer Key
        - Use $$display math$$ for equations
        - Include a brief answer key at the end

        FORMAT:
        **Problem 1:**
        [problem statement]

        **Problem 2:**
        [problem statement]

        ---
        **Answer Key:**
        1. [answer]
        2. [answer]

        Student's request: {problem_text}"""

    def direct_response_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            topic         = parsed.get("topic") or "mathematics"

            plan          = state.get("solution_plan") or {}
            intent_type   = plan.get("intent_type", "explain")
            difficulty    = plan.get("difficulty") or "medium"

            ltm_context   = state.get("ltm_context") or {}
            ltm_hint      = format_ltm_for_explainer(ltm_context, topic)

            # ── Web search for research / generate ───────────────────────────
            web_context = ""
            if intent_type in ("research", "generate"):
                if intent_type == "research":
                    query = f"{problem_text} mathematics {topic} recent 2024 2025"
                else:
                    query = f"JEE {topic} {difficulty} exam questions {problem_text[:60]}"
                logger.info(f"[DirectResponse] Web search: {query[:80]}")
                web_context = _web_search(query)
                logger.info(f"[DirectResponse] Web context: {len(web_context)} chars")

            # ── Build prompt ──────────────────────────────────────────────────
            if intent_type == "explain":
                prompt = self._explain_prompt(problem_text, topic, ltm_hint)
            elif intent_type == "hint":
                prompt = self._hint_prompt(problem_text, topic)
            elif intent_type == "formula_lookup":
                prompt = self._formula_prompt(problem_text, topic)
            elif intent_type == "research":
                prompt = self._research_prompt(problem_text, topic, web_context)
            elif intent_type == "generate":
                prompt = self._generate_prompt(problem_text, topic, difficulty, web_context)
            else:
                prompt = self._explain_prompt(problem_text, topic, ltm_hint)

            # ── LLM call ──────────────────────────────────────────────────────
            human_msg = HumanMessage(content=problem_text or prompt)
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content  = (response.content or "").strip()

            if not content:
                content = (
                    f"I wasn't able to generate a response for your request. "
                    f"Please try rephrasing."
                )

            intent_headers = {
                "explain":        "## 📖 Explanation",
                "hint":           "## 💡 Hint",
                "formula_lookup": "## 📐 Formula",
                "research":       "## 🔬 Research",
                "generate":       "## 📝 Practice Problems",
            }
            header   = intent_headers.get(intent_type, "## 📘 Response")
            final_md = f"{header}\n\n{content}"

            # ── Stub solver/verifier so store_ltm / downstream nodes don't crash
            stub_solver = {
                "solution":         content,
                "final_answer":     "",
                "rag_context_used": False,
                "calculator_used":  bool(web_context),
            }
            stub_verifier = {
                "status":        "correct",
                "verdict":       f"Direct response — intent={intent_type}. No verification needed.",
                "suggested_fix": None,
                "confidence":    1.0,
                "hitl_reason":   None,
            }

            payload(
                state, "direct_response_agent",
                summary=(
                    f"intent={intent_type} | topic={topic} | "
                    f"web={'yes' if web_context else 'no'} | "
                    f"personalised={'yes' if ltm_hint else 'no'}"
                ),
                fields={
                    "Intent":     intent_type,
                    "Topic":      topic,
                    "Web search": "yes" if web_context else "no",
                    "LTM hint":   ltm_hint[:80] if ltm_hint else "none",
                    "Preview":    content[:120],
                },
            )
            logger.info(
                f"[DirectResponse] done | intent={intent_type} topic={topic} "
                f"web={bool(web_context)} ltm={bool(ltm_hint)}"
            )

            return {
                "messages":           [human_msg, response],
                "solver_output":      stub_solver,
                "verifier_output":    stub_verifier,
                "safety_passed":      True,
                "explainer_output":   None,
                "manim_scene_code":   None,   # prevent stale code from being rendered
                "final_response":     final_md,
                "conversation_log":   [final_md],
                "hitl_required":      True,
                "hitl_type":          "satisfaction",
                "hitl_reason":        "Response delivered — was this helpful?",
                "follow_up_question": None,
                "student_satisfied":  None,
            }

        except Exception as e:
            logger.error(f"[DirectResponse] failed: {e}")
            raise Agent_Exception(e, sys)