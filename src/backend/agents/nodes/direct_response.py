from __future__ import annotations

from backend.agents import *
from backend.agents.nodes import *
from backend.agents.utils.helper import _parse_xml_response
from backend.agents.nodes.memory.memory_manager import format_ltm_for_explainer
from backend.agents.nodes.tools.tools import web_search_tool as _ws_tool

tool_signals: list = []
 
def _web_search(query: str) -> str:
    """
    Call web_search_tool's underlying Tavily function directly.
    .func() bypasses LangChain ToolException wrapping for sync use inside a node.
    """
    try:
        result = _ws_tool.func(query)
        return result or ""
    except Exception as exc:
        logger.warning(f"[DirectResponse] web_search failed: {exc}")
        return ""
 
 
class DirectResponseAgent(BaseAgent):
    """
    Handles explain / hint / formula_lookup / research / generate intents.
    Uses a single LLM call that returns a markdown response inside <content> tags.
    """
 
    # ── Prompt builders ───────────────────────────────────────────────────────
 
    def _explain_prompt(self, problem_text: str, topic: str, ltm_hint: str) -> str:
        base = f"""You are an expert teacher. A student has asked for an explanation.
        Explain the concept, theorem, or method clearly and rigorously.
        
        RESPONSE STRUCTURE (content field):
        1. **Core Idea** — what is it and why does it matter? (2-3 sentences)
        2. **Key Formula / Statement** — the precise mathematical statement using LaTeX ($...$)
        3. **Intuition** — a concrete analogy or geometric interpretation
        4. **When to Use** — what types of problems call for this concept?
        5. **Worked Mini-Example** — a short illustrative calculation
        6. **Common Pitfalls** — 2-3 mistakes students typically make
        
        Use $inline math$ and $$display math$$ for all expressions.
        Topic: {topic}
        Question: {problem_text}"""
 
        if ltm_hint:
            base += f"\n\nPERSONALISATION — tailor your explanation to this student:\n{ltm_hint}"
        return base
 
    def _hint_prompt(self, problem_text: str, topic: str) -> str:
        return f"""You are an expert teacher giving a Socratic hint.
        The student wants a nudge — NOT a full solution.
        
        RESPONSE STRUCTURE (content field):
        1. **Direction** — the right technique or approach (1-2 sentences, no answer revealed)
        2. **Key Insight** — the one thing they need to notice
        3. **First Step** — describe the very first move only
        
        Do NOT solve the problem. Do NOT reveal the answer. Be encouraging.
        Topic: {topic}
        Problem: {problem_text}"""
 
    def _formula_prompt(self, problem_text: str, topic: str) -> str:
        return f"""You are an expert teacher. The student wants a formula or theorem statement.
 
        RESPONSE STRUCTURE (content field):
        1. **Formula** — precise statement in LaTeX ($$...$$)
        2. **Variables** — what each symbol means
        3. **Conditions** — when does this formula apply?
        4. **Usage** — in what types of problems is this used?
        5. **Related Formulae** — 1-2 closely related results
        Topic: {topic}
        Request: {problem_text}"""
        
    def _research_prompt(self, problem_text: str, topic: str, web_context: str) -> str:
        ctx_block = (
            f"\n\nINFORMATION FROM WEB SEARCH (use as primary source):\n{web_context}\n"
            if web_context
            else "\n\n(No web results available — use your knowledge.)\n"
        )
        return f"""You are an expert educator answering a research or knowledge question.
        {ctx_block}
        RESPONSE STRUCTURE (content field):
        1. **Overview** — what is this about? (2-3 sentences)
        2. **Key Details** — the substantive content
        3. **Why It Matters** — relevance and real-world applications
        4. **Connections** — how does this relate to other ideas in this field?
        5. **Explore Next** — 1-2 topics worth investigating further
        
        Use $inline$ or $$display$$ for any mathematical expressions.
        Topic: {topic}
        Question: {problem_text}"""
 
    def _generate_prompt(
        self,
        problem_text: str,
        topic:        str,
        difficulty:   str,
        web_context:  str,
    ) -> str:
        ctx_block = (
            f"\n\nWEB CONTEXT (use for current patterns and styles):\n{web_context}\n"
            if web_context else ""
        )
        return f"""You are an expert educator creating practice material.
        Generate high-quality problems or examples matching the student's request.
        {ctx_block}
        REQUIREMENTS for content field:
        - Difficulty level: {difficulty}
        - Topic: {topic}
        - Each problem must be self-contained
        - For multiple choice: provide 4 options (A/B/C/D), mark correct answer in Answer Key
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
        
    # ── Main node ─────────────────────────────────────────────────────────────
 
    def direct_response_agent(self, state: AgentState) -> dict:
        try:
            parsed        = state.get("parsed_data") or {}
            problem_text  = parsed.get("problem_text") or ""
            topic         = parsed.get("topic") or "general"
 
            plan          = state.get("solution_plan") or {}
            intent_type   = plan.get("intent_type", "explain")
            difficulty    = plan.get("difficulty") or "medium"
            strategy      = plan.get("solver_strategy") or ""
 
            ltm_context   = state.get("ltm_context") or {}
            ltm_hint      = format_ltm_for_explainer(ltm_context, topic)

            tool_signals.clear()  # ← wipe signals from any previous call
 
            # ── Web search for research / generate intents ────────────────────
            web_context  = ""
            search_query = ""
 
            if intent_type == "research":
                search_query = f"{topic} {problem_text[:100]}"
                logger.info(f"[DirectResponse] Tavily search (research): {search_query[:100]}")
                tool_signals.append({"name": "web_search_tool", "args": {"query": search_query}})
                web_context = _web_search(search_query)
                logger.info(f"[DirectResponse] Web context: {len(web_context)} chars")
 
            elif intent_type == "generate":
                search_query = f"{topic} {difficulty} practice problems {problem_text[:60]}"
                logger.info(f"[DirectResponse] Tavily search (generate): {search_query[:100]}")
                tool_signals.append({"name": "web_search_tool", "args": {"query": search_query}})
                web_context = _web_search(search_query)
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
                logger.warning(f"[DirectResponse] Unknown intent '{intent_type}' — defaulting to explain")
                prompt = self._explain_prompt(problem_text, topic, ltm_hint)
 
            _SYSTEM = (
                "Respond using EXACTLY these XML tags — nothing outside them:\n"
                "<content>\n"
                "YOUR FULL MARKDOWN RESPONSE HERE\n"
                "</content>\n"
                "Rules:\n"
                "- Write all your explanation inside <content>...</content>.\n"
                "- Use markdown freely inside <content> (headers, bold, LaTeX $...$, lists).\n"
                "- Do NOT add any text before <content>."
            )

            raw_response = self.llm.invoke(
                [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
            )
            raw_text = (raw_response.content or "").strip()
            content = _parse_xml_response(raw_text)

            if not content:
                content = (
                    "I wasn't able to generate a response for your request. "
                    "Please try rephrasing."
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

 
            # ── Stubs so downstream nodes (store_ltm etc.) don't crash ─────────
            stub_solver = {
                "solution":         content,
                "final_answer":     "",
                "rag_context_used": False,
                "calculator_used":  False,
                "web_search_used":  bool(web_context),
            }
            stub_verifier = {
                "status":        "correct",
                "verdict":       f"Direct response — intent={intent_type}. No verification needed.",
                "suggested_fix": None,
                "confidence":    1.0,
                "hitl_reason":   None,
            }
 
            # ── Payload ───────────────────────────────────────────────────────
            payload(
                state, "direct_response_agent",
                summary=(
                    f"intent={intent_type} | topic={topic} | difficulty={difficulty} | "
                    f"web={'yes' if web_context else 'no'} | "
                    f"personalised={'yes' if ltm_hint else 'no'}"
                ),
                fields={
                    "Intent":          intent_type,
                    "Topic":           topic,
                    "Difficulty":      difficulty,
                    "Strategy":        strategy[:100] if strategy else "n/a",
                    "Web search":      f"yes — {search_query[:80]}" if web_context else "no",
                    "Web result size": f"{len(web_context)} chars" if web_context else "0",
                    "LTM hint":        ltm_hint[:80] if ltm_hint else "none",
                    "Response size":   f"{len(content)} chars",
                    "Preview":         content[:120],
                },
            )
            logger.info(
                f"[DirectResponse] done | intent={intent_type} topic={topic} "
                f"web={bool(web_context)} ltm={bool(ltm_hint)}"
            )
 
            return {
                "messages":           [HumanMessage(content=problem_text or prompt), AIMessage(content=content)],
                "solver_output":      stub_solver,
                "verifier_output":    stub_verifier,
                "safety_passed":      True,
                "explainer_output":   None,
                "final_response":     final_md,
                "conversation_log":   [final_md],
                "hitl_required":      True,
                "hitl_type":          "satisfaction",
                "hitl_reason":        "Response delivered — was this helpful?",
                "follow_up_question": None,
                "student_satisfied":  None,
                "agent_payload_log":          state.get("agent_payload_log") or [],
                "direct_response_tool_calls": tool_signals,
            }
 
        except Exception as e:
            logger.error(f"[DirectResponse] failed: {e}")
            raise Agent_Exception(e, sys)