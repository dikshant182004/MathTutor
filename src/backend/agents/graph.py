from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from agents import *


class BaseAgent:

    def __init__(self):

        key = _get_secret("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets.")
        
        self.llm = ChatGroq(
            api_key=key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
        )
        
        self.media_processor = MediaProcessor()
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile(checkpointer=checkpointer)



class MathTutorWorkflow(
    ParserAgent,
    IntentRouterAgent,
    SolverAgent,
    VerifierAgent,
    ExplainerAgent,
):
    def _create_workflow(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # ── Nodes ──────────────────────────────────────────────────────────────
        graph.add_node("detect_input",    detect_input_type)
        graph.add_node("ocr_node",        ocr_node)
        graph.add_node("asr_node",        asr_node)
        graph.add_node("parser_agent",    self.parser_agent)
        graph.add_node("intent_router",   self.intent_router_agent)
        graph.add_node("solver_agent",    self.solver_agent)
        graph.add_node("tool_node",       ToolNode(SOLVER_TOOLS))  # <-- ReAct tool executor
        graph.add_node("verifier_agent",  self.verifier_agent)
        graph.add_node("hitl_node",       hitl_node)
        graph.add_node("explainer_agent", self.explainer_agent)
        graph.add_node("manim_node",      manim_node)
        graph.add_node("satisfaction_hitl",   satisfaction_hitl_node)

        # ── Entry ──────────────────────────────────────────────────────────────
        graph.set_entry_point("detect_input")

        # ── detect_input -> [ocr | asr | parser | hitl] ───────────────────────
        graph.add_conditional_edges(
            "detect_input",
            _route_after_detect,
            {
                "ocr_node":     "ocr_node",
                "asr_node":     "asr_node",
                "parser_agent": "parser_agent",
                "hitl_node":    "hitl_node",
            },
        )

        # ── ocr / asr -> parser ───────────────────────────────────────────────
        graph.add_edge("ocr_node", "parser_agent")
        graph.add_edge("asr_node", "parser_agent")

        # ── parser -> [hitl | intent_router] ─────────────────────────────────
        graph.add_conditional_edges(
            "parser_agent",
            _route_after_parser,
            {
                "hitl_node":    "hitl_node",
                "intent_router": "intent_router",
            },
        )

        # ── intent_router -> solver_agent (kick off ReAct loop) ───────────────
        graph.add_edge("intent_router", "solver_agent")

        # ── ReAct loop: solver_agent <-> tool_node ────────────────────────────
        #    solver emits tool_calls  -> tool_node executes them -> back to solver
        #    solver emits final answer (no tool_calls) -> verifier_agent
        graph.add_conditional_edges(
            "solver_agent",
            _route_solver_or_tools,
            {
                "tool_node":      "tool_node",
                "verifier_agent": "verifier_agent",
            },
        )
        # tool_node always returns to solver so it can reason over results
        graph.add_edge("tool_node", "solver_agent")

        # ── verifier -> [explainer | solver(retry) | hitl] ───────────────────
        graph.add_conditional_edges(
            "verifier_agent",
            _route_after_verifier,
            {
                "explainer_agent": "explainer_agent",
                "solver_agent":    "solver_agent",
                "hitl_node":       "hitl_node",
            },
        )

        # ── hitl -> [parser | solver] ─────────────────────────────────────────
        graph.add_conditional_edges(
            "hitl_node",
            _route_after_hitl,
            {
                "parser_agent": "parser_agent",
                "solver_agent": "solver_agent",
            },
        )

        # ── explainer -> manim -> END ─────────────────────────────────────────
        graph.add_edge("explainer_agent", "manim_node")
        graph.add_edge("manim_node",      "satisfaction_hitl")
        graph.add_conditional_edges(
            "satisfaction_hitl",
            _route_after_satisfaction,
            {
                "END":          END,
                "parser_agent": "parser_agent",
            },
        )

        return graph


# ── Module-level singletons exposed to app.py ─────────────────────────────────
workflow = MathTutorWorkflow()
chatbot  = workflow.app