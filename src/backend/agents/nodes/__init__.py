from backend.agents.base import BaseAgent
from backend.agents.state import AgentState
from backend.agents.utils.helper import _log_payload as payload, _render_markdown as render_md
from backend.agents.utils.artifacts import (
    ParserOutput,
    IntentRouterOutput,
    VerifierOutput,
    ExplainerOutput,
    GuardrailOutput,
    SafetyOutput,
)
from backend.agents.nodes.tools.tools import rag_tool, web_search_tool, calculator_tool

# Keywords that warrant an immediate block without LLM call
_HARD_BLOCK_KEYWORDS = [
    "synthesise", "synthesize", "synthesis route",
    "how to make", "manufacture", "explosive",
    "detonate", "nerve agent", "poison",
    "self-harm", "suicide method",
    "child", "minor",          # context-dependent but always worth LLM review
]

__all__ = [
    "BaseAgent", "AgentState", "payload", "render_md", "_HARD_BLOCK_KEYWORDS",
    "ParserOutput", "IntentRouterOutput", "VerifierOutput", "ExplainerOutput", "GuardrailOutput", "SafetyOutput",
    "rag_tool", "web_search_tool", "calculator_tool"
]

