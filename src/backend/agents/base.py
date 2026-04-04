from __future__ import annotations

from langchain_groq import ChatGroq
from backend.agents.utils.helper import _get_secret, MediaProcessor


class BaseAgent:
    """
    Shared base for all agent nodes.

    Node mixins (e.g. SolverAgent, ParserAgent) can safely inherit this so they
    always have access to `self.llm` and any shared utilities.

    self.llm              — llama-3.3-70b-versatile — for reasoning nodes like guardrail, parser, router

    self.reserve_llm      — llama-3.3-70b-versatile — for heavy reasoning nodes
                            (solver, verifier, explainer)
    """

    def __init__(self):
        key = _get_secret("GROQ_API_KEY")
        key_2 = _get_secret("GROQ_API_KEY_2")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets."
            )
        if not key_2:
            raise ValueError(
                "GROQ_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets."
            )

        self.llm = ChatGroq(
            api_key=key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048,
            max_retries=2,
        )

        # well we are using different api key for our solver and direct_response_agent
        # to avoid groq rate limiting errors 
        self.reserve_llm = ChatGroq(
            api_key=key_2,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=2048,
            max_retries=2,
        )

        # Used by input nodes (OCR/ASR) and any media-based tools.
        self.media_processor = MediaProcessor()