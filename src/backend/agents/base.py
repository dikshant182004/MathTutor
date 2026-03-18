from __future__ import annotations

from langchain_groq import ChatGroq

from agents import MediaProcessor
from utils.helper import _get_secret


class BaseAgent:
    """
    Shared base for all agent nodes.

    Node mixins (e.g. SolverAgent, ParserAgent) can safely inherit this so they
    always have access to `self.llm` and any shared utilities.
    """

    def __init__(self):
        key = _get_secret("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY is not set — add it to .env (local) or Streamlit Cloud secrets."
            )

        self.llm = ChatGroq(
            api_key=key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
        )

        # Used by input nodes (OCR/ASR) and any media-based tools.
        self.media_processor = MediaProcessor()

