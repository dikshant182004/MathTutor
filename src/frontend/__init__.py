import streamlit as st
from pathlib import Path
from typing import Optional

__all__=["st", "Path", "Optional"]

# ── Node metadata ─────────────────────────────────────────────────────────────
AGENT_META: dict[str, dict] = {
    "detect_input":          {"icon": "🔍", "label": "Detect Input Type"},
    "ocr_node":              {"icon": "📸", "label": "OCR  (Image → Text)"},
    "asr_node":              {"icon": "🎤", "label": "ASR  (Audio → Text)"},
    "guardrail_agent":       {"icon": "🛡️", "label": "Guardrail Agent"},
    "retrieve_ltm":          {"icon": "🧠", "label": "Retrieve Long-Term Memory"},
    "parser_agent":          {"icon": "🧩", "label": "Parser Agent"},
    "intent_router":         {"icon": "🗺️",  "label": "Intent Router"},
    "solver_agent":          {"icon": "🧮", "label": "Solver Agent (ReAct)"},
    "tool_node":             {"icon": "🔧", "label": "Tool Executor"},
    "verifier_agent":        {"icon": "✅", "label": "Verifier / Critic"},
    "safety_agent":          {"icon": "🔒", "label": "Safety Agent"},
    "explainer_agent":       {"icon": "📚", "label": "Explainer / Tutor"},
    "direct_response_node":  {"icon": "💬", "label": "Direct Response Agent"},
    "manim_node":            {"icon": "🎬", "label": "Manim Visualiser"},
    "hitl_node":             {"icon": "🙋", "label": "Human-in-the-Loop"},
    "store_ltm":             {"icon": "💾", "label": "Store Long-Term Memory"},
}

# ── Tool metadata ─────────────────────────────────────────────────────────────
TOOL_META: dict[str, dict] = {
    "rag_tool":        {"icon": "📄", "label": "RAG — PDF search"},
    "web_search_tool": {"icon": "🌐", "label": "Web Search (Tavily)"},
    "calculator_tool": {"icon": "🔢", "label": "Symbolic Calculator"},
}

# ── Answer node filter ────────────────────────────────────────────────────────
# Both explainer_agent and direct_response_node produce final_response
ANSWER_NODES: set[str] = {"explainer_agent", "direct_response_node"}

# ── HITL banner prefixes ──────────────────────────────────────────────────────
HITL_PREFIX     = "__HITL__:"
HITL_SAT_PREFIX = "__SATQ__:"