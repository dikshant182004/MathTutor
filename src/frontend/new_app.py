from __future__ import annotations

import html
import os
import time
import uuid
from typing import Optional

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from backend.agents import chatbot, retrieve_all_threads
from backend.tools.tools import clear_store, get_store_info, ingest_pdf

st.set_page_config(
    page_title="Math Tutor Agent",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.makedirs("uploads",       exist_ok=True)
os.makedirs("manim_outputs", exist_ok=True)

st.markdown("""
<style>
/* ── Agent activity card ─────────────────────────────────────────────── */
.step-card {
    border-left: 3px solid #4CAF50;
    border-radius: 6px;
    padding: 8px 12px;
    margin-bottom: 6px;
    background: #0e1117;
    font-size: 0.84rem;
    color: #e0e0e0;
    line-height: 1.4;
}
.step-card.active {
    border-left-color: #2196F3;
    background: #0d1f33;
    animation: pulse 1.2s ease-in-out infinite;
}
.step-card.tool {
    border-left-color: #FF9800;
    background: #1a130a;
}
.step-card.hitl {
    border-left-color: #F44336;
    background: #1a0808;
}
.step-card.done {
    border-left-color: #4CAF50;
    opacity: 0.72;
}
.step-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}
.step-label  { font-weight: 600; }
.step-ts     { font-size: 0.70rem; color: #666; }
.step-detail { font-size: 0.76rem; color: #999; margin-top: 3px; }

@keyframes pulse {
    0%, 100% { opacity: 1.0; }
    50%       { opacity: 0.5; }
}

/* ── Payload details (collapsible) ───────────────────────────────── */
.step-payload {
    margin-top: 6px;
    border-top: 1px solid #333;
    padding-top: 5px;
}
.step-payload summary {
    cursor: pointer;
    font-size: 0.72rem;
    color: #888;
    user-select: none;
    list-style: none;
    outline: none;
}
.step-payload summary::-webkit-details-marker { display: none; }
.step-payload summary::before { content: "▶ "; font-size: 0.65rem; }
details[open] .step-payload summary::before { content: "▼ "; }
.payload-table {
    width: 100%;
    margin-top: 4px;
    border-collapse: collapse;
    font-size: 0.72rem;
}
.payload-table td {
    padding: 2px 4px;
    vertical-align: top;
    border-bottom: 1px solid #2a2a2a;
    color: #ccc;
    word-break: break-word;
}
.payload-table td:first-child {
    color: #888;
    white-space: nowrap;
    padding-right: 8px;
    font-weight: 600;
    width: 36%;
}

/* ── HITL banner ─────────────────────────────────────────────────────── */
.hitl-banner {
    border: 1px solid #F44336;
    border-radius: 8px;
    padding: 14px 18px;
    background: #1a0808;
    margin-bottom: 14px;
}
.hitl-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ff6b6b;
    margin-bottom: 6px;
}
.hitl-body { color: #e0e0e0; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

AGENT_META: dict[str, dict] = {
    "detect_input":    {"icon": "🔍", "label": "Detect Input Type"},
    "ocr_node":        {"icon": "📸", "label": "OCR  (Image → Text)"},
    "asr_node":        {"icon": "🎤", "label": "ASR  (Audio → Text)"},
    "parser_agent":    {"icon": "🧩", "label": "Parser Agent"},
    "intent_router":   {"icon": "🗺️",  "label": "Intent Router"},
    "solver_agent":    {"icon": "🧮", "label": "Solver Agent (ReAct)"},
    "tool_node":       {"icon": "🔧", "label": "Tool Executor"},
    "verifier_agent":  {"icon": "✅", "label": "Verifier / Critic"},
    "explainer_agent": {"icon": "📚", "label": "Explainer / Tutor"},
    "manim_node":      {"icon": "🎬", "label": "Manim Visualiser"},
    "hitl_node":       {"icon": "🙋", "label": "Human-in-the-Loop"},
}

TOOL_META: dict[str, dict] = {
    "rag_tool":               {"icon": "📄", "label": "RAG — PDF search"},
    "web_search_tool":        {"icon": "🌐", "label": "Web Search"},
    "python_calculator_tool": {"icon": "🔢", "label": "Python Calculator"},
}

# Only explainer_agent produces the user-facing answer.
# solver_agent output must NOT be streamed — it's raw reasoning text that
# would duplicate (and precede) the explainer's formatted markdown.
ANSWER_NODES = {"explainer_agent"}
SATISFACTION_NODE = "satisfaction_hitl"


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION-STATE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _new_tid() -> str:
    return str(uuid.uuid4())


def _cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}, "metadata": {"thread_id": tid}}


def _init_session() -> None:
    defaults: dict = {
        "thread_id":       _new_tid(),
        "chat_threads":    [],
        "message_history": [],
        "pdf_ingested":    False,
        "hitl_pending":    False,
        "hitl_question":   None,
        "hitl_type":       "",       # "clarification" | "satisfaction"
        "activity_log":    [],      # [{node, icon, label, status, detail, ts}, ...]
        "current_node":    None,    # name of the node currently streaming
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["chat_threads"]:
        try:
            st.session_state["chat_threads"] = retrieve_all_threads()
        except Exception:
            st.session_state["chat_threads"] = []

    _register(st.session_state["thread_id"])


def _register(tid: str) -> None:
    if tid not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(tid)


def reset_chat() -> None:
    old = st.session_state.get("thread_id")
    if old:
        clear_store(old)
    tid = _new_tid()
    st.session_state.update(
        thread_id       = tid,
        message_history = [],
        pdf_ingested    = False,
        hitl_pending    = False,
        hitl_question   = None,
        hitl_type       = "",
        activity_log    = [],
        current_node    = None,
    )
    _register(tid)


# Prefixes used when storing HITL context in message_history
_HITL_PREFIX      = "__HITL__:"
_HITL_SAT_PREFIX  = "__SATQ__:"

def _load_history(tid: str) -> list[dict]:
    """
    Re-hydrate the visible conversation from the LangGraph checkpointer.

    Single source of truth rules:
    - User messages:  HumanMessage in state["messages"], skip internal plumbing
    - AI answer:      state["final_response"] ONLY — AIMessages that start with
                      "## 📘 Solution" are always skipped from state["messages"]
                      to prevent the same content appearing twice
    - HITL banners:   state["conversation_log"] (prefixed strings)
    """
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        vals = snap.values or {}
        msgs = vals.get("messages", [])
        conv = vals.get("conversation_log") or []
        final_resp = vals.get("final_response", "")
    except Exception:
        return []

    history: list[dict] = []
    seen: set[str] = set()

    _skip_human = (
        "[Human feedback]", "[Human clarification]",
        "[Feedback]:", "[Clarification]:",
        "Solve this problem:", "[Clarification]",
    )

    for m in msgs:
        raw = m.content
        if isinstance(raw, list):
            raw = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in raw
            )
        content = str(raw or "").strip()
        if not content:
            continue

        if isinstance(m, HumanMessage):
            if any(content.startswith(p) for p in _skip_human):
                continue
            key = f"u:{content[:100]}"
            if key not in seen:
                seen.add(key)
                history.append({"role": "user", "content": content})

        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            # Always skip explainer markdown from messages — shown via final_response
            if content.startswith("## 📘"):
                continue
            # Skip HITL prefixed entries — come from conversation_log below
            if content.startswith((_HITL_PREFIX, _HITL_SAT_PREFIX)):
                continue
            # Skip short stubs / internal echoes
            if len(content) < 30:
                continue
            key = f"a:{content[:100]}"
            if key not in seen:
                seen.add(key)
                history.append({"role": "assistant", "content": content})

    # ── Inject final_response (the one true explainer output) ────────────────
    if final_resp and final_resp.strip():
        key = f"a:{final_resp[:100]}"
        if key not in seen:
            seen.add(key)
            history.append({"role": "assistant", "content": final_resp.strip()})

    # ── Append HITL banners from conversation_log ─────────────────────────────
    for entry in conv:
        entry = str(entry).strip()
        if not entry:
            continue
        key = f"a:{entry[:100]}"
        if key not in seen:
            seen.add(key)
            history.append({"role": "assistant", "content": entry})

    return history


def _check_hitl(tid: str) -> tuple[bool, Optional[str], str]:
    """
    Return (is_interrupted, question_text, hitl_type).
    hitl_type is "clarification" | "satisfaction" | ""

    Both hitl_node and satisfaction_hitl_node use interrupt() internally,
    so we detect them via snap.next (the node whose interrupt() paused the graph).
    """
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        next_nodes = list(snap.next or [])

        # Determine type from which node is next (i.e. paused mid-execution)
        if "satisfaction_hitl" in next_nodes:
            hitl_type = "satisfaction"
        elif "hitl_node" in next_nodes:
            hitl_type = "clarification"
        else:
            return False, None, ""

        # Extract question from the interrupt payload
        for task in (snap.tasks or []):
            for iv in getattr(task, "interrupts", []):
                val = getattr(iv, "value", {}) or {}
                q   = (val.get("question") or val.get("reason")
                       or "Please clarify the problem.")
                return True, str(q), hitl_type

        # Fallback question if no payload
        default_q = (
            "Are you satisfied with this explanation?\n\nClick **✅ Yes** to ask your next question, or **🔄 No** to re-explain."
            if hitl_type == "satisfaction"
            else "Please clarify the problem."
        )
        return True, default_q, hitl_type
    except Exception:
        pass
    return False, None, ""


# ══════════════════════════════════════════════════════════════════════════════
#  ACTIVITY LOG HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _add_step(
    node:    str,
    status:  str  = "done",
    detail:  str  = "",
    payload: dict = None,   # structured fields from agent_payload_log
) -> None:
    """
    Append a step card to the activity log.
    `detail` and all payload values are plain text — html.escape()'d before
    rendering so no raw HTML can leak into the card template.
    """
    meta = AGENT_META.get(node, {"icon": "⚙️", "label": node})
    st.session_state["activity_log"].append({
        "node":    node,
        "icon":    meta["icon"],
        "label":   meta["label"],
        "status":  status,
        "detail":  str(detail)[:120],
        "payload": payload or {},   # {summary, fields}
        "ts":      time.strftime("%H:%M:%S"),
    })


def _mark_previous_done(current_node: str) -> None:
    """Flip any 'active' card that isn't current_node to 'done'."""
    for step in st.session_state["activity_log"]:
        if step["status"] == "active" and step["node"] != current_node:
            step["status"] = "done"


def _mark_all_done() -> None:
    for step in st.session_state["activity_log"]:
        if step["status"] == "active":
            step["status"] = "done"


def _build_payload_html(payload: dict) -> str:
    """Build the collapsible <details> section for a card from a payload dict."""
    if not payload:
        return ""
    summary_text = html.escape(str(payload.get("summary", "")))
    fields: dict = payload.get("fields", {})
    if not summary_text and not fields:
        return ""

    rows = ""
    for k, v in fields.items():
        if v:
            rows += (
                f"<tr><td>{html.escape(str(k))}</td>"
                f"<td>{html.escape(str(v))}</td></tr>"
            )

    table_html = f'<table class="payload-table">{rows}</table>' if rows else ""

    return (
        f'<details class="step-payload">'
        f'<summary>{summary_text}</summary>'
        f'{table_html}'
        f'</details>'
    )


def render_activity_panel(placeholder) -> None:
    """
    Rewrite the activity panel placeholder in-place.
    All text is html.escape()-d — no raw HTML can appear in cards.
    Cards with payload show a collapsible <details> section with the
    structured data each agent produced.
    """
    log: list[dict] = st.session_state.get("activity_log", [])

    with placeholder.container():
        st.markdown("### 🤖 Agent Activity")

        if not log:
            st.caption("Activity will appear here once you send a message.")
            return

        for step in log:
            css   = f"step-card {step['status']}"
            icon  = html.escape(step["icon"])
            label = html.escape(step["label"])
            ts    = html.escape(step["ts"])

            detail_html = ""
            if step.get("detail"):
                detail_html = f'<p class="step-detail">{html.escape(step["detail"])}</p>'

            payload_html = _build_payload_html(step.get("payload") or {})

            card = (
                f'<div class="{css}">'
                f'  <div class="step-header">'
                f'    <span class="step-label">{icon}&nbsp;{label}</span>'
                f'    <span class="step-ts">{ts}</span>'
                f'  </div>'
                f'  {detail_html}'
                f'  {payload_html}'
                f'</div>'
            )
            st.markdown(card, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  STREAM HANDLER
# ══════════════════════════════════════════════════════════════════════════════


def _process_node_update(node_name: str, state_patch: dict,
                         badge_placeholder) -> Optional[str]:
    """
    Process a single node's state patch from the "updates" stream event.

    Actions:
    1. Register / update the activity card for this node.
    2. Attach agent_payload_log to the card (collapsible details).
    3. Extract any text to stream to the user:
       - final_response  (explainer's rich markdown — always use this)
       - messages[-1].content if it's a plain AIMessage with content
         and no tool_calls (solver's final answer text)

    Returns text to yield to chat, or None.
    """
    log = st.session_state["activity_log"]
    # Add card if new node, else just mark current
    if not log or log[-1]["node"] != node_name:
        _mark_previous_done(node_name)
        _add_step(node_name, status="active")
    st.session_state["current_node"] = node_name

    # ── Attach payload to card ────────────────────────────────────────────────
    payload_log: list = state_patch.get("agent_payload_log") or []
    node_entry = None
    for entry in reversed(payload_log):
        if entry.get("node") == node_name:
            node_entry = entry
            break
    if node_entry:
        payload = {"summary": node_entry.get("summary", ""),
                   "fields":  node_entry.get("fields", {})}
        for step in reversed(log):
            if step["node"] == node_name:
                step["payload"] = payload
                if payload["summary"]:
                    step["detail"] = payload["summary"]
                step["status"] = "done"
                break

    # ── Extract text to stream ────────────────────────────────────────────────
    # Priority 1: final_response (explainer's pre-built rich markdown)
    final_resp = state_patch.get("final_response")
    if final_resp:
        badge_placeholder.empty()
        # Mark explainer card done
        for step in reversed(log):
            if step["node"] == node_name:
                step["status"] = "done"
                break
        return str(final_resp)

    # Priority 2: last message in the patch if it's a real AI answer
    # (solver final answer arrives here when it doesn't use tools)
    patch_msgs = state_patch.get("messages") or []
    if patch_msgs:
        last = patch_msgs[-1] if isinstance(patch_msgs, list) else None
        if (last and isinstance(last, AIMessage)
                and last.content
                and not getattr(last, "tool_calls", None)
                and node_name in ANSWER_NODES):
            badge_placeholder.empty()
            return str(last.content)

    # Priority 3: tool calls in patch → show tool badge
    if patch_msgs:
        last = patch_msgs[-1] if isinstance(patch_msgs, list) else None
        if last and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            names = [tc["name"] for tc in last.tool_calls]
            for n in names:
                tmeta = TOOL_META.get(n, {"icon": "🔧", "label": n})
                _add_step("tool_node", status="tool",
                          detail=f'{tmeta["icon"]} {tmeta["label"]}')
            badge_placeholder.caption("🔧 " + "  ·  ".join(names) + "…")

    return None


def _handle_chunk(chunk, badge_placeholder) -> Optional[str]:
    """
    Process one event from chatbot.stream(stream_mode="updates").

    LangGraph "updates" mode yields a plain dict per node:
      {node_name: state_patch_dict}

    Returns: str to yield to chat bubble, or None.
    """
    if isinstance(chunk, dict):
        result_text: Optional[str] = None
        for node_name, state_patch in chunk.items():
            if not isinstance(state_patch, dict):
                continue
            text = _process_node_update(node_name, state_patch, badge_placeholder)
            if text:
                result_text = text
        return result_text

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════

_init_session()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧮 Math Tutor")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    # ── PDF Study Material ─────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Study Material")
    st.caption("Upload a PDF — solver will call `rag_tool` automatically when needed.")

    uploaded_pdf = st.file_uploader(
        "Upload PDF", type=["pdf"],
        key="pdf_uploader", label_visibility="collapsed",
    )

    if uploaded_pdf is not None:
        tid  = st.session_state["thread_id"]
        info = get_store_info(tid)
        already_indexed = info and uploaded_pdf.name in info.get("filenames", [])
        if already_indexed:
            st.info(f"📚 Already indexed: **{uploaded_pdf.name}** ({info['chunks']} chunks)")
        else:
            with st.spinner(f"Embedding **{uploaded_pdf.name}** via Cohere…"):
                try:
                    stats = ingest_pdf(
                        file_bytes=uploaded_pdf.read(),
                        thread_id=tid,
                        filename=uploaded_pdf.name,
                    )
                    st.session_state["pdf_ingested"] = True
                    st.success(
                        f"✅ **{stats['filename']}**  "
                        f"{stats['pages']} pages · {stats['chunks']} chunks"
                    )
                except Exception as exc:
                    st.error(f"❌ Ingestion failed: {exc}")

    if st.session_state["pdf_ingested"]:
        info = get_store_info(st.session_state["thread_id"])
        if info:
            st.info(f"📚 **{info['filename']}**\n\n🔖 {info['chunks']} chunks indexed")

    # ── Conversation Threads ───────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Conversations")

    for tid in st.session_state["chat_threads"][::-1]:
        is_active = tid == st.session_state["thread_id"]
        label     = f"{'▶ ' if is_active else ''}{tid[:10]}…"
        if st.button(label, key=f"t_{tid}", use_container_width=True):
            st.session_state["thread_id"]       = tid
            st.session_state["message_history"]  = _load_history(tid)
            st.session_state["pdf_ingested"]     = get_store_info(tid) is not None
            st.session_state["activity_log"]     = []
            st.session_state["current_node"]     = None
            interrupted, question, htype = _check_hitl(tid)
            st.session_state["hitl_pending"]  = interrupted
            st.session_state["hitl_question"] = question
            st.session_state["hitl_type"]     = htype
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT  — 30 % activity panel | 70 % chat
# ══════════════════════════════════════════════════════════════════════════════

col_activity, col_chat = st.columns([3, 7], gap="medium")

with col_activity:
    activity_ph = st.empty()          # rewritten in-place during streaming
    render_activity_panel(activity_ph)

# ══════════════════════════════════════════════════════════════════════════════
#  CHAT COLUMN
# ══════════════════════════════════════════════════════════════════════════════

with col_chat:
    st.title("🧮 Math Tutor Agent")

    # ── Render conversation history ────────────────────────────────────────────
    # Messages with __HITL__: / __SATQ__: prefix are rendered as styled banners.
    # Everything else renders as normal chat bubbles.
    for msg in st.session_state["message_history"]:
        content_raw = msg.get("content", "")
        role        = msg["role"]

        if content_raw.startswith(_HITL_PREFIX):
            q_text = content_raw[len(_HITL_PREFIX):]
            st.markdown(
                f'<div class="hitl-banner">'
                f'<div class="hitl-title">🙋 Clarification Was Needed</div>'
                f'<div class="hitl-body">{html.escape(q_text)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif content_raw.startswith(_HITL_SAT_PREFIX):
            q_text = content_raw[len(_HITL_SAT_PREFIX):]
            st.markdown(
                f'<div class="hitl-banner" style="border-color:#4CAF50;background:#0a1a0a;">'
                f'<div class="hitl-title" style="color:#81C784;">✅ Solution Complete</div>'
                f'<div class="hitl-body">{html.escape(q_text)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            with st.chat_message(role):
                st.markdown(content_raw)

    # ══════════════════════════════════════════════════════════════════════════
    #  HITL CLARIFICATION BLOCK
    #  Replaces the input widgets while the graph is paused for human input.
    # ══════════════════════════════════════════════════════════════════════════

    if st.session_state.get("hitl_pending"):
        question   = st.session_state.get("hitl_question") or "Please clarify."
        hitl_type  = st.session_state.get("hitl_type", "clarification")
        tid        = st.session_state["thread_id"]

        is_satisfaction = (hitl_type == "satisfaction")

        # ── Activity panel card ───────────────────────────────────────────────
        log       = st.session_state["activity_log"]
        card_node = "satisfaction_hitl" if is_satisfaction else "hitl_node"
        if not log or log[-1]["node"] != card_node:
            _add_step(card_node, status="hitl", detail=question[:120])
            render_activity_panel(activity_ph)

        # ── Persist HITL question in message_history (in-memory + LangGraph) ─
        # We write a prefixed AIMessage into state["conversation_log"] — a
        # separate list from state["messages"] so we never corrupt the graph
        # message chain. _load_history reads both.
        _prefix   = _HITL_SAT_PREFIX if is_satisfaction else _HITL_PREFIX
        _stored_q = _prefix + question
        _history  = st.session_state["message_history"]
        if not _history or _history[-1].get("content") != _stored_q:
            # Append to in-memory history
            _history.append({"role": "assistant", "content": _stored_q})
            # Persist into LangGraph state under conversation_log (not messages)
            try:
                snap = chatbot.get_state(config=_cfg(tid))
                existing_log = (snap.values or {}).get("conversation_log") or []
                if not existing_log or existing_log[-1] != _stored_q:
                    chatbot.update_state(
                        _cfg(tid),
                        {"conversation_log": existing_log + [_stored_q]},
                    )
            except Exception:
                pass

        # ── Render the appropriate banner ─────────────────────────────────────
        if is_satisfaction:
            st.markdown(
                f'<div class="hitl-banner" style="border-color:#4CAF50;background:#0a1a0a;">'
                f'<div class="hitl-title" style="color:#81C784;">✅ Solution Complete</div>'
                f'<div class="hitl-body">{html.escape(question)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="hitl-banner">'
                f'<div class="hitl-title">🙋 Clarification Needed</div>'
                f'<div class="hitl-body">{html.escape(question)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ── Input form ────────────────────────────────────────────────────────
        if is_satisfaction:
            col1, col2 = st.columns(2)
            with col1:
                satisfied = st.button("✅ Yes, next question", use_container_width=True, type="primary")
            with col2:
                not_satisfied = st.button("🔄 No, re-explain", use_container_width=True)

            human_answer = None
            if satisfied:
                human_answer = "yes"
            elif not_satisfied:
                with st.form("reexplain_form", clear_on_submit=True):
                    clarify_text = st.text_area(
                        "What needs clarification?",
                        placeholder="Describe what was unclear…",
                        height=80,
                        label_visibility="visible",
                    )
                    if st.form_submit_button("Submit 🔄", use_container_width=True):
                        human_answer = clarify_text or "please re-explain"
        else:
            with st.form("hitl_form", clear_on_submit=True):
                clarify_input = st.text_area(
                    "Your answer",
                    placeholder="Type your clarification here…",
                    height=100,
                    label_visibility="collapsed",
                )
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    human_answer = clarify_input
                else:
                    human_answer = None

        if human_answer is not None and str(human_answer).strip():
            human_answer = str(human_answer).strip()

            # ── Store user reply visibly in history ───────────────────────────
            label = "✅ Satisfied" if is_satisfaction else "💬 Clarification"
            st.session_state["message_history"].append(
                {"role": "user", "content": f"**{label}:** {human_answer}"}
            )
            st.session_state["hitl_pending"]  = False
            st.session_state["hitl_question"] = None
            st.session_state["hitl_type"]     = ""

            # Mark HITL card done in activity panel
            for step in reversed(st.session_state["activity_log"]):
                if step["node"] == card_node:
                    step["status"] = "done"
                    step["detail"] = f"User replied: {human_answer[:60]}"
                    break
            render_activity_panel(activity_ph)

            if is_satisfaction:
                # ── SATISFACTION PATH ─────────────────────────────────────────
                # satisfaction_hitl_node paused via interrupt().
                # The ONLY correct way to resume an interrupt() is to pass
                # Command(resume=value) as the input to stream/invoke.
                # update_state alone does NOT resume the interrupt — it just
                # patches state, causing stream(None) to re-run the node from
                # the top and hit interrupt() again (the loop bug).
                response_parts: list[str] = []

                if human_answer.lower() in ("yes", "y"):
                    # User satisfied — drain graph to END, no assistant bubble needed
                    for _ in chatbot.stream(
                        Command(resume=human_answer),
                        config=_cfg(tid),
                        stream_mode="updates",
                    ):
                        pass  # just drain to END
                else:
                    # User not satisfied — re-explain, stream the new response
                    with st.chat_message("assistant"):
                        badge_ph = st.empty()

                        def _sat_stream():
                            for event in chatbot.stream(
                                Command(resume=human_answer),
                                config=_cfg(tid),
                                stream_mode="updates",
                            ):
                                text = _handle_chunk(event, badge_ph)
                                render_activity_panel(activity_ph)
                                if text:
                                    response_parts.append(text)
                                    yield text

                        st.write_stream(_sat_stream())

                    ai_msg = "".join(response_parts)
                    if ai_msg:
                        st.session_state["message_history"].append(
                            {"role": "assistant", "content": ai_msg}
                        )

            else:
                # ── CLARIFICATION PATH ────────────────────────────────────────
                # hitl_node paused via interrupt().
                # Use Command(resume=value) to correctly resume the interrupt,
                # then stream processes _route_after_hitl → parser/solver.
                response_parts: list[str] = []
                with st.chat_message("assistant"):
                    badge_ph = st.empty()

                    def _clar_stream():
                        for event in chatbot.stream(
                            Command(resume=human_answer),
                            config=_cfg(tid),
                            stream_mode="updates",
                        ):
                            text = _handle_chunk(event, badge_ph)
                            render_activity_panel(activity_ph)
                            if text:
                                response_parts.append(text)
                                yield text

                    st.write_stream(_clar_stream())

                ai_msg = "".join(response_parts)
                if ai_msg:
                    st.session_state["message_history"].append(
                        {"role": "assistant", "content": ai_msg}
                    )

            _mark_all_done()
            render_activity_panel(activity_ph)

            # ── Remove resolved HITL question from conversation_log ───────────
            # After the user answers, the HITL question stored in conversation_log
            # must be cleared. Otherwise _load_history re-appends it and the
            # banner appears again in the chat history on rerun.
            try:
                snap_c = chatbot.get_state(config=_cfg(tid))
                existing_log = (snap_c.values or {}).get("conversation_log") or []
                if existing_log:
                    last = existing_log[-1]
                    if str(last).startswith((_HITL_PREFIX, _HITL_SAT_PREFIX)):
                        chatbot.update_state(
                            _cfg(tid),
                            {"conversation_log": existing_log[:-1]},
                        )
            except Exception:
                pass

            # Reload from LangGraph state — single source of truth
            st.session_state["message_history"] = _load_history(tid)

            # ── Show Manim video if it was rendered before this HITL ──────────
            # manim_node runs before satisfaction_hitl, so its video_path is
            # already in state by the time we resume here.
            try:
                snap2 = chatbot.get_state(config=_cfg(tid))
                vpath2 = (snap2.values or {}).get("manim_video_path")
                if vpath2 and os.path.exists(str(vpath2)):
                    st.divider()
                    st.subheader("🎬 Visual Explanation")
                    st.video(str(vpath2))
            except Exception:
                pass

            # Check if graph paused again for another HITL
            interrupted, q2, htype2 = _check_hitl(tid)
            if interrupted:
                st.session_state["hitl_pending"]  = True
                st.session_state["hitl_question"] = q2
                st.session_state["hitl_type"]     = htype2

            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  NORMAL INPUT WIDGETS  (hidden while HITL is pending)
    # ══════════════════════════════════════════════════════════════════════════

    else:
        input_mode = st.radio(
            "Input type", ["Text", "Image", "Audio"],
            horizontal=True, label_visibility="collapsed",
        )

        text_input: Optional[str] = None
        image_file = None
        audio_file = None

        if input_mode == "Text":
            text_input = st.chat_input("Enter your math problem…")
        elif input_mode == "Image":
            image_file = st.file_uploader(
                "Upload image", type=["png", "jpg", "jpeg"], key="img_up"
            )
        elif input_mode == "Audio":
            audio_file = st.file_uploader(
                "Upload audio", type=["wav", "mp3", "m4a"], key="aud_up"
            )

        # ── Submission ─────────────────────────────────────────────────────────
        if text_input or image_file or audio_file:
            tid = st.session_state["thread_id"]

            # Reset activity log for this new turn
            st.session_state["activity_log"] = []
            st.session_state["current_node"] = None
            render_activity_panel(activity_ph)

            # Build agent payload with all TypedDict fields pre-populated
            payload: dict = {
                "input_mode":          None,
                "raw_text":            None,
                "image_path":          None,
                "audio_path":          None,
                "thread_id":           tid,
                "parsed_data":         None,
                "solution_plan":       None,
                "solver_output":       None,
                "verifier_output":     None,
                "explainer_output":    None,
                "solve_iterations":    0,
                "hitl_required":       False,
                "hitl_reason":         None,
                "human_feedback":      None,
                "retrieved_context":   None,
                "manim_video_path":    None,
                "ocr_text":            None,
                "ocr_confidence":      None,
                "transcript":          None,
                "asr_confidence":      None,
                "user_corrected_text": None,
                "messages":            [],
                "agent_payload_log":   [],
                "conversation_log":    [],
                "final_response":      None,
            }

            # ── Populate payload + show user message ──────────────────────────
            if text_input:
                payload["input_mode"] = "text"
                payload["raw_text"]   = text_input
                st.session_state["message_history"].append(
                    {"role": "user", "content": text_input}
                )
                with st.chat_message("user"):
                    st.markdown(text_input)

            if image_file:
                fp = f"uploads/{image_file.name}"
                with open(fp, "wb") as fh:
                    fh.write(image_file.getbuffer())
                payload["input_mode"] = "image"
                payload["image_path"] = fp
                st.session_state["message_history"].append(
                    {"role": "user", "content": "[Image uploaded]"}
                )
                with st.chat_message("user"):
                    st.image(image_file, caption="Uploaded question")

            if audio_file:
                fp = f"uploads/{audio_file.name}"
                with open(fp, "wb") as fh:
                    fh.write(audio_file.getbuffer())
                payload["input_mode"] = "audio"
                payload["audio_path"] = fp
                st.session_state["message_history"].append(
                    {"role": "user", "content": "[Audio uploaded]"}
                )
                with st.chat_message("user"):
                    st.audio(audio_file)

            # ── Stream agent response ──────────────────────────────────────────
            with st.chat_message("assistant"):
                badge_ph       = st.empty()
                response_parts: list[str] = []

                def _stream():
                    for event in chatbot.stream(
                        payload,
                        config=_cfg(tid),
                        stream_mode="updates",
                    ):
                        text = _handle_chunk(event, badge_ph)
                        render_activity_panel(activity_ph)
                        if text:
                            response_parts.append(text)
                            yield text

                st.write_stream(_stream())

            # Mark all remaining active cards as done
            _mark_all_done()
            render_activity_panel(activity_ph)

            # Reload message_history from LangGraph state — this is the single
            # source of truth and avoids duplicates that occur when both the
            # live stream appends AND _load_history add the same content.
            st.session_state["message_history"] = _load_history(tid)

            # ── Show Manim video if produced ──────────────────────────────────
            # Must happen BEFORE the HITL rerun check below, because if the graph
            # paused at satisfaction_hitl the st.rerun() would short-circuit this block.
            try:
                final_state = chatbot.get_state(config=_cfg(tid))
                vpath = (final_state.values or {}).get("manim_video_path")
                if vpath and os.path.exists(str(vpath)):
                    st.divider()
                    st.subheader("🎬 Visual Explanation")
                    st.video(str(vpath))
                    _add_step(
                        "manim_node", status="done",
                        detail=f"Video: {os.path.basename(str(vpath))}"
                    )
                    render_activity_panel(activity_ph)
            except Exception:
                pass

            # ── Check if graph paused for HITL ────────────────────────────────
            interrupted, question, htype = _check_hitl(tid)
            if interrupted:
                st.session_state["hitl_pending"]  = True
                st.session_state["hitl_question"] = question
                st.session_state["hitl_type"]     = htype
                st.rerun()