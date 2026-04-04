from frontend.templates import html, time
from frontend import st, AGENT_META

def add_step(
    node:    str,
    status:  str  = "done",
    detail:  str  = "",
    payload: dict = None,
) -> None:
    """
    Append a new step card to the activity log.

    Called by:
        _process_node_update() — when a new node fires during streaming
        HITL block              — when the graph pauses for human input
    """
    meta = AGENT_META.get(node, {"icon": "⚙️", "label": node})
    st.session_state["activity_log"].append({
        "node":    node,
        "icon":    meta["icon"],
        "label":   meta["label"],
        "status":  status,
        "detail":  str(detail)[:120],
        "payload": payload or {},
        "ts":      time.strftime("%H:%M:%S"),
    })


def mark_previous_done(current_node: str) -> None:
    """
    Flip any card that is still "active" but is NOT current_node to "done".
    Called each time a new node fires so only one card pulses at a time.
    """
    for step in st.session_state["activity_log"]:
        if step["status"] == "active" and step["node"] != current_node:
            step["status"] = "done"


def mark_all_done() -> None:
    """
    Flip ALL remaining "active" cards to "done".
    Called at the very end of streaming (both normal and HITL resume paths).
    """
    for step in st.session_state["activity_log"]:
        if step["status"] == "active":
            step["status"] = "done"


# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def build_payload_html(payload: dict) -> str:
    """
    Builds a collapsible <details> block for a step card.

    The payload dict looks like:
        {
            "summary": "PASSED | topic=calculus",
            "fields":  {"Topic": "calculus", "Difficulty": "hard", ...}
        }

    The summary becomes the <summary> click target.
    The fields become a two-column table inside the details.

    Returns "" if payload is empty (nothing to show).
    """
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
                f"<tr>"
                f"<td>{html.escape(str(k))}</td>"
                f"<td>{html.escape(str(v))}</td>"
                f"</tr>"
            )
    table_html = f'<table class="payload-table">{rows}</table>' if rows else ""

    return (
        f'<div class="step-payload">'
        f'  <div class="payload-summary">▸ {summary_text}</div>'
        f'  {table_html}'
        f'</div>'
    )


def build_step_card(step: dict) -> str:
    """
    Builds the full HTML for one activity card.

    CSS classes on the outer div control appearance:
        step-card active   → blue, pulsing (node currently running)
        step-card done     → green, dimmed  (node finished)
        step-card tool     → orange         (tool call fired)
        step-card hitl     → red            (waiting for human)
    """
    css          = f"step-card {step['status']}"
    icon         = html.escape(step["icon"])
    label        = html.escape(step["label"])
    ts           = html.escape(step["ts"])
    detail_html  = (
        f'<p class="step-detail">{html.escape(step["detail"])}</p>'
        if step.get("detail") else ""
    )
    payload_html = build_payload_html(step.get("payload") or {})

    return (
        f'<div class="{css}">'
        f'  <div class="step-header">'
        f'    <span class="step-label">{icon}&nbsp;{label}</span>'
        f'    <span class="step-ts">{ts}</span>'
        f'  </div>'
        f'  {detail_html}'
        f'  {payload_html}'
        f'</div>'
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL RENDERER
# ══════════════════════════════════════════════════════════════════════════════

def render_activity_panel(placeholder) -> None:
    """
    Rewrites the entire activity panel inside a st.empty() placeholder.

    Called:
        1. On first render (before any streaming starts) — shows empty state
        2. Inside the stream loop after every chunk — live updates
        3. After HITL form submission — marks HITL card done
        4. After stream ends — marks all remaining cards done

    The placeholder pattern (st.empty()) means Streamlit replaces the entire
    block rather than appending — so we never see duplicate cards.

    Args:
        placeholder: the st.empty() object created once in new_app.py
    """
    log: list[dict] = st.session_state.get("activity_log", [])

    with placeholder.container():
        st.markdown("### 🤖 Agent Activity")

        if not log:
            st.caption("Activity will appear here once you send a message.")
            return

        cards_html = "\n".join(build_step_card(step) for step in log)
        st.markdown(cards_html, unsafe_allow_html=True)


def build_history_hitl_banner(hitl_type: str, text: str) -> str:
    """
    Builds the banner rendered inside message_history replay.
    Used by new_app.py when iterating past messages that carry a HITL prefix.

    hitl_type: "satisfaction" → green, everything else → red
    """
    t = html.escape(text)
    if hitl_type == "satisfaction":
        return (
            f'<div class="hitl-banner satisfaction">'
            f'  <div class="hitl-title">✅ Solution Complete</div>'
            f'  <div class="hitl-body">{t}</div>'
            f'</div>'
        )
    return (
        f'<div class="hitl-banner">'
        f'  <div class="hitl-title">🙋 Clarification Was Needed</div>'
        f'  <div class="hitl-body">{t}</div>'
        f'</div>'
    )