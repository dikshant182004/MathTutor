from __future__ import annotations
import os
import hashlib
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from langgraph.errors import GraphInterrupt

from backend.agents import logger
from backend.agents.graph import chatbot, checkpointer
from backend.agents.state import make_initial_state
from backend.agents.nodes.memory.memory_manager import prune_stale_episodic
from backend.agents.utils.db_utils import (
    get_user_profile, get_or_create_user, create_thread,
    get_thread_history
)
from backend.agents.nodes.tools.tools import clear_store, get_store_info, ingest_pdf

from frontend.templates.login import render_login_page
from frontend import st, TOOL_META, ANSWER_NODES, HITL_PREFIX, HITL_SAT_PREFIX, Path, Optional
from frontend.templates.activity_panel import (
    render_activity_panel, add_step,
    mark_previous_done, mark_all_done,
    build_history_hitl_banner,
)
from frontend.templates.profile import build_profile_card

st.set_page_config(page_title="Math Tutor", page_icon="🧮",
                   layout="wide", initial_sidebar_state="expanded")

# ── Load global styles ────────────────────────────────────────────────────────
try:
    _css = (Path(__file__).parent / "templates" / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Suppress Streamlit sidebar nav tooltip ("keyboard_double_arrow_right") ───
st.markdown("""
<style>
[data-testid="stSidebarNav"],
[data-testid="stSidebarNav"] *,
[data-testid="stSidebarNavItems"],
[data-testid="stSidebarNavSeparator"] { display: none !important; }
[data-testid="stSidebar"] [title="keyboard_double_arrow_right"],
[data-testid="stSidebar"] [aria-label="keyboard_double_arrow_right"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

os.makedirs("uploads", exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  AUTH GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.user.is_logged_in:
    render_login_page()
    st.stop()

if "google_user" not in st.session_state:
    st.session_state["google_user"] = {
        "name":  st.user.name  or "",
        "email": st.user.email or "",
    }

_email = st.user.email or ""
_name  = st.user.name  or _email.split("@")[0]

if "student_id" not in st.session_state:
    st.session_state["student_id"] = get_or_create_user(
        email        = _email,
        display_name = _name,
    )


def _logout() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.logout()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}, "metadata": {"thread_id": tid}}


def _init_session() -> None:
    sid = st.session_state["student_id"]

    defaults: dict = {
        "thread_id":             None,
        "threads_meta":          [],
        "message_history":       [],
        "pdf_ingested":          False,
        "hitl_pending":          False,
        "hitl_question":         None,
        "hitl_type":             "",
        "hitl_payload":          None,
        "activity_log":          [],
        "current_node":          None,
        "user_profile":          None,
        "_show_reexplain_form":  False,
        "hitl_resuming":         False,
        "history_loaded":        False,
        "current_question":      None,
        "current_question_mode": None,   # "text" | "image" | "audio"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state["threads_meta"]:
        try:
            st.session_state["threads_meta"] = get_thread_history(sid)
        except Exception:
            pass

    if not st.session_state["user_profile"]:
        try:
            st.session_state["user_profile"] = get_user_profile(sid)
        except Exception:
            pass

    if not st.session_state["thread_id"]:
        metas = st.session_state["threads_meta"]
        st.session_state["thread_id"] = (
            metas[0]["thread_id"] if metas else create_thread(sid)
        )

    if not st.session_state.get("history_loaded"):
        st.session_state["message_history"] = _load_history(st.session_state["thread_id"])
        st.session_state["history_loaded"] = True


def _reset_chat() -> None:
    old = st.session_state.get("thread_id")
    if old:
        clear_store(old)
    sid = st.session_state["student_id"]
    tid = create_thread(sid)
    st.session_state.update(
        thread_id=tid, message_history=[], pdf_ingested=False,
        hitl_pending=False, hitl_question=None, hitl_type="",
        hitl_payload=None, activity_log=[], current_node=None,
        _show_reexplain_form=False, hitl_resuming=False,
        history_loaded=True,
        current_question=None,
        current_question_mode=None,
    )
    st.session_state.pop("clarification_prefill", None)
    try:
        st.session_state["threads_meta"] = get_thread_history(sid)
    except Exception:
        pass


def _fallback_threads() -> list[str]:
    ids: set[str] = set()
    try:
        for cp in checkpointer.list(None):
            t = cp.config["configurable"].get("thread_id")
            if t:
                ids.add(t)
    except Exception:
        pass
    return list(ids)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _build_history_from_vals(vals: dict) -> list[dict]:
    msgs  = vals.get("messages", [])
    conv  = vals.get("conversation_log") or []
    final = vals.get("final_response", "")

    seen = set()
    _skip = ("Solve this problem:", "[Human feedback]", "[Human clarification]",
             "[Feedback]:", "[Clarification]:")

    user_msgs = []
    for m in msgs:
        if not isinstance(m, HumanMessage):
            continue
        raw = m.content
        if isinstance(raw, list):
            raw = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        c = str(raw or "").strip()
        if not c or any(c.startswith(p) for p in _skip):
            continue
        k = f"u:{hashlib.md5(c.encode()).hexdigest()[:12]}"
        if k not in seen:
            seen.add(k)
            user_msgs.append(c)

    solutions = []
    for entry in conv:
        e = str(entry).strip()
        if not e or e.startswith((HITL_PREFIX, HITL_SAT_PREFIX)) or not e.startswith("## "):
            continue
        k = f"a:{hashlib.md5(e.encode()).hexdigest()[:12]}"
        if k not in seen:
            seen.add(k)
            solutions.append(e)

    if final and final.strip() and final.startswith("## "):
        k = f"a:{hashlib.md5(final.encode()).hexdigest()[:12]}"
        if k not in seen:
            seen.add(k)
            solutions.append(final.strip())

    history = []
    for i, user_q in enumerate(user_msgs):
        history.append({"role": "user", "content": user_q})
        if i < len(solutions):
            history.append({"role": "assistant", "content": solutions[i]})
    for sol in solutions[len(user_msgs):]:
        history.append({"role": "assistant", "content": sol})
    return history


def _extract_hitl_from_snap(snap) -> tuple[bool, Optional[dict]]:
    next_nodes = list(snap.next or [])
    if "hitl_node" not in next_nodes:
        return False, None
    for task in (snap.tasks or []):
        for iv in getattr(task, "interrupts", []):
            val = getattr(iv, "value", {}) or {}
            if isinstance(val, dict):
                return True, val

    vals           = snap.values or {}
    stored_payload = vals.get("hitl_interrupt")
    if isinstance(stored_payload, dict) and stored_payload.get("hitl_type"):
        return True, stored_payload
    hitl_type   = vals.get("hitl_type") or "clarification"
    hitl_reason = vals.get("hitl_reason") or ""
    return True, {
        "hitl_type": hitl_type,
        "prompt":    hitl_reason or "Please provide the required input.",
        "message":   hitl_reason or "Please provide the required input.",
    }


def _load_history(tid: str) -> list[dict]:
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        return _build_history_from_vals(snap.values or {})
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════════
#  HITL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _check_hitl(tid: str) -> tuple[bool, Optional[dict]]:
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        return _extract_hitl_from_snap(snap)
    except Exception as e:
        logger.warning(f"[app] _check_hitl failed: {e}")
        return False, None


# ══════════════════════════════════════════════════════════════════════════════
#  STREAM HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def _process_node_update(node_name: str, patch: dict, badge_ph) -> Optional[str]:
    log = st.session_state["activity_log"]
    if not log or log[-1]["node"] != node_name:
        mark_previous_done(node_name)
        add_step(node_name, status="active")
    st.session_state["current_node"] = node_name

    for entry in reversed(patch.get("agent_payload_log") or []):
        if entry.get("node") == node_name:
            pld = {"summary": entry.get("summary", ""), "fields": entry.get("fields", {})}
            for s in reversed(log):
                if s["node"] == node_name:
                    s["payload"] = pld
                    s["detail"]  = pld["summary"] or s["detail"]
                    s["status"]  = "done"
                    break
            break

    if patch.get("ltm_stored"):
        for s in reversed(log):
            if s["node"] == "store_ltm":
                s["detail"] = "Memory saved ✓"
                break

    # ── DirectResponse tool signals (web search bypasses ToolNode) ──────────
    for tc in patch.get("direct_response_tool_calls") or []:   ## just a hack to show our direct response tool node in the ui 
        n     = tc.get("name", "web_search_tool")
        m     = TOOL_META.get(n, {"icon": "🌐", "label": "Web Search"})
        query = (tc.get("args") or {}).get("query", "")
        add_step(  
            n, status="tool",
            detail  = query[:80] or m["label"],
            payload = {
                "summary": f'{m["icon"]} {m["label"]}',
                "fields":  {"Query": query[:120] or "—", "Tool": n},
            },
        )
        badge_ph.caption(f'🔧 {m["label"]}…')

    if patch.get("final_response"):
        badge_ph.empty()
        for s in reversed(log):
            if s["node"] == node_name:
                s["status"] = "done"
                break
        return str(patch["final_response"])

    msgs = patch.get("messages") or []
    if msgs:
        last = msgs[-1] if isinstance(msgs, list) else None
        if (last and isinstance(last, AIMessage) and last.content
                and not getattr(last, "tool_calls", None)
                and node_name in ANSWER_NODES):
            badge_ph.empty()
            return str(last.content)

        # ── BUG 1 FIX: tool cards with correct icon/label/payload ─────────────
        if last and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            names = [tc["name"] for tc in last.tool_calls]
            for tc in last.tool_calls:
                n     = tc["name"]
                m     = TOOL_META.get(n, {"icon": "🔧", "label": n})
                query = (tc.get("args") or {}).get("query", "")
                add_step(
                    n,                    # real tool name → AGENT_META resolves icon+label
                    status  = "tool",
                    detail  = query[:80] if query else m["label"],
                    payload = {
                        "summary": f'{m["icon"]} {m["label"]}',
                        "fields":  {
                            "Query": query[:120] if query else "—",
                            "Tool":  n,
                        },
                    },
                )
            badge_ph.caption("🔧 " + " · ".join(names) + "…")

    return None


def _handle_chunk(chunk, badge_ph) -> Optional[str]:
    if isinstance(chunk, dict):
        result = None
        for node_name, patch in chunk.items():
            if isinstance(patch, dict):
                t = _process_node_update(node_name, patch, badge_ph)
                if t:
                    result = t
        return result
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  QUESTION BANNER (BUG 2 FIX)
#  Renders into a st.empty() placeholder passed in — can be wiped cleanly.
#  The OLD version used st.markdown() directly which burns into the DOM and
#  cannot be removed without a full rerun.
# ══════════════════════════════════════════════════════════════════════════════

def _render_question_banner(placeholder) -> None:
    """
    Renders the current question into a st.empty() placeholder.
    Call placeholder.empty() to wipe it instantly without a rerun.
    """
    import html as _html

    q    = st.session_state.get("current_question")
    mode = st.session_state.get("current_question_mode", "text")

    if not q:
        placeholder.empty()
        return

    if mode == "image":
        icon, display_text = "📸", "Image uploaded"
    elif mode == "audio":
        icon, display_text = "🎤", "Audio uploaded"
    else:
        icon         = "❓"
        display_text = str(q)

    if len(display_text) > 200:
        display_text = display_text[:197] + "…"

    with placeholder.container():
        st.markdown(
            f"""<div style="
                background: linear-gradient(90deg, #0d1f33 0%, #0a1628 100%);
                border: 1px solid #1e3a5a;
                border-left: 4px solid #3b82f6;
                border-radius: 8px;
                padding: 10px 16px;
                margin-bottom: 12px;
                font-size: 0.88rem;
                color: #c8d8f0;
                display: flex;
                align-items: flex-start;
                gap: 10px;
            ">
                <span style="font-size:1.1rem;flex-shrink:0">{icon}</span>
                <div>
                    <div style="font-size:0.70rem;color:#4a6080;text-transform:uppercase;
                                letter-spacing:0.08em;margin-bottom:3px">Current Question</div>
                    <div style="line-height:1.5">{_html.escape(display_text)}</div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
_init_session()

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🧮 Math Tutor")

    gu = st.session_state.get("google_user") or {}
    up = st.session_state.get("user_profile") or {}
    if gu:
        with st.expander("👤 " + gu.get("name", "Profile"), expanded=False):
            st.markdown(build_profile_card(
                name            = gu.get("name", ""),
                email           = gu.get("email", ""),
                problems_solved = int(up.get("total_problems_solved", 0)),
                last_login      = float(up.get("last_login", 0)) or None,
                member_since    = float(up.get("created_at", 0)) or None,
            ), unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True):
                _logout()

    if st.button("➕ New Chat", use_container_width=True):
        _reset_chat()
        st.rerun()

    # ── PDF ───────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📄 Study Material")
    st.caption("Upload a PDF — solver calls `rag_tool` automatically.")
    pdf = st.file_uploader("Upload PDF", type=["pdf"],
                           key="pdf_uploader", label_visibility="collapsed")
    if pdf:
        tid  = st.session_state["thread_id"]
        info = get_store_info(tid)
        if info and pdf.name in info.get("filenames", []):
            st.info(f"📚 Already indexed: **{pdf.name}** ({info['chunks']} chunks)")
        else:
            with st.spinner(f"Embedding **{pdf.name}**…"):
                try:
                    s = ingest_pdf(file_bytes=pdf.read(), thread_id=tid, filename=pdf.name)
                    st.session_state["pdf_ingested"] = True
                    st.success(f"✅ **{s['filename']}** — {s['pages']} pages · {s['chunks']} chunks")
                except Exception as e:
                    st.error(f"❌ {e}")
    if st.session_state["pdf_ingested"]:
        info = get_store_info(st.session_state["thread_id"])
        if info:
            st.info(f"📚 **{info['filename']}** — {info['chunks']} chunks")

    # ── Thread history ────────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 Conversations")
    metas = st.session_state.get("threads_meta") or []
    if not metas:
        metas = [
            {"thread_id": t, "problem_summary": "", "topic": "", "outcome": ""}
            for t in sorted(_fallback_threads(), reverse=True)[:20]
        ]

    scroll_container = st.container(height=420, border=True)
    with scroll_container:
        if not metas:
            st.info("No conversations yet. Start solving problems!")
        else:
            for meta in metas:
                tid = meta.get("thread_id", "")
                if not tid:
                    continue
                active  = tid == st.session_state["thread_id"]
                summary = (meta.get("problem_summary") or "").strip()
                topic   = (meta.get("topic") or "").strip()
                outcome = (meta.get("outcome") or "").strip()
                title   = summary[:40] or f"{tid[:10]}…"
                label   = f"{'▶ ' if active else ''}{title}"
                sub     = " · ".join(filter(None, [topic, outcome]))

                if st.button(label, key=f"t_{tid}", use_container_width=True, help=sub or None):
                    try:
                        snap = chatbot.get_state(config=_cfg(tid))
                        snap_vals  = snap.values or {}
                        next_nodes = list(snap.next or [])
                        history    = _build_history_from_vals(snap_vals)
                        interrupted, payload = _extract_hitl_from_snap(snap)
                    except Exception:
                        snap_vals  = {}
                        next_nodes = []
                        history    = []
                        interrupted, payload = False, None

                    is_genuinely_pending = interrupted and "hitl_node" in next_nodes

                    st.session_state.update(
                        thread_id       = tid,
                        message_history = history,
                        history_loaded  = True,
                        pdf_ingested    = get_store_info(tid) is not None,
                        activity_log    = [],
                        current_node    = None,
                        hitl_pending    = is_genuinely_pending,
                        hitl_payload    = payload if is_genuinely_pending else None,
                        hitl_question   = ((payload or {}).get("prompt") or
                                        (payload or {}).get("message")) if is_genuinely_pending else None,
                        hitl_type       = (payload or {}).get("hitl_type", "") if is_genuinely_pending else "",
                        hitl_resuming   = False,
                        _show_reexplain_form = False,
                        # BUG 2 FIX: clear question banner when switching threads
                        current_question      = None,
                        current_question_mode = None,
                    )
                    st.rerun()

    # ── Admin ─────────────────────────────────────────────────────────────────
    st.divider()
    with st.expander("⚙️ Admin", expanded=False):
        st.caption("Delete episodic memories with decay_score < 0.05 and age > 30 days.")
        if st.button("🗑️ Prune Stale Memories", use_container_width=True):
            sid    = st.session_state.get("student_id")
            pruned = prune_stale_episodic(student_id=sid)
            st.success(f"Pruned {pruned} stale episodic record(s).")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_activity, col_chat = st.columns([3, 7], gap="medium")

with col_activity:
    activity_ph = st.empty()
    render_activity_panel(activity_ph)

with col_chat:
    h1, h2 = st.columns([7, 3], vertical_alignment="center")
    with h1:
        st.title("🧮 Math Tutor Agent")
    with h2:
        if st.button("🧠 Memory", use_container_width=True, help="View your memory graph"):
            st.switch_page("pages/memory_viz.py")

    # ── Message history replay ────────────────────────────────────────────────
    for msg in st.session_state["message_history"]:
        c    = msg.get("content", "")
        role = msg["role"]
        if c.startswith(HITL_PREFIX):
            st.markdown(build_history_hitl_banner("clarification", c[len(HITL_PREFIX):]),
                        unsafe_allow_html=True)
        elif c.startswith(HITL_SAT_PREFIX):
            st.markdown(build_history_hitl_banner("satisfaction", c[len(HITL_SAT_PREFIX):]),
                        unsafe_allow_html=True)
        else:
            with st.chat_message(role):
                st.markdown(c)

    # ── BUG 2 FIX: question banner placeholder ────────────────────────────────
    # Created ONCE here at a fixed position in the render tree.
    # _render_question_banner() writes into it; placeholder.empty() wipes it.
    # The old approach called st.markdown() directly inside the HITL block which
    # burned the banner into the DOM with no way to remove it without a rerun.
    question_banner_ph = st.empty()
    _render_question_banner(question_banner_ph)

    # ══════════════════════════════════════════════════════════════════════════
    #  HITL BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("hitl_pending"):
        payload   = st.session_state.get("hitl_payload") or {}
        question  = (st.session_state.get("hitl_question")
                     or payload.get("prompt") or payload.get("message")
                     or "Please provide input.")
        hitl_type = st.session_state.get("hitl_type") or payload.get("hitl_type", "clarification")
        tid       = st.session_state["thread_id"]

        is_sat = hitl_type == "satisfaction"
        is_bad = hitl_type == "bad_input"
        is_ver = hitl_type == "verification"

        log = st.session_state["activity_log"]
        if not log or log[-1]["node"] != "hitl_node":
            add_step("hitl_node", status="hitl", detail=question[:120])
            render_activity_panel(activity_ph)

        # BUG 2 FIX: banner is now rendered above at fixed position, NOT here.
        # For satisfaction HITL wipe the banner immediately so student sees it's done.
        if is_sat:
            question_banner_ph.empty()

        expander_title = {
            "verification":  "📋 Verifier Notes",
            "clarification": "📝 Clarify the Problem",
            "bad_input":     "📤 Please Provide Better Input",
            "satisfaction":  "✅ Feedback",
        }.get(hitl_type, "📋 Details")

        with st.expander(expander_title, expanded=True):
            st.markdown(question if question else "No additional details.")

        human_answer: Optional[dict] = None

        if is_sat:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, next question", use_container_width=True, type="primary"):
                    human_answer = {"satisfied": True, "follow_up": ""}
                    st.session_state["_show_reexplain_form"] = False
            with c2:
                if st.button("🔄 No, re-explain", use_container_width=True):
                    st.session_state["_show_reexplain_form"] = True

            if st.session_state.get("_show_reexplain_form"):
                with st.form("reexplain", clear_on_submit=True):
                    txt = st.text_area("What was unclear?", height=80)
                    if st.form_submit_button("Submit 🔄", use_container_width=True):
                        human_answer = {"satisfied": False, "follow_up": txt or ""}
                        st.session_state["_show_reexplain_form"] = False

        elif is_bad:
            with st.form("bad_input", clear_on_submit=True):
                nt = st.text_area("Type the problem", height=90)
                ni = st.file_uploader("Re-upload image", type=["png", "jpg", "jpeg"])
                na = st.file_uploader("Re-upload audio", type=["wav", "mp3", "m4a"])
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    u: dict = {}
                    if ni:
                        fp = f"uploads/{ni.name}"
                        with open(fp, "wb") as f:
                            f.write(ni.getbuffer())
                        u["new_image_path"] = fp
                    if na:
                        fp = f"uploads/{na.name}"
                        with open(fp, "wb") as f:
                            f.write(na.getbuffer())
                        u["new_audio_path"] = fp
                    if nt and nt.strip():
                        u["raw_text"] = nt.strip()
                    human_answer = u or None

        elif is_ver:
            with st.form("verification", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1:
                    ok = st.checkbox("Mark as correct", value=False)
                with c2:
                    st.caption("Add hint if incorrect.")
                hint = st.text_area("Hint (optional)", height=80)
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    human_answer = {"is_correct": bool(ok), "fix_hint": hint.strip()}

        else:  # clarification
            prefill = payload.get("problem_text", "")
            if "clarification_prefill" not in st.session_state:
                st.session_state["clarification_prefill"] = prefill
            with st.form("hitl_clarify", clear_on_submit=True):
                cl = st.text_area(
                    "Clarification",
                    value=st.session_state.get("clarification_prefill", ""),
                    height=100,
                    label_visibility="collapsed"
                )
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    human_answer = {"corrected_text": cl.strip()}
                    st.session_state.pop("clarification_prefill", None)

        # ── Resume flow ───────────────────────────────────────────────────────
        if human_answer is not None:
            if st.session_state.get("hitl_resuming"):
                st.stop()

            st.session_state["hitl_resuming"] = True

            lbl = ("✅ Satisfied" if is_sat else "🖼️ Input" if is_bad
                   else "🧾 Verification" if is_ver else "💬 Clarification")
            st.session_state["message_history"].append(
                {"role": "user", "content": f"**{lbl}:** {str(human_answer)[:300]}"}
            )
            st.session_state.update(
                hitl_pending=False, hitl_question=None,
                hitl_type="", hitl_payload=None,
            )
            st.session_state.pop("clarification_prefill", None)
            for s in reversed(st.session_state["activity_log"]):
                if s["node"] == "hitl_node":
                    s["status"] = "done"
                    s["detail"] = f"Answered: {str(human_answer)[:60]}"
                    break
            render_activity_panel(activity_ph)

            # BUG 2 FIX: clear question + banner on satisfaction
            if is_sat and human_answer.get("satisfied"):
                st.session_state["current_question"]      = None
                st.session_state["current_question_mode"] = None
                question_banner_ph.empty()

            # BUG 2 FIX: guard the assistant bubble — only paint if stream yields content.
            # The old code opened st.chat_message("assistant") unconditionally which
            # permanently painted a blank red robot icon even when the stream was empty.
            parts: list[str] = []
            bph = st.empty()   # badge placeholder lives outside the bubble

            def _resume():
                try:
                    snap = chatbot.get_state(config=_cfg(tid))
                    if "hitl_node" not in list(snap.next or []):
                        logger.warning("[app] Resume called but graph not at hitl_node — skipping")
                        return
                    for ev in chatbot.stream(
                        Command(resume=human_answer),
                        config=_cfg(tid), stream_mode="updates",
                    ):
                        t = _handle_chunk(ev, bph)
                        render_activity_panel(activity_ph)
                        if t:
                            parts.append(t)
                            yield t
                except GraphInterrupt:
                    pass
                except Exception as e:
                    error_msg = f"⚠️ Something went wrong: {str(e)[:200]}"
                    parts.append(error_msg)
                    yield error_msg

            # Collect stream first — only paint bubble when there is actual content
            collected = list(_resume())
            bph.empty()
            if collected:
                with st.chat_message("assistant"):
                    st.markdown("".join(collected))
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": "".join(collected)}
                )

            mark_all_done()
            render_activity_panel(activity_ph)

            try:
                post_snap  = chatbot.get_state(config=_cfg(tid))
                post_vals  = post_snap.values or {}
                post_next  = list(post_snap.next or [])
                st.session_state["message_history"] = _build_history_from_vals(post_vals)
                st.session_state["history_loaded"]  = True

                interrupted, p2 = _extract_hitl_from_snap(post_snap)
                is_pending = (
                    interrupted
                    and "hitl_node" in post_next
                    and not (is_sat and human_answer.get("satisfied"))
                )
                if is_pending and p2:
                    st.session_state.update(
                        hitl_pending=True, hitl_payload=p2,
                        hitl_question=p2.get("prompt") or p2.get("message"),
                        hitl_type=p2.get("hitl_type", ""),
                    )
                else:
                    # Graph finished with no further HITL — ensure banner is cleared
                    st.session_state["current_question"]      = None
                    st.session_state["current_question_mode"] = None
                    question_banner_ph.empty()

            except Exception:
                st.session_state["message_history"] = _load_history(tid)
                st.session_state["history_loaded"]  = True

            try:
                st.session_state["user_profile"] = get_user_profile(
                    st.session_state["student_id"])
            except Exception:
                pass

            st.session_state["hitl_resuming"] = False
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  NORMAL INPUT BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    else:
        mode = st.radio("Input", ["Text", "Image", "Audio"],
                        horizontal=True, label_visibility="collapsed")
        text_input = image_file = audio_file = None
        if mode == "Text":
            text_input = st.chat_input("Enter your math problem…")
        elif mode == "Image":
            image_file = st.file_uploader("Image", type=["png", "jpg", "jpeg"], key="img_up")
        elif mode == "Audio":
            audio_file = st.file_uploader("Audio", type=["wav", "mp3", "m4a"], key="aud_up")

        if text_input or image_file or audio_file:
            tid = st.session_state["thread_id"]
            sid = st.session_state["student_id"]

            if not sid or sid == "anonymous":
                st.error("Please log in before solving problems.")
                st.stop()

            st.session_state["activity_log"] = []
            st.session_state["current_node"] = None
            render_activity_panel(activity_ph)

            # BUG 2 FIX: wipe old banner and reset BEFORE setting new question
            # so there is never a frame where the old question text shows briefly
            question_banner_ph.empty()
            st.session_state["current_question"]      = None
            st.session_state["current_question_mode"] = None

            if text_input:
                st.session_state["current_question"]      = text_input
                st.session_state["current_question_mode"] = "text"
            elif image_file:
                st.session_state["current_question"]      = image_file.name
                st.session_state["current_question_mode"] = "image"
            elif audio_file:
                st.session_state["current_question"]      = audio_file.name
                st.session_state["current_question_mode"] = "audio"

            # Render the new banner immediately into the placeholder
            _render_question_banner(question_banner_ph)

            sp = make_initial_state(student_id=sid, thread_id=tid)

            if text_input:
                sp["raw_text"] = text_input
                st.session_state["message_history"].append({"role": "user", "content": text_input})
                with st.chat_message("user"):
                    st.markdown(text_input)

            if image_file:
                fp = f"uploads/{image_file.name}"
                with open(fp, "wb") as f:
                    f.write(image_file.getbuffer())
                sp["image_path"] = fp
                st.session_state["message_history"].append(
                    {"role": "user", "content": "[Image uploaded]"})
                with st.chat_message("user"):
                    st.image(image_file, caption="Uploaded question")

            if audio_file:
                fp = f"uploads/{audio_file.name}"
                with open(fp, "wb") as f:
                    f.write(audio_file.getbuffer())
                sp["audio_path"] = fp
                st.session_state["message_history"].append(
                    {"role": "user", "content": "[Audio uploaded]"})
                with st.chat_message("user"):
                    st.audio(audio_file)

            with st.chat_message("assistant"):
                bph    = st.empty()
                rparts: list[str] = []

                def _stream():
                    try:
                        for ev in chatbot.stream(sp, config=_cfg(tid), stream_mode="updates"):
                            t = _handle_chunk(ev, bph)
                            render_activity_panel(activity_ph)
                            if t:
                                rparts.append(t)
                                yield t
                    except GraphInterrupt:
                        pass

                st.write_stream(_stream())

            mark_all_done()
            render_activity_panel(activity_ph)

            try:
                fs      = chatbot.get_state(config=_cfg(tid))
                fs_vals = fs.values or {}
                fs_next = list(fs.next or [])
                st.session_state["message_history"] = _build_history_from_vals(fs_vals)
                st.session_state["history_loaded"]  = True

                interrupted, p = _extract_hitl_from_snap(fs)
                is_pending = interrupted and "hitl_node" in fs_next
                if is_pending and p:
                    st.session_state.update(
                        hitl_pending=True, hitl_payload=p,
                        hitl_question=p.get("prompt") or p.get("message"),
                        hitl_type=p.get("hitl_type", ""),
                    )
                    st.rerun()
                else:
                    # BUG 2 FIX: graph finished cleanly with no HITL pending
                    # (e.g. research/explain intents that skip satisfaction HITL)
                    # Clear the banner now so it doesn't persist into next question.
                    st.session_state["current_question"]      = None
                    st.session_state["current_question_mode"] = None
                    question_banner_ph.empty()

            except Exception:
                st.session_state["message_history"] = _load_history(tid)
                st.session_state["history_loaded"]  = True
                interrupted, p = _check_hitl(tid)
                if interrupted and p:
                    st.session_state.update(
                        hitl_pending=True, hitl_payload=p,
                        hitl_question=p.get("prompt") or p.get("message"),
                        hitl_type=p.get("hitl_type", ""),
                    )
                    st.rerun()
                else:
                    # Exception path — still clear the banner
                    st.session_state["current_question"]      = None
                    st.session_state["current_question_mode"] = None
                    question_banner_ph.empty()