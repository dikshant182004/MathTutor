from __future__ import annotations
import os
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from authlib.integrations.requests_client import OAuth2Session

from backend.agents.graph import chatbot, checkpointer
from backend.agents.state import make_initial_state
from backend.agents.nodes.memory.memory_manager import prune_stale_episodic
from backend.agents.utils.db_utils import (
    get_user_profile, get_or_create_user, create_thread,
    get_thread_history, load_thread_state,          
)
from backend.agents.nodes.tools.tools import clear_store, get_store_info, ingest_pdf

from frontend import st, TOOL_META, ANSWER_NODES, HITL_PREFIX, HITL_SAT_PREFIX, Path, Optional
from frontend.templates.activity_panel import (
    render_activity_panel, add_step,
    mark_previous_done, mark_all_done,
    build_hitl_banner, build_history_hitl_banner,
)
from frontend.templates.profile import build_profile_card

st.set_page_config(page_title="Math Tutor", page_icon="🧮",
                   layout="wide", initial_sidebar_state="expanded")

_css = (Path(__file__).parent / "templates" / "styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

os.makedirs("uploads", exist_ok=True)
os.makedirs("manim_outputs", exist_ok=True)

# ── OAuth constants ───────────────────────────────────────────────────────────
_AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO  = "https://www.googleapis.com/oauth2/v3/userinfo"
_SCOPE     = "openid email profile"


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _handle_oauth_callback() -> None:
    """Exchange Google ?code= for user info, create/update Redis user record."""
    code = dict(st.query_params).get("code")
    if not code:
        return

    cid  = st.secrets.get("GOOGLE_CLIENT_ID", "")
    csec = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
    ruri = st.secrets.get("OAUTH_REDIRECT_URI", "")
    if not all([cid, csec, ruri]):
        st.error("OAuth secrets not configured."); st.stop()

    oauth = OAuth2Session(client_id=cid, client_secret=csec, scope=_SCOPE, redirect_uri=ruri)
    oauth.token = oauth.fetch_token(_TOKEN_URL, code=code, client_secret=csec)
    user  = oauth.get(_USERINFO).json()
    email = (user or {}).get("email", "")
    name  = (user or {}).get("name", email.split("@")[0])
    if not email:
        st.error("No email from Google."); st.stop()

    sid = get_or_create_user(email=email, display_name=name)
    st.session_state["google_user"] = {"email": email, "name": name}
    st.session_state["student_id"]  = sid
    try:
        st.query_params.clear()
    except Exception:
        pass


def _require_login() -> None:
    _handle_oauth_callback()

    if st.session_state.get("student_id"):
        return

    st.title("🧮 JEE Math Tutor")
    st.info("Please sign in with your Google account to continue.")
    cid  = st.secrets.get("GOOGLE_CLIENT_ID", "")
    csec = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
    ruri = st.secrets.get("OAUTH_REDIRECT_URI", "")
    if not all([cid, csec, ruri]):
        st.warning("Configure OAuth secrets (GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, OAUTH_REDIRECT_URI)."); st.stop()
    oauth = OAuth2Session(client_id=cid, client_secret=csec, scope=_SCOPE, redirect_uri=ruri)
    uri, state = oauth.create_authorization_url(_AUTH_URL, access_type="offline", prompt="consent")
    st.session_state["oauth_state"] = state
    st.link_button("Continue with Google", uri, use_container_width=True, type="primary")
    st.stop()


def _logout() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    try:
        st.query_params.clear()
    except Exception:
        pass
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cfg(tid: str) -> dict:
    return {"configurable": {"thread_id": tid}, "metadata": {"thread_id": tid}}


def _init_session() -> None:
    _require_login()
    sid = st.session_state["student_id"]

    # Bug 4 fix — "user_profile" included in defaults so it is always initialised
    defaults: dict = {
        "thread_id":       None,
        "threads_meta":    [],
        "message_history": [],
        "pdf_ingested":    False,
        "hitl_pending":    False,
        "hitl_question":   None,
        "hitl_type":       "",
        "hitl_payload":    None,
        "activity_log":    [],
        "current_node":    None,
        "user_profile":    None,   # Bug 4 fix
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

    st.session_state["message_history"] = _load_history(st.session_state["thread_id"])


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
    )
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

def _load_history(tid: str) -> list[dict]:
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        vals = snap.values or {}
        msgs, conv, final = (
            vals.get("messages", []),
            vals.get("conversation_log") or [],
            vals.get("final_response", ""),
        )
    except Exception:
        return []

    history, seen = [], set()
    _skip = ("Solve this problem:", "[Human feedback]", "[Human clarification]",
             "[Feedback]:", "[Clarification]:")

    for m in msgs:
        raw = m.content
        if isinstance(raw, list):
            raw = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in raw)
        c = str(raw or "").strip()
        if not c:
            continue

        if isinstance(m, HumanMessage):
            if any(c.startswith(p) for p in _skip):
                continue
            # use full content hash instead of prefix slice
            k = f"u:{hash(c)}"
            if k not in seen:
                seen.add(k)
                history.append({"role": "user", "content": c})

        elif isinstance(m, AIMessage) and not getattr(m, "tool_calls", None):
            if c.startswith("## 📘"):
                continue
            if c.startswith((HITL_PREFIX, HITL_SAT_PREFIX)):
                continue
            if len(c) < 30:
                continue
            k = f"a:{hash(c)}"
            if k not in seen:
                seen.add(k)
                history.append({"role": "assistant", "content": c})

    if final and final.strip():
        k = f"a:{hash(final)}"
        if k not in seen:
            seen.add(k)
            history.append({"role": "assistant", "content": final.strip()})

    for entry in conv:
        e = str(entry).strip()
        if not e:
            continue
        k = f"a:{hash(e)}"
        if k not in seen:
            seen.add(k)
            history.append({"role": "assistant", "content": e})

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  HITL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _check_hitl(tid: str) -> tuple[bool, Optional[dict]]:
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        if "hitl_node" not in list(snap.next or []):
            return False, None
        for task in (snap.tasks or []):
            for iv in getattr(task, "interrupts", []):
                val = getattr(iv, "value", {}) or {}
                if isinstance(val, dict):
                    return True, val
        return True, {"hitl_type": "clarification", "prompt": "Please clarify."}
    except Exception:
        return False, None


# ══════════════════════════════════════════════════════════════════════════════
#  THREAD RESUME HELPER
#  Builds a human-readable activity summary from the last checkpoint snapshot.
#  Shown in the activity panel when a past thread is loaded.
# ══════════════════════════════════════════════════════════════════════════════

def _build_activity_from_snapshot(snapshot: dict) -> list[dict]:
    """
    Reconstructs a minimal activity log from a restored AgentState snapshot
    so the activity panel shows meaningful context when resuming a past thread.
    """
    import time as _time

    log = []

    def _entry(node: str, detail: str, status: str = "done") -> dict:
        from frontend import AGENT_META
        meta = AGENT_META.get(node, {"icon": "⚙️", "label": node})
        return {
            "node":    node,
            "icon":    meta["icon"],
            "label":   meta["label"],
            "status":  status,
            "detail":  detail[:120],
            "payload": {},
            "ts":      _time.strftime("%H:%M:%S"),
        }

    if snapshot.get("guardrail_passed") is True:
        log.append(_entry("guardrail_agent", "Passed ✓"))
    if snapshot.get("parsed_data"):
        pd = snapshot["parsed_data"]
        log.append(_entry("parser_agent", f"Topic: {pd.get('topic', '?')}"))
    if snapshot.get("solution_plan"):
        sp = snapshot["solution_plan"]
        log.append(_entry("intent_router", f"{sp.get('topic','?')} | {sp.get('difficulty','?')}"))
    if snapshot.get("solver_output"):
        so = snapshot["solver_output"]
        log.append(_entry("solver_agent", f"Attempt {snapshot.get('solve_iterations', 1)}"))
    if snapshot.get("verifier_output"):
        vo = snapshot["verifier_output"]
        log.append(_entry("verifier_agent", f"Status: {vo.get('status', '?')}"))
    if snapshot.get("safety_passed") is True:
        log.append(_entry("safety_agent", "Passed ✓"))
    if snapshot.get("explainer_output"):
        log.append(_entry("explainer_agent", "Explanation delivered"))
    if snapshot.get("manim_video_path"):
        log.append(_entry("manim_node", "Video rendered"))
    if snapshot.get("ltm_stored"):
        log.append(_entry("store_ltm", "Memory saved ✓"))

    return log


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

        if last and isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            names = [tc["name"] for tc in last.tool_calls]
            for n in names:
                m = TOOL_META.get(n, {"icon": "🔧", "label": n})
                add_step("tool_node", status="tool", detail=f'{m["icon"]} {m["label"]}')
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
            # restore AgentState snapshot for richer activity panel context
            snapshot = load_thread_state(tid, checkpointer) or {}

            # Rebuild activity log from snapshot so past work is visible in the panel
            restored_log = _build_activity_from_snapshot(snapshot)

            st.session_state.update(
                thread_id       = tid,
                message_history = _load_history(tid),
                pdf_ingested    = get_store_info(tid) is not None,
                # 4b — show restored activity instead of blank panel
                activity_log    = restored_log,
                current_node    = None,
            )

            # Restore HITL state if thread is mid-interrupt
            interrupted, payload = _check_hitl(tid)
            st.session_state.update(
                hitl_pending  = interrupted,
                hitl_payload  = payload,
                hitl_question = ((payload or {}).get("prompt") or
                                 (payload or {}).get("message")) if payload else None,
                hitl_type     = (payload or {}).get("hitl_type", "") if payload else "",
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
        gu = st.session_state.get("google_user") or {}
        up = st.session_state.get("user_profile") or {}
        if gu:
            with st.popover("👤 " + gu.get("name", "Profile"), use_container_width=True):
                st.markdown(build_profile_card(
                    name            = gu.get("name", ""),
                    email           = gu.get("email", ""),
                    problems_solved = int(up.get("total_problems_solved", 0)),
                    last_login      = float(up.get("last_login", 0)) or None,
                    member_since    = float(up.get("created_at", 0)) or None,
                ), unsafe_allow_html=True)
                if st.button("🚪 Logout", key="logout_top", use_container_width=True):
                    _logout()

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

        pfx      = HITL_SAT_PREFIX if is_sat else HITL_PREFIX
        stored_q = pfx + question
        hist     = st.session_state["message_history"]
        if not hist or hist[-1].get("content") != stored_q:
            hist.append({"role": "assistant", "content": stored_q})
            try:
                snap = chatbot.get_state(config=_cfg(tid))
                exl  = (snap.values or {}).get("conversation_log") or []
                if not exl or exl[-1] != stored_q:
                    chatbot.update_state(_cfg(tid), {"conversation_log": exl + [stored_q]})
            except Exception:
                pass

        st.markdown(build_hitl_banner(hitl_type, question), unsafe_allow_html=True)

        human_answer: Optional[dict] = None

        if is_sat:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, next question", use_container_width=True, type="primary"):
                    human_answer = {"satisfied": True, "follow_up": ""}
            with c2:
                if st.button("🔄 No, re-explain", use_container_width=True):
                    st.session_state["_show_reexplain_form"] = True

            #  using session_state flag 
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
            with st.form("hitl_clarify", clear_on_submit=True):
                cl = st.text_area("Clarification", placeholder="Type here…",
                                  height=100, label_visibility="collapsed")
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    human_answer = {"corrected_text": cl.strip()}

        if human_answer is not None:
            lbl = ("✅ Satisfied" if is_sat else "🖼️ Input" if is_bad
                   else "🧾 Verification" if is_ver else "💬 Clarification")
            st.session_state["message_history"].append(
                {"role": "user", "content": f"**{lbl}:** {str(human_answer)[:300]}"}
            )
            st.session_state.update(
                hitl_pending=False, hitl_question=None,
                hitl_type="", hitl_payload=None,
            )
            for s in reversed(st.session_state["activity_log"]):
                if s["node"] == "hitl_node":
                    s["status"] = "done"
                    s["detail"] = f"Answered: {str(human_answer)[:60]}"
                    break
            render_activity_panel(activity_ph)

            parts: list[str] = []
            with st.chat_message("assistant"):
                bph = st.empty()

                def _resume():
                    for ev in chatbot.stream(
                        Command(resume=human_answer),
                        config=_cfg(tid), stream_mode="updates",
                    ):
                        t = _handle_chunk(ev, bph)
                        render_activity_panel(activity_ph)
                        if t:
                            parts.append(t)
                            yield t

                st.write_stream(_resume())

            if parts:
                st.session_state["message_history"].append(
                    {"role": "assistant", "content": "".join(parts)}
                )

            mark_all_done()
            render_activity_panel(activity_ph)

            # Clean HITL banner from conversation_log
            try:
                sc = chatbot.get_state(config=_cfg(tid))
                el = (sc.values or {}).get("conversation_log") or []
                # Cap conversation_log at 20 entries (Warning fix)
                if el and str(el[-1]).startswith((HITL_PREFIX, HITL_SAT_PREFIX)):
                    el = el[:-1]
                chatbot.update_state(_cfg(tid), {"conversation_log": el[-20:]})
            except Exception:
                pass

            st.session_state["message_history"] = _load_history(tid)

            # Manim video
            try:
                snap2 = chatbot.get_state(config=_cfg(tid))
                vp = (snap2.values or {}).get("manim_video_path")
                if vp and os.path.exists(str(vp)):
                    st.divider()
                    st.subheader("🎬 Visual Explanation")
                    st.video(str(vp))
            except Exception:
                pass

            # Refresh profile
            try:
                st.session_state["user_profile"] = get_user_profile(
                    st.session_state["student_id"])
            except Exception:
                pass

            # Check chained HITL
            interrupted, p2 = _check_hitl(tid)
            if interrupted and p2:
                st.session_state.update(
                    hitl_pending=True, hitl_payload=p2,
                    hitl_question=p2.get("prompt") or p2.get("message"),
                    hitl_type=p2.get("hitl_type", ""),
                )
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

            # guard: sid must never be "anonymous" at this point
            # _require_login() enforces login before we reach here, but be explicit
            if not sid or sid == "anonymous":
                st.error("Please log in before solving problems.")
                st.stop()

            st.session_state["activity_log"] = []
            st.session_state["current_node"] = None
            render_activity_panel(activity_ph)

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
                bph = st.empty()
                rparts: list[str] = []

                def _stream():
                    for ev in chatbot.stream(sp, config=_cfg(tid), stream_mode="updates"):
                        t = _handle_chunk(ev, bph)
                        render_activity_panel(activity_ph)
                        if t:
                            rparts.append(t)
                            yield t

                st.write_stream(_stream())

            mark_all_done()
            render_activity_panel(activity_ph)
            st.session_state["message_history"] = _load_history(tid)

            # Manim video
            try:
                fs = chatbot.get_state(config=_cfg(tid))
                vp = (fs.values or {}).get("manim_video_path")
                if vp and os.path.exists(str(vp)):
                    st.divider()
                    st.subheader("🎬 Visual Explanation")
                    st.video(str(vp))
                    add_step("manim_node", "done", f"Video: {os.path.basename(str(vp))}")
                    render_activity_panel(activity_ph)
            except Exception:
                pass

            # HITL after normal stream
            interrupted, p = _check_hitl(tid)
            if interrupted and p:
                st.session_state.update(
                    hitl_pending=True, hitl_payload=p,
                    hitl_question=p.get("prompt") or p.get("message"),
                    hitl_type=p.get("hitl_type", ""),
                )
                st.rerun()