from __future__ import annotations
import os
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from authlib.integrations.requests_client import OAuth2Session

from backend.agents.graph import chatbot, checkpointer
from backend.agents.state import make_initial_state
from backend.agents.nodes.memory.memory_manager import prune_stale_episodic
from backend.agents.utils.db_utils import get_user_profile, get_or_create_user, create_thread, get_thread_history
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
_AUTH_URL   = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL  = "https://oauth2.googleapis.com/token"
_USERINFO   = "https://www.googleapis.com/oauth2/v3/userinfo"
_SCOPE      = "openid email profile"


# ══════════════════════════════════════════════════════════════════════════════
#  AUTH
# ══════════════════════════════════════════════════════════════════════════════

def _handle_oauth_callback() -> None:
    """
    Exchange Google ?code= for user info.
    ── MEMORY ── get_or_create_user() writes user:<student_id> hash to Redis.
    """
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

    # ── MEMORY ──
    sid = get_or_create_user(email=email, display_name=name)
    st.session_state["google_user"] = {"email": email, "name": name}
    st.session_state["student_id"]  = sid
    try: st.query_params.clear()
    except Exception: pass


def _require_login() -> None:
    _handle_oauth_callback()

    if st.session_state.get("student_id"):
        return
    st.title("🧮 JEE Math Tutor")
    cid = st.secrets.get("GOOGLE_CLIENT_ID", "")
    csec = st.secrets.get("GOOGLE_CLIENT_SECRET", "")
    ruri = st.secrets.get("OAUTH_REDIRECT_URI", "")
    if not all([cid, csec, ruri]):
        st.info("Configure OAuth secrets."); st.stop()
    oauth = OAuth2Session(client_id=cid, client_secret=csec, scope=_SCOPE, redirect_uri=ruri)
    uri, state = oauth.create_authorization_url(_AUTH_URL, access_type="offline", prompt="consent")
    st.session_state["oauth_state"] = state
    st.link_button("Continue with Google", uri, use_container_width=True, type="primary")
    st.stop()


def _logout() -> None:
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    try: st.query_params.clear()
    except Exception: pass
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _cfg(tid: str) -> dict:
    """
    LangGraph config dict. Passed to every chatbot.stream() / get_state() call.
    LangGraph uses thread_id to find the right RedisSaver checkpoint.
    """
    return {"configurable": {"thread_id": tid}, "metadata": {"thread_id": tid}}


def _init_session() -> None:
    """
    Runs once per browser session.
    ── MEMORY ──
      get_thread_history()  → loads sidebar thread list from Redis sorted set
      get_user_profile()    → loads problems_solved / last_login for profile card
      create_thread()       → creates first Redis thread record for new users
    """
    _require_login()
    sid = st.session_state["student_id"]

    defaults: dict = {
        "thread_id":       None,
        "threads_meta":    [],   # list[dict] from get_thread_history
        "message_history": [],
        "pdf_ingested":    False,
        "hitl_pending":    False,
        "hitl_question":   None,
        "hitl_type":       "",       # "clarification" | "satisfaction"
        "hitl_payload":    None,     # full interrupt payload dict
        "activity_log":    [],      # [{node, icon, label, status, detail, ts}, ...]
        "current_node":    None,    # name of the node currently streaming
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── MEMORY ── sidebar thread list
    if not st.session_state["threads_meta"]:
        try: st.session_state["threads_meta"] = get_thread_history(sid)
        except Exception: pass

    # ── MEMORY ── user profile for card (problems solved, login times)
    if not st.session_state["user_profile"]:
        try: st.session_state["user_profile"] = get_user_profile(sid)
        except Exception: pass

    # Ensure active thread
    if not st.session_state["thread_id"]:
        metas = st.session_state["threads_meta"]
        st.session_state["thread_id"] = (
            metas[0]["thread_id"] if metas else create_thread(sid)  # ── MEMORY ──
        )

    st.session_state["message_history"] = _load_history(st.session_state["thread_id"])


def _reset_chat() -> None:
    """
    New Chat button handler.
    ── MEMORY ── create_thread() writes new thread record to Redis.
    """
    old = st.session_state.get("thread_id")
    if old: clear_store(old)
    sid = st.session_state["student_id"]
    tid = create_thread(sid)  # ── MEMORY ──
    st.session_state.update(
        thread_id=tid, message_history=[], pdf_ingested=False,
        hitl_pending=False, hitl_question=None, hitl_type="",
        hitl_payload=None, activity_log=[], current_node=None,
    )
    try: st.session_state["threads_meta"] = get_thread_history(sid)
    except Exception: pass


def _fallback_threads() -> list[str]:
    """Scan checkpointer directly when Redis thread registry is empty."""
    ids: set[str] = set()
    try:
        for cp in checkpointer.list(None):
            t = cp.config["configurable"].get("thread_id")
            if t: ids.add(t)
    except Exception: pass
    return list(ids)


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION HISTORY
# ══════════════════════════════════════════════════════════════════════════════

def _load_history(tid: str) -> list[dict]:
    """
    Rebuilds visible chat from the LangGraph RedisSaver checkpoint.
    Single source of truth — called after every stream + on thread switch.

    Sources merged:
      state["messages"]        → HumanMessages + non-internal AIMessages
      state["final_response"]  → explainer formatted markdown
      state["conversation_log"]→ HITL banner strings (prefixed)

    ── MEMORY ── reads the full AgentState from RedisSaver checkpoint.
    ── HITL ──   conversation_log contains banner strings that render
                 as coloured divs, not plain chat bubbles.
    """
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
            raw = " ".join(p.get("text","") if isinstance(p,dict) else str(p) for p in raw)
        c = str(raw or "").strip()
        if not c: continue

        if isinstance(m, HumanMessage):
            if any(c.startswith(p) for p in _skip): continue
            k = f"u:{c[:100]}"
            if k not in seen:
                seen.add(k); history.append({"role":"user","content":c})

        elif isinstance(m, AIMessage) and not getattr(m,"tool_calls",None):
            if c.startswith("## 📘"): continue
            if c.startswith((HITL_PREFIX, HITL_SAT_PREFIX)): continue
            if len(c) < 30: continue
            k = f"a:{c[:100]}"
            if k not in seen:
                seen.add(k); history.append({"role":"assistant","content":c})

    if final and final.strip():
        k = f"a:{final[:100]}"
        if k not in seen:
            seen.add(k); history.append({"role":"assistant","content":final.strip()})

    for entry in conv:
        e = str(entry).strip()
        if not e: continue
        k = f"a:{e[:100]}"
        if k not in seen:
            seen.add(k); history.append({"role":"assistant","content":e})

    return history


# ══════════════════════════════════════════════════════════════════════════════
#  HITL DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _check_hitl(tid: str) -> tuple[bool, Optional[dict]]:
    """
    Checks if the graph paused at hitl_node after streaming.

    ── HITL ── When hitl_node calls interrupt(payload):
      1. LangGraph checkpoints AgentState to Redis
      2. Sets snap.next = ["hitl_node"]
      3. Stores payload in snap.tasks[n].interrupts
      4. chatbot.stream() stops yielding

    This function detects that state by reading snap.next.
    Returns (True, payload) or (False, None).
    """
    try:
        snap = chatbot.get_state(config=_cfg(tid))
        if "hitl_node" not in list(snap.next or []):
            return False, None
        for task in (snap.tasks or []):
            for iv in getattr(task, "interrupts", []):
                val = getattr(iv, "value", {}) or {}
                if isinstance(val, dict): return True, val
        return True, {"hitl_type": "clarification", "prompt": "Please clarify."}
    except Exception:
        return False, None


# ══════════════════════════════════════════════════════════════════════════════
#  STREAM HANDLER
# ══════════════════════════════════════════════════════════════════════════════

def _process_node_update(node_name: str, patch: dict, badge_ph) -> Optional[str]:
    """
    Processes one node's state patch from stream_mode="updates".
    Each event = {node_name: {fields that node wrote to state}}.

    Actions:
      1. Add/update activity card for the node
      2. Attach agent_payload_log data to the card
      3. Return text to yield to chat bubble (or None)

    Text priority:
      1. patch["final_response"]    — explainer's full markdown answer
      2. last AIMessage in messages — only for ANSWER_NODES
      3. tool_calls present         — show tool badges, no text to chat

    ── MEMORY ── patch["ltm_context"] set by retrieve_ltm (silent, not displayed)
                 patch["ltm_stored"]=True set by store_ltm (shown in card detail)
    """
    log = st.session_state["activity_log"]
    if not log or log[-1]["node"] != node_name:
        mark_previous_done(node_name)
        add_step(node_name, status="active")
    st.session_state["current_node"] = node_name

    # Attach payload
    for entry in reversed(patch.get("agent_payload_log") or []):
        if entry.get("node") == node_name:
            pld = {"summary": entry.get("summary",""), "fields": entry.get("fields",{})}
            for s in reversed(log):
                if s["node"] == node_name:
                    s["payload"] = pld
                    s["detail"]  = pld["summary"] or s["detail"]
                    s["status"]  = "done"
                    break
            break

    # LTM store confirmation
    if patch.get("ltm_stored"):
        for s in reversed(log):
            if s["node"] == "store_ltm":
                s["detail"] = "Memory saved ✓"; break

    # Priority 1: final_response
    if patch.get("final_response"):
        badge_ph.empty()
        for s in reversed(log):
            if s["node"] == node_name: s["status"] = "done"; break
        return str(patch["final_response"])

    # Priority 2: plain AIMessage answer
    msgs = patch.get("messages") or []
    if msgs:
        last = msgs[-1] if isinstance(msgs, list) else None
        if (last and isinstance(last, AIMessage) and last.content
                and not getattr(last,"tool_calls",None)
                and node_name in ANSWER_NODES):
            badge_ph.empty()
            return str(last.content)

        # Priority 3: tool call badges
        if last and isinstance(last, AIMessage) and getattr(last,"tool_calls",None):
            names = [tc["name"] for tc in last.tool_calls]
            for n in names:
                m = TOOL_META.get(n, {"icon":"🔧","label":n})
                add_step("tool_node", status="tool", detail=f'{m["icon"]} {m["label"]}')
            badge_ph.caption("🔧 " + " · ".join(names) + "…")

    return None


def _handle_chunk(chunk, badge_ph) -> Optional[str]:
    if isinstance(chunk, dict):
        result = None
        for node_name, patch in chunk.items():
            if isinstance(patch, dict):
                t = _process_node_update(node_name, patch, badge_ph)
                if t: result = t
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

    # ── Profile card ──────────────────────────────────────────────────────────
    # ── MEMORY ── user_profile fetched from Redis in _init_session
    gu = st.session_state.get("google_user") or {}
    up = st.session_state.get("user_profile") or {}
    if gu:
        with st.expander("👤 " + gu.get("name","Profile"), expanded=False):
            st.markdown(build_profile_card(
                name            = gu.get("name",""),
                email           = gu.get("email",""),
                problems_solved = int(up.get("total_problems_solved", 0)),
                last_login      = float(up.get("last_login", 0)) or None,
                member_since    = float(up.get("created_at", 0)) or None,
            ), unsafe_allow_html=True)
            if st.button("🚪 Logout", use_container_width=True): _logout()

    if st.button("➕ New Chat", use_container_width=True):
        _reset_chat(); st.rerun()

    # ── PDF ───────────────────────────────────────────────────────────────────
    st.divider(); st.subheader("📄 Study Material")
    st.caption("Upload a PDF — solver calls `rag_tool` automatically.")
    pdf = st.file_uploader("Upload PDF", type=["pdf"],
                           key="pdf_uploader", label_visibility="collapsed")
    if pdf:
        tid  = st.session_state["thread_id"]
        info = get_store_info(tid)
        if info and pdf.name in info.get("filenames",[]):
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
        if info: st.info(f"📚 **{info['filename']}** — {info['chunks']} chunks")

    # ── Thread history ────────────────────────────────────────────────────────
    # ── MEMORY ── threads loaded from Redis. Clicking one reloads checkpoint.
    st.divider(); st.subheader("💬 Conversations")
    metas = st.session_state.get("threads_meta") or []
    if not metas:
        metas = [{"thread_id":t,"problem_summary":"","topic":"","outcome":""}
                 for t in sorted(_fallback_threads(), reverse=True)[:20]]

    for meta in metas:
        tid = meta.get("thread_id","")
        if not tid: continue
        active  = tid == st.session_state["thread_id"]
        summary = (meta.get("problem_summary") or "").strip()
        topic   = (meta.get("topic") or "").strip()
        outcome = (meta.get("outcome") or "").strip()
        title   = summary[:40] or f"{tid[:10]}…"
        label   = f"{'▶ ' if active else ''}{title}"
        sub     = " · ".join(filter(None, [topic, outcome]))

        if st.button(label, key=f"t_{tid}", use_container_width=True, help=sub or None):
            # ── MEMORY ── _load_history reads RedisSaver checkpoint for this thread
            st.session_state.update(
                thread_id       = tid,
                message_history = _load_history(tid),   # ← full history from checkpoint
                pdf_ingested    = get_store_info(tid) is not None,
                activity_log    = [],
                current_node    = None,
            )
            # ── HITL ── thread may be mid-interrupt; restore HITL state
            interrupted, payload = _check_hitl(tid)
            st.session_state.update(
                hitl_pending  = interrupted,
                hitl_payload  = payload,
                hitl_question = ((payload or {}).get("prompt") or
                                 (payload or {}).get("message")) if payload else None,
                hitl_type     = (payload or {}).get("hitl_type","") if payload else "",
            )
            st.rerun()

    # ── Admin: Manual Pruning ─────────────────────────────────────────────────
    # ── MEMORY ── prune_stale_episodic deletes decay_score < 0.05 AND age > 30d
    st.divider()
    with st.expander("⚙️ Admin", expanded=False):
        st.caption("Delete episodic memories with decay_score < 0.05 and age > 30 days.")
        if st.button("🗑️ Prune Stale Memories", use_container_width=True):
            sid    = st.session_state.get("student_id")
            pruned = prune_stale_episodic(student_id=sid)  # ── MEMORY ──
            st.success(f"Pruned {pruned} stale episodic record(s).")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_activity, col_chat = st.columns([3, 7], gap="medium")

with col_activity:
    activity_ph = st.empty()          # rewritten in-place — no duplicate cards
    render_activity_panel(activity_ph)

with col_chat:
    # ── Header ────────────────────────────────────────────────────────────────
    h1, h2 = st.columns([7,3], vertical_alignment="center")
    with h1: st.title("🧮 Math Tutor Agent")
    with h2:
        gu = st.session_state.get("google_user") or {}
        up = st.session_state.get("user_profile") or {}
        if gu:
            with st.popover("👤 "+gu.get("name","Profile"), use_container_width=True):
                st.markdown(build_profile_card(
                    name            = gu.get("name",""),
                    email           = gu.get("email",""),
                    problems_solved = int(up.get("total_problems_solved",0)),
                    last_login      = float(up.get("last_login",0)) or None,
                    member_since    = float(up.get("created_at",0)) or None,
                ), unsafe_allow_html=True)
                if st.button("🚪 Logout", key="logout_top", use_container_width=True):
                    _logout()

    # ── Message history replay ────────────────────────────────────────────────
    for msg in st.session_state["message_history"]:
        c    = msg.get("content","")
        role = msg["role"]
        if c.startswith(HITL_PREFIX):
            st.markdown(build_history_hitl_banner("clarification", c[len(HITL_PREFIX):]),
                        unsafe_allow_html=True)
        elif c.startswith(HITL_SAT_PREFIX):
            st.markdown(build_history_hitl_banner("satisfaction", c[len(HITL_SAT_PREFIX):]),
                        unsafe_allow_html=True)
        else:
            with st.chat_message(role): st.markdown(c)

    # ══════════════════════════════════════════════════════════════════════════
    #  HITL BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.get("hitl_pending"):
        payload   = st.session_state.get("hitl_payload") or {}
        question  = (st.session_state.get("hitl_question")
                     or payload.get("prompt") or payload.get("message")
                     or "Please provide input.")
        hitl_type = st.session_state.get("hitl_type") or payload.get("hitl_type","clarification")
        tid       = st.session_state["thread_id"]

        is_sat  = hitl_type == "satisfaction"
        is_bad  = hitl_type == "bad_input"
        is_ver  = hitl_type == "verification"

        log = st.session_state["activity_log"]
        if not log or log[-1]["node"] != "hitl_node":
            add_step("hitl_node", status="hitl", detail=question[:120])
            render_activity_panel(activity_ph)

        # ── HITL ── Persist HITL question into conversation_log so it survives reload
        pfx = HITL_SAT_PREFIX if is_sat else HITL_PREFIX
        stored_q = pfx + question
        hist = st.session_state["message_history"]
        if not hist or hist[-1].get("content") != stored_q:
            hist.append({"role":"assistant","content":stored_q})
            try:
                snap = chatbot.get_state(config=_cfg(tid))
                exl  = (snap.values or {}).get("conversation_log") or []
                if not exl or exl[-1] != stored_q:
                    chatbot.update_state(_cfg(tid), {"conversation_log": exl+[stored_q]})
            except Exception: pass

        st.markdown(build_hitl_banner(hitl_type, question), unsafe_allow_html=True)

        human_answer: Optional[dict] = None

        if is_sat:
            c1, c2 = st.columns(2)
            with c1:
                if st.button("✅ Yes, next question", use_container_width=True, type="primary"):
                    human_answer = {"satisfied": True, "follow_up": ""}
            with c2:
                if st.button("🔄 No, re-explain", use_container_width=True):
                    with st.form("reexplain", clear_on_submit=True):
                        txt = st.text_area("What was unclear?", height=80)
                        if st.form_submit_button("Submit 🔄", use_container_width=True):
                            human_answer = {"satisfied": False, "follow_up": txt or ""}

        elif is_bad:
            with st.form("bad_input", clear_on_submit=True):
                nt = st.text_area("Type the problem", height=90)
                ni = st.file_uploader("Re-upload image", type=["png","jpg","jpeg"])
                na = st.file_uploader("Re-upload audio", type=["wav","mp3","m4a"])
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    u: dict = {}
                    if ni:
                        fp=f"uploads/{ni.name}"; open(fp,"wb").write(ni.getbuffer()); u["new_image_path"]=fp
                    if na:
                        fp=f"uploads/{na.name}"; open(fp,"wb").write(na.getbuffer()); u["new_audio_path"]=fp
                    if nt and nt.strip(): u["raw_text"]=nt.strip()
                    human_answer = u or None

        elif is_ver:
            with st.form("verification", clear_on_submit=True):
                c1, c2 = st.columns(2)
                with c1: ok = st.checkbox("Mark as correct", value=False)
                with c2: st.caption("Add hint if incorrect.")
                hint = st.text_area("Hint (optional)", height=80)
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    human_answer = {"is_correct": bool(ok), "fix_hint": hint.strip()}

        else:  # clarification
            with st.form("hitl_clarify", clear_on_submit=True):
                cl = st.text_area("Clarification", placeholder="Type here…",
                                  height=100, label_visibility="collapsed")
                if st.form_submit_button("Submit ✅", use_container_width=True):
                    human_answer = {"corrected_text": cl.strip()}

        # ── HITL ── Command(resume=...) resumes the paused graph ──────────────
        # When hitl_node called interrupt(payload), execution froze at:
        #     human_response = interrupt(payload)
        #
        # Command(resume=human_answer) tells LangGraph to:
        #   1. Reload AgentState from Redis checkpoint
        #   2. Re-enter hitl_node at the SAME line
        #   3. Make human_answer the return value of interrupt()
        #   4. Continue graph execution normally
        if human_answer is not None:
            lbl = ("✅ Satisfied" if is_sat else "🖼️ Input" if is_bad
                   else "🧾 Verification" if is_ver else "💬 Clarification")
            st.session_state["message_history"].append(
                {"role":"user","content":f"**{lbl}:** {str(human_answer)[:300]}"}
            )
            st.session_state.update(hitl_pending=False, hitl_question=None,
                                     hitl_type="", hitl_payload=None)
            for s in reversed(st.session_state["activity_log"]):
                if s["node"] == "hitl_node":
                    s["status"]="done"; s["detail"]=f"Answered: {str(human_answer)[:60]}"; break
            render_activity_panel(activity_ph)

            parts: list[str] = []
            with st.chat_message("assistant"):
                bph = st.empty()
                def _resume():
                    for ev in chatbot.stream(
                        Command(resume=human_answer),   # ← HITL resume
                        config=_cfg(tid), stream_mode="updates",
                    ):
                        t = _handle_chunk(ev, bph)
                        render_activity_panel(activity_ph)
                        if t: parts.append(t); yield t
                st.write_stream(_resume())

            if parts: st.session_state["message_history"].append(
                {"role":"assistant","content":"".join(parts)})

            mark_all_done(); render_activity_panel(activity_ph)

            # Clean HITL banner from conversation_log
            try:
                sc = chatbot.get_state(config=_cfg(tid))
                el = (sc.values or {}).get("conversation_log") or []
                if el and str(el[-1]).startswith((HITL_PREFIX, HITL_SAT_PREFIX)):
                    chatbot.update_state(_cfg(tid), {"conversation_log": el[:-1]})
            except Exception: pass

            st.session_state["message_history"] = _load_history(tid)

            # Manim video
            try:
                snap2 = chatbot.get_state(config=_cfg(tid))
                vp = (snap2.values or {}).get("manim_video_path")
                if vp and os.path.exists(str(vp)):
                    st.divider(); st.subheader("🎬 Visual Explanation"); st.video(str(vp))
            except Exception: pass

            # ── MEMORY ── Refresh profile (problems_solved may have incremented)
            try: st.session_state["user_profile"] = get_user_profile(
                    st.session_state["student_id"])
            except Exception: pass

            # ── HITL ── Check if graph paused again (chained HITL)
            interrupted, p2 = _check_hitl(tid)
            if interrupted and p2:
                st.session_state.update(hitl_pending=True, hitl_payload=p2,
                    hitl_question=p2.get("prompt") or p2.get("message"),
                    hitl_type=p2.get("hitl_type",""))
            st.rerun()

    # ══════════════════════════════════════════════════════════════════════════
    #  NORMAL INPUT BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    else:
        mode = st.radio("Input", ["Text","Image","Audio"],
                        horizontal=True, label_visibility="collapsed")
        text_input = image_file = audio_file = None
        if mode == "Text":  text_input = st.chat_input("Enter your math problem…")
        elif mode == "Image": image_file = st.file_uploader("Image", type=["png","jpg","jpeg"], key="img_up")
        elif mode == "Audio": audio_file = st.file_uploader("Audio", type=["wav","mp3","m4a"], key="aud_up")

        if text_input or image_file or audio_file:
            tid = st.session_state["thread_id"]
            sid = st.session_state["student_id"]
            st.session_state["activity_log"] = []
            st.session_state["current_node"] = None
            render_activity_panel(activity_ph)

            # ── MEMORY ── make_initial_state seeds student_id + thread_id so:
            #   → retrieve_ltm node fetches student context before parsing
            #   → store_ltm node saves episodic/semantic/procedural after satisfaction
            sp = make_initial_state(student_id=sid, thread_id=tid)

            if text_input:
                sp["raw_text"] = text_input
                st.session_state["message_history"].append({"role":"user","content":text_input})
                with st.chat_message("user"): st.markdown(text_input)

            if image_file:
                fp=f"uploads/{image_file.name}"; open(fp,"wb").write(image_file.getbuffer())
                sp["image_path"]=fp
                st.session_state["message_history"].append({"role":"user","content":"[Image uploaded]"})
                with st.chat_message("user"): st.image(image_file, caption="Uploaded question")

            if audio_file:
                fp=f"uploads/{audio_file.name}"; open(fp,"wb").write(audio_file.getbuffer())
                sp["audio_path"]=fp
                st.session_state["message_history"].append({"role":"user","content":"[Audio uploaded]"})
                with st.chat_message("user"): st.audio(audio_file)

            with st.chat_message("assistant"):
                bph = st.empty(); rparts: list[str] = []
                def _stream():
                    for ev in chatbot.stream(sp, config=_cfg(tid), stream_mode="updates"):
                        t = _handle_chunk(ev, bph)
                        render_activity_panel(activity_ph)
                        if t: rparts.append(t); yield t
                st.write_stream(_stream())

            mark_all_done(); render_activity_panel(activity_ph)
            st.session_state["message_history"] = _load_history(tid)

            # Manim video
            try:
                fs = chatbot.get_state(config=_cfg(tid))
                vp = (fs.values or {}).get("manim_video_path")
                if vp and os.path.exists(str(vp)):
                    st.divider(); st.subheader("🎬 Visual Explanation"); st.video(str(vp))
                    add_step("manim_node","done",f"Video: {os.path.basename(str(vp))}")
                    render_activity_panel(activity_ph)
            except Exception: pass

            # ── HITL ── Did the graph pause after normal stream?
            interrupted, p = _check_hitl(tid)
            if interrupted and p:
                st.session_state.update(hitl_pending=True, hitl_payload=p,
                    hitl_question=p.get("prompt") or p.get("message"),
                    hitl_type=p.get("hitl_type",""))
                st.rerun()