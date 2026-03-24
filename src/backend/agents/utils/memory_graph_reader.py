from __future__ import annotations
import json
import time
from typing import Any

# ── Redis key patterns ────────────────────────────────────────────────────────
# LTM keys follow: ltm:{student_id}:{memory_type}:{memory_id}
# STM checkpoints: managed by LangGraph RedisSaver

_LTM_PREFIX  = "ltm:"
_USER_PREFIX = "user:"


def _safe_json(val: Any) -> Any:
    """Return val if JSON-serialisable, else str(val)."""
    try:
        json.dumps(val)
        return val
    except Exception:
        return str(val)


def _short_id(full_id: str, length: int = 8) -> str:
    return full_id[:length] + "…"


def _epoch_to_date(ts) -> str:
    try:
        return time.strftime("%d %b %Y", time.localtime(float(ts)))
    except Exception:
        return str(ts)


# ── Main graph builder ────────────────────────────────────────────────────────

def build_graph_data(
    student_id: str,
    redis_client,        # raw redis.Redis sync client
    checkpointer,        # LangGraph RedisSaver
    get_thread_history,  # callable(student_id) -> list[dict]
    max_threads: int = 15,
    include_agent_nodes: bool = True,
) -> dict:
    """
    Returns {"nodes": [...], "edges": [...]} ready for vis.js.

    Node schema:
        id, label, type, group, title (hover HTML), detail (sidebar dict)

    Edge schema:
        from, to, label, arrows, dashes
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    seen_node_ids: set[str] = set()

    def _add_node(node: dict):
        if node["id"] not in seen_node_ids:
            seen_node_ids.add(node["id"])
            nodes.append(node)

    def _add_edge(edge: dict):
        edges.append(edge)

    # ── Student root ──────────────────────────────────────────────────────────
    try:
        user_raw = redis_client.hgetall(f"{_USER_PREFIX}{student_id}")
        user_info = {
            k.decode() if isinstance(k, bytes) else k:
            v.decode() if isinstance(v, bytes) else v
            for k, v in user_raw.items()
        } if user_raw else {}
    except Exception:
        user_info = {}

    _add_node({
        "id":     student_id,
        "label":  user_info.get("display_name", "Student"),
        "type":   "student",
        "group":  "student",
        "detail": {
            "Student ID":      _short_id(student_id),
            "Name":            user_info.get("display_name", "—"),
            "Email":           user_info.get("email", "—"),
            "Problems solved": user_info.get("total_problems_solved", "—"),
            "Member since":    _epoch_to_date(user_info.get("created_at", 0)),
            "Last login":      _epoch_to_date(user_info.get("last_login", 0)),
        },
        "title": f"<b>{user_info.get('display_name','Student')}</b><br>Root node",
    })

    # ── Thread / STM nodes ────────────────────────────────────────────────────
    try:
        threads_meta = get_thread_history(student_id)[:max_threads]
    except Exception:
        threads_meta = []

    for meta in threads_meta:
        tid     = meta.get("thread_id", "")
        summary = (meta.get("problem_summary") or "")[:35] or _short_id(tid)
        topic   = meta.get("topic", "")
        outcome = meta.get("outcome", "")

        session_node = {
            "id":     tid,
            "label":  summary,
            "type":   "session",
            "group":  "session",
            "detail": {
                "Thread ID": _short_id(tid, 12),
                "Problem":   summary,
                "Topic":     topic or "—",
                "Outcome":   outcome or "—",
                "Date":      _epoch_to_date(meta.get("created_at", 0)),
            },
            "title": f"<b>{summary}</b><br>Topic: {topic}",
        }
        _add_node(session_node)
        _add_edge({"from": student_id, "to": tid, "label": "has_session",
                   "arrows": "to", "dashes": False})

        # Read checkpoint state for this thread
        if not include_agent_nodes:
            continue
        try:
            cfg  = {"configurable": {"thread_id": tid}}
            snap = checkpointer.get(cfg)
            vals = (snap.values or {}) if snap else {}
        except Exception:
            vals = {}

        # Agent payload log → agent_node per entry
        for entry in (vals.get("agent_payload_log") or []):
            node_name = entry.get("node", "unknown")
            nid       = f"{tid}__{node_name}"
            fields    = {k: str(v)[:120] for k, v in (entry.get("fields") or {}).items()
                         if v is not None and str(v) not in ("", "None", "none")}
            _add_node({
                "id":     nid,
                "label":  node_name.replace("_", "\n"),
                "type":   "agent",
                "group":  "agent",
                "detail": {
                    "Node":    node_name,
                    "Summary": entry.get("summary", "")[:100],
                    **fields,
                },
                "title": f"<b>{node_name}</b><br>{entry.get('summary','')[:80]}",
            })
            _add_edge({"from": tid, "to": nid, "label": "ran_node",
                       "arrows": "to", "dashes": True})

        # Tool calls from messages
        for msg in (vals.get("messages") or []):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                tool_name = tc.get("name", "tool")
                tnid      = f"{tid}__tool__{tool_name}"
                _add_node({
                    "id":     tnid,
                    "label":  tool_name.replace("_", "\n"),
                    "type":   "tool",
                    "group":  "tool",
                    "detail": {
                        "Tool":  tool_name,
                        "Args":  str(tc.get("args", {}))[:200],
                        "Thread": _short_id(tid, 12),
                    },
                    "title": f"<b>{tool_name}</b>",
                })
                _add_edge({"from": tid, "to": tnid, "label": "used_tool",
                           "arrows": "to", "dashes": True})

    # ── LTM nodes from Redis ──────────────────────────────────────────────────
    # Key pattern: ltm:{student_id}:{type}:* (hash keys)
    for mem_type in ("episodic", "semantic", "procedural"):
        try:
            pattern = f"{_LTM_PREFIX}{student_id}:{mem_type}:*"
            keys    = redis_client.keys(pattern)
        except Exception:
            keys = []

        for raw_key in keys:
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            try:
                raw_data = redis_client.hgetall(key)
                mem = {
                    (k.decode() if isinstance(k, bytes) else k):
                    (v.decode() if isinstance(v, bytes) else v)
                    for k, v in raw_data.items()
                }
            except Exception:
                mem = {}

            mem_id  = key.split(":")[-1]
            node_id = f"ltm__{mem_type}__{mem_id}"
            summary = (mem.get("summary") or mem.get("description") or
                       mem.get("content") or mem_id)[:45]

            detail = {
                "Type":        mem_type.capitalize(),
                "Summary":     summary,
                "Created":     _epoch_to_date(mem.get("created_at", 0)),
                "Decay score": mem.get("decay_score", "—"),
            }
            if mem_type == "episodic":
                detail.update({
                    "Problem":  mem.get("problem_text", "—")[:80],
                    "Outcome":  mem.get("outcome", "—"),
                    "Error":    mem.get("error_pattern", "—"),
                    "Topic":    mem.get("topic", "—"),
                })
            elif mem_type == "semantic":
                detail.update({
                    "Concept":     mem.get("concept", "—"),
                    "Confidence":  mem.get("confidence", "—"),
                    "Formula":     mem.get("formula", "—"),
                    "Last seen":   _epoch_to_date(mem.get("last_seen", 0)),
                })
            elif mem_type == "procedural":
                detail.update({
                    "Strategy":      mem.get("strategy", "—"),
                    "Effectiveness": mem.get("effectiveness", "—"),
                    "Applied":       mem.get("applied_count", "—") + " times",
                    "Learned from":  mem.get("learned_from", "—"),
                })

            _add_node({
                "id":     node_id,
                "label":  summary[:30],
                "type":   mem_type,
                "group":  mem_type,
                "detail": detail,
                "title":  f"<b>{mem_type.capitalize()}</b><br>{summary}",
            })
            _add_edge({
                "from":   student_id,
                "to":     node_id,
                "label":  f"has_{mem_type}",
                "arrows": "to",
                "dashes": False,
            })

            # Link episodic memories to their source thread if thread_id is stored
            src_thread = mem.get("thread_id")
            if src_thread and src_thread in seen_node_ids:
                _add_edge({
                    "from":   src_thread,
                    "to":     node_id,
                    "label":  "generated",
                    "arrows": "to",
                    "dashes": True,
                })

    return {"nodes": nodes, "edges": edges}