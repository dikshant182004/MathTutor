from __future__ import annotations
import hashlib
import time

_USER_PREFIX = "user:"

def _short_id(full_id: str, length: int = 8) -> str:
    return full_id[:length] + "…"


def _epoch_to_date(ts) -> str:
    try:
        return time.strftime("%d %b %Y", time.localtime(float(ts)))
    except Exception:
        return str(ts)


def build_graph_data(
    student_id: str,
    redis_client,
    checkpointer,
    get_thread_history,
    max_threads: int = 15,
    include_agent_nodes: bool = True,
) -> dict:
    """
    Returns {"nodes": [...], "edges": [...]} ready for vis.js.

    Key pattern alignment (must match db_utils.py / memory_manager.py):
        episodic:{student_id}:{episode_id}   — JSON doc, client.json().set(...)
        semantic:{student_id}                — JSON doc, client.json().set(...)
        procedural:{student_id}              — JSON doc, client.json().set(...)
        user:{student_id}                    — Redis hash, client.hset(...)
        thread:{thread_id}:meta              — Redis hash, client.hset(...)
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

        if not include_agent_nodes:
            continue

        try:
            cfg  = {"configurable": {"thread_id": tid}}
            snap = checkpointer.get(cfg)
            # RedisSaver.get() returns a CheckpointTuple(config, checkpoint, metadata, ...)
            # .values lives on the checkpoint field, not directly on the tuple
            if snap is None:
                vals = {}
            elif hasattr(snap, "checkpoint") and isinstance(getattr(snap, "checkpoint", None), dict):
                # LangGraph RedisSaver returns a CheckpointTuple; channel_values is the state
                vals = snap.checkpoint.get("channel_values", {})
            elif hasattr(snap, "values") and isinstance(snap.values, dict):
                vals = snap.values
            else:
                vals = {}
        except Exception:
            vals = {}

        # Agent payload log → one node per entry
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

    # ── LTM nodes — FIXED key patterns to match db_utils.py / memory_manager.py
    # ─────────────────────────────────────────────────────────────────────────
    # episodic:{student_id}:{episode_id}  → JSON doc via client.json().set(...)
    # semantic:{student_id}               → JSON doc via client.json().set(...)
    # procedural:{student_id}             → JSON doc via client.json().set(...)

    # ── Episodic memories ─────────────────────────────────────────────────────
    try:
        ep_keys = redis_client.keys(f"episodic:{student_id}:*")
    except Exception:
        ep_keys = []

    for raw_key in ep_keys:
        key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
        try:
            raw_data = redis_client.json().get(key, "$")
            mem = raw_data[0] if raw_data else {}
        except Exception:
            mem = {}
        if not mem:
            continue

        ep_id   = mem.get("episode_id", key.split(":")[-1])
        node_id = f"episodic__{ep_id}"
        summary = (mem.get("problem_summary") or ep_id)[:45]

        decay_val = mem.get("decay_score", "—")
        _add_node({
            "id":     node_id,
            "label":  summary[:30],
            "type":   "episodic",
            "group":  "episodic",
            "detail": {
                "Type":        "Episodic",
                "Problem":     summary,
                "Topic":       mem.get("topic", "—"),
                "Difficulty":  mem.get("difficulty", "—"),
                "Outcome":     mem.get("outcome", "—"),
                "Answer":      mem.get("final_answer", "—"),
                "Attempts":    str(mem.get("solve_attempts", "—")),
                "Decay score": str(decay_val),
                "Created":     _epoch_to_date(mem.get("timestamp", 0)),
            },
            "title": f"<b>Episodic</b><br>{summary}",
        })
        _add_edge({
            "from":   student_id,
            "to":     node_id,
            "label":  "has_episodic",
            "arrows": "to",
            "dashes": False,
        })

    # ── Semantic memory ───────────────────────────────────────────────────────
    try:
        sem_raw = redis_client.json().get(f"semantic:{student_id}", "$")
        sem = sem_raw[0] if sem_raw else {}
    except Exception:
        sem = {}

    if sem:
        weak   = sem.get("weak_topics", {})
        strong = sem.get("strong_topics", {})
        patterns = sem.get("mistake_patterns", [])

        # One node for the semantic profile
        sem_node_id = f"semantic__{student_id}"
        weak_str   = ", ".join(f"{k}({v})" for k, v in weak.items() if v > 0) or "—"
        strong_str = ", ".join(f"{k}({v})" for k, v in strong.items() if v > 0) or "—"
        pat_str    = "; ".join(
            p.get("pattern", "")[:40] for p in patterns[:3]
        ) or "—"
        _add_node({
            "id":     sem_node_id,
            "label":  "Semantic profile",
            "type":   "semantic",
            "group":  "semantic",
            "detail": {
                "Type":             "Semantic",
                "Weak topics":      weak_str[:120],
                "Strong topics":    strong_str[:120],
                "Mistake patterns": pat_str[:200],
                "Last updated":     _epoch_to_date(sem.get("last_updated", 0)),
            },
            "title": "<b>Semantic memory</b><br>Topic strengths &amp; mistakes",
        })
        _add_edge({
            "from":   student_id,
            "to":     sem_node_id,
            "label":  "has_semantic",
            "arrows": "to",
            "dashes": False,
        })

        # One child node per weak topic (count > 0)
        for topic, count in weak.items():
            if count < 1:
                continue
            wnid = f"semantic_weak__{student_id}__{topic}"
            _add_node({
                "id":    wnid,
                "label": topic,
                "type":  "semantic",
                "group": "semantic",
                "detail": {
                    "Type":       "Weak topic",
                    "Topic":      topic,
                    "Fail count": str(count),
                },
                "title": f"<b>Weak: {topic}</b><br>{count} struggles",
            })
            _add_edge({"from": sem_node_id, "to": wnid, "label": "weak_topic",
                       "arrows": "to", "dashes": True})

        # One child node per strong topic (count > 0)
        for topic, count in strong.items():
            if count < 1:
                continue
            snid = f"semantic_strong__{student_id}__{topic}"
            _add_node({
                "id":    snid,
                "label": topic,
                "type":  "semantic",
                "group": "semantic",
                "detail": {
                    "Type":          "Strong topic",
                    "Topic":         topic,
                    "Success count": str(count),
                },
                "title": f"<b>Strong: {topic}</b><br>{count} successes",
            })
            _add_edge({"from": sem_node_id, "to": snid, "label": "strong_topic",
                       "arrows": "to", "dashes": True})

        # One child node per unique mistake pattern
        for pat in patterns:
            pattern_text = pat.get("pattern", "")
            pat_topic    = pat.get("topic", "")
            pat_count    = pat.get("count", 1)
            if not pattern_text:
                continue
            # Use a hash of pattern text as a stable node ID
            pid  = hashlib.md5(f"{student_id}{pat_topic}{pattern_text}".encode()).hexdigest()[:10]
            mnid = f"semantic_mistake__{student_id}__{pid}"
            _add_node({
                "id":    mnid,
                "label": pattern_text[:28] + ("…" if len(pattern_text) > 28 else ""),
                "type":  "semantic",
                "group": "semantic",
                "detail": {
                    "Type":    "Mistake pattern",
                    "Topic":   pat_topic or "—",
                    "Pattern": pattern_text[:200],
                    "Count":   str(pat_count),
                },
                "title": f"<b>Mistake ({pat_topic})</b><br>{pattern_text[:60]}",
            })
            _add_edge({"from": sem_node_id, "to": mnid, "label": "mistake_pattern",
                       "arrows": "to", "dashes": True})

    # ── Procedural memory ─────────────────────────────────────────────────────
    try:
        proc_raw = redis_client.json().get(f"procedural:{student_id}", "$")
        proc = proc_raw[0] if proc_raw else {}
    except Exception:
        proc = {}

    if proc:
        strats = proc.get("strategy_success", {})
        proc_node_id = f"procedural__{student_id}"

        # Show only the BEST strategy per topic in the summary (not all strategies)
        best_per_topic = []
        for _topic, _topic_strats in strats.items():
            if not _topic_strats:
                continue
            _best_name, _best_data = max(
                _topic_strats.items(),
                key=lambda kv: (kv[1].get("success_rate", 0),
                                -kv[1].get("attempts_avg", 99)),
            )
            best_per_topic.append(
                f"{_topic}: {_best_name[:40]} "
                f"({_best_data.get('success_rate', 0):.0%} | "
                f"avg {_best_data.get('attempts_avg', '?')} attempts)"
            )

        _add_node({
            "id":     proc_node_id,
            "label":  "Procedural profile",
            "type":   "procedural",
            "group":  "procedural",
            "detail": {
                "Type":         "Procedural",
                "Best strategy per topic": "; ".join(best_per_topic[:5]) or "—",
                "Topics tracked": str(len(strats)),
                "Last updated": _epoch_to_date(proc.get("last_updated", 0)),
            },
            "title": "<b>Procedural memory</b><br>Strategy effectiveness",
        })
        _add_edge({
            "from":   student_id,
            "to":     proc_node_id,
            "label":  "has_procedural",
            "arrows": "to",
            "dashes": False,
        })

        # One child node per topic strategy
        for topic, topic_strats in strats.items():
            if not topic_strats:
                continue
            best = max(topic_strats.items(),
                       key=lambda kv: (kv[1].get("success_rate", 0),
                                       -kv[1].get("attempts_avg", 99)))
            strat_name, data = best
            snid = f"procedural_strat__{student_id}__{topic}"
            _add_node({
                "id":    snid,
                "label": topic,
                "type":  "procedural",
                "group": "procedural",
                "detail": {
                    "Type":         "Strategy",
                    "Topic":        topic,
                    "Best strategy":strat_name,
                    "Success rate": f"{data.get('success_rate', 0):.0%}",
                    "Avg attempts": str(data.get("attempts_avg", "—")),
                },
                "title": f"<b>{topic}</b><br>{strat_name}",
            })
            _add_edge({"from": proc_node_id, "to": snid, "label": "topic_strategy",
                       "arrows": "to", "dashes": True})

    return {"nodes": nodes, "edges": edges}