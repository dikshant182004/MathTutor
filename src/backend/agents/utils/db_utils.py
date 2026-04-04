from __future__ import annotations

import hashlib
import time
import redis as syncredis
import tiktoken
from backend.agents import logger, Optional
from langgraph.checkpoint.redis import RedisSaver

from backend.agents.nodes.memory import (
    REDIS_URL,
    TIKTOKEN_MODEL,
    STM_SUMMARY_TTL,
    THREAD_TTL,
    MAX_THREADS_SHOWN,
)

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — SINGLETONS
#  One connection / encoder instance shared for the lifetime of the process.
# ══════════════════════════════════════════════════════════════════════════════

_sync_client: Optional[syncredis.Redis] = None
_tokenizer = None


def get_sync_client() -> syncredis.Redis:
    """
    Returns a cached synchronous Redis client.
    Used by: ALL db_utils functions, memory_manager.py LTM read/write.
    Default: redis://:jee_secret@localhost:6379
    """
    global _sync_client
    if _sync_client is None:
        _sync_client = syncredis.from_url(REDIS_URL, decode_responses=True)
        logger.info("[DB] Redis sync client created")
    return _sync_client


def get_tokenizer():
    """
    Returns a cached tiktoken encoder for token counting.
    Used by: _count_tokens() in memory_manager.py (STM trimming).
    """
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model(TIKTOKEN_MODEL)
        logger.info(f"[DB] tiktoken encoder loaded ({TIKTOKEN_MODEL})")
    return _tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — REDIS KEY HELPERS
#  ALL Redis key patterns are defined here in one place.
#  Key naming convention:
#    <type>:<student_id>           for per-student data
#    <type>:<thread_id>:meta       for per-thread data
#    <type>:<student_id>:<ep_id>   for per-episode data
# ══════════════════════════════════════════════════════════════════════════════

def student_id_from_email(email: str) -> str:
    """
    Hashes email to a 16-char hex string used as the student's Redis namespace.
    """
    return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:16]


def user_key(sid: str) -> str:
    """user:<student_id>  — hash of user profile fields"""
    return f"user:{sid}"


def threads_key(sid: str) -> str:
    """threads:<student_id>  — sorted set, score = last_active timestamp"""
    return f"threads:{sid}"


def thread_meta_key(tid: str) -> str:
    """thread:<thread_id>:meta  — hash of sidebar card fields"""
    return f"thread:{tid}:meta"


def stm_summary_key(tid: str) -> str:
    """stm:summary:<thread_id>  — plain string, rolling LLM summary of trimmed messages"""
    return f"stm:summary:{tid}"


def episodic_key(sid: str, eid: str) -> str:
    """episodic:<student_id>:<episode_id>  — JSON doc of one solved problem"""
    return f"episodic:{sid}:{eid}"


def semantic_key(sid: str) -> str:
    """semantic:<student_id>  — JSON doc of weak/strong topics + mistake patterns"""
    return f"semantic:{sid}"


def procedural_key(sid: str) -> str:
    """procedural:<student_id>  — JSON doc of strategy success rates"""
    return f"procedural:{sid}"

# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — USER REGISTRY
#  Tracks every student who has ever logged in.
#  Schema: user:<student_id> is a Redis hash with these fields:
#    student_id, email, display_name, created_at, last_login,
#    total_problems_solved
# ══════════════════════════════════════════════════════════════════════════════

def get_or_create_user(email: str, display_name: str) -> str:
    """
    Called on every Google OAuth login (in new_app.py _handle_google_oauth_callback).

    First login  → creates user hash in Redis, returns student_id
    Subsequent   → updates last_login + display_name, returns same student_id

    The student_id returned here is stored in st.session_state["student_id"]
    for the entire browser session, and threaded through every graph invocation
    as state["student_id"].
    """
    client = get_sync_client()
    student_id = student_id_from_email(email)
    key = user_key(student_id)
    now = time.time()

    if not client.exists(key):
        client.hset(key, mapping={
            "student_id":            student_id,
            "email":                 email,
            "display_name":          display_name,
            "created_at":            now,
            "last_login":            now,
            "total_problems_solved": 0,
        })
        logger.info(f"[User] New user created | id={student_id}")
    else:
        client.hset(key, mapping={"last_login": now, "display_name": display_name})
        logger.info(f"[User] Login | id={student_id}")

    return student_id


def get_user_profile(student_id: str) -> Optional[dict]:
    """
    Returns the full user hash as a dict, or None if the user doesn't exist.
    Not called in the current flow — available for a future profile page.
    """
    client = get_sync_client()
    data   = client.hgetall(user_key(student_id))
    return data if data else None


def increment_problems_solved(student_id: str) -> None:
    """
    Atomically increments total_problems_solved in the user hash.
    Called in memory_manager_node (store mode) when verifier_status == "correct".
    Uses Redis HINCRBY — safe under concurrent writes.
    """
    get_sync_client().hincrby(user_key(student_id), "total_problems_solved", 1)


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — THREAD REGISTRY
#  Each "thread" = one problem-solving session (one chat conversation).
#  Schema:
#    threads:<student_id>        sorted set, member=thread_id, score=timestamp
#    thread:<thread_id>:meta     hash: problem_summary, topic, outcome, timestamps
# ══════════════════════════════════════════════════════════════════════════════

def create_thread(student_id: str) -> str:
    """
    Creates a new thread for one problem-solving session.
    Called by: new_app.py _init_session() and reset_chat()

    Thread ID format: "<student_id>:<timestamp_ms>"
    This embeds student ownership directly in the ID (useful for scanning).

    Usage in new_app.py:
        tid    = create_thread(student_id)
        config = {"configurable": {"thread_id": tid}}
        chatbot.stream(make_initial_state(...), config)
    """
    client    = get_sync_client()
    thread_id = f"{student_id}:{int(time.time() * 1000)}"
    now       = time.time()

    # Add to sorted set (score = now, so zrevrange gives newest first)
    client.zadd(threads_key(student_id), {thread_id: now})

    # Create metadata hash for the sidebar card
    client.hset(thread_meta_key(thread_id), mapping={
        "thread_id":       thread_id,
        "student_id":      student_id,
        "created_at":      now,
        "last_active":     now,
        "problem_summary": "",
        "topic":           "",
        "outcome":         "in_progress",
    })
    client.expire(thread_meta_key(thread_id), THREAD_TTL)

    logger.info(f"[Thread] Created | id={thread_id}")
    return thread_id


def update_thread_meta(
    thread_id: str, problem_summary: str, topic: str, outcome: str
) -> None:
    """
    Updates the sidebar card for this thread after a session completes.
    Called by memory_manager_node in store mode (after student is satisfied).

    outcome values: "correct" | "incorrect" | "hitl" | "in_progress"
    """
    client     = get_sync_client()
    student_id = thread_id.split(":")[0]
    now        = time.time()

    client.hset(thread_meta_key(thread_id), mapping={
        "last_active":     now,
        "problem_summary": problem_summary[:120],
        "topic":           topic,
        "outcome":         outcome,
    })
    client.expire(thread_meta_key(thread_id), THREAD_TTL)
    # Update score in sorted set so it stays "recent"
    client.zadd(threads_key(student_id), {thread_id: now}, xx=True)


def get_thread_history(student_id: str) -> list[dict]:
    """
    Returns the most recent MAX_THREADS_SHOWN threads for the sidebar.
    Called by: new_app.py _init_session() and after reset_chat().

    Returns list of dicts (thread metadata hashes), newest first.
    """
    client     = get_sync_client()
    thread_ids = client.zrevrange(
        threads_key(student_id), 0, MAX_THREADS_SHOWN - 1
    )
    threads = []
    for tid in thread_ids:
        meta = client.hgetall(thread_meta_key(tid))
        if meta:
            threads.append(meta)
    logger.info(f"[Thread] Loaded {len(threads)} threads | student={student_id}")
    return threads


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — STM SUMMARY PERSISTENCE
#  The rolling summary of trimmed messages lives in Redis so it survives
#  server restarts within the 2h TTL window.
#  Key: stm:summary:<thread_id>  (plain Redis string, not JSON)
# ══════════════════════════════════════════════════════════════════════════════

def save_stm_summary(thread_id: str, summary: str) -> None:
    """
    Persists a new rolling summary string to Redis.
    Called by trim_messages_if_needed() in memory_manager.py after each trim.
    TTL is reset to 2h on every write.
    """
    get_sync_client().setex(stm_summary_key(thread_id), STM_SUMMARY_TTL, summary)
    logger.debug(f"[STM] Summary saved | thread={thread_id} len={len(summary)}")


def load_stm_summary(thread_id: str) -> Optional[str]:
    """
    Retrieves the current rolling summary for a thread.
    Returns None if the key has expired or never existed.
    Called by trim_messages_if_needed() to build on the previous summary
    (so each trim absorbs history, not just the last segment).
    Also available for "resume past session" to seed the context.
    """
    val = get_sync_client().get(stm_summary_key(thread_id))
    logger.debug(f"[STM] Summary loaded | thread={thread_id} found={val is not None}")
    return val


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — GRAPH INFRASTRUCTURE
#  build_stm_checkpointer is here because it returns a LangGraph object
#  (RedisSaver) that graph.py needs at compile time.
# ══════════════════════════════════════════════════════════════════════════════

def build_stm_checkpointer() -> RedisSaver:
    """
    Builds and returns the RedisSaver checkpointer for graph compilation.
    Called once in graph.py: MathTutorWorkflow.__init__()

    What RedisSaver does automatically (no code needed from us):
      - After EVERY node runs, serialises the entire AgentState and writes it
        to Redis under key: checkpoint:<thread_id>:*
      - On HITL interrupt(), the state is checkpointed so graph.invoke(Command(resume=...))
        can restore exactly where it left off
      - TTL: 2 hours (matches STM_SUMMARY_TTL)

    This is DIFFERENT from LTM (which we manage manually).
    STM = the live conversation state, scoped to one thread.
    LTM = cross-thread student profile, managed in memory_manager.py.
    """
    saver = RedisSaver(redis_url=REDIS_URL)
    saver.setup()
    logger.info("[STM] RedisSaver checkpointer ready")
    return saver