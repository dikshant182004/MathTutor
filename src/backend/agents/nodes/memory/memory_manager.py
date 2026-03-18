"""
memory_manager.py
─────────────────
Redis-backed STM + LTM for the JEE tutor agent.

KEY DESIGN DECISIONS
────────────────────

1. Student identity
   Google OAuth gives us email "rahul@gmail.com".
   We hash it: student_id = sha256(email)[:16]
   All Redis keys namespace under this ID.
   Raw email stored once in user registry for display only.

2. STM — trimming + summarization
   LangGraph RedisSaver checkpoints AgentState at every step (handled automatically).
   What it does NOT manage is the growing `messages` list inside that state.
   We add on top of RedisSaver:
     a) Token counting via tiktoken after every solver_agent call.
     b) Sliding window — always keep last KEEP_LAST_N messages verbatim.
     c) Summarization — dropped context is summarized via LLM and injected
        as an AIMessage at position [1] (after SystemMessage).
        Summary is also persisted in Redis (stm:summary:<thread_id>) so it
        survives a server restart within the 2h TTL.

3. LTM — per-user, cross-thread
   Episodic / Semantic / Procedural — keyed by student_id.

4. Thread registry
   Sorted set threads:<student_id> (score = last_active timestamp).
   On login UI reads this to show "Previous conversations" sidebar.
   Each thread has a metadata hash with problem_summary, topic, outcome.

5. User registry
   Hash user:<student_id> — email, display_name, created_at, last_login,
   total_problems_solved.

KEY SCHEMA
──────────
  STM (RedisSaver internal):   checkpoint:<thread_id>:*    TTL 2h
  STM summary (ours):          stm:summary:<thread_id>     TTL 2h
  User registry:               user:<student_id>           no TTL
  Thread sorted set:           threads:<student_id>        scores = timestamps
  Thread metadata:             thread:<thread_id>:meta     TTL 7d
  LTM episodic:                episodic:<student_id>:<ep>  TTL 90d + decay
  LTM semantic:                semantic:<student_id>        no TTL
  LTM procedural:              procedural:<student_id>      no TTL

FLOW ON GOOGLE LOGIN
────────────────────
  1. OAuth callback gives email + display_name
  2. student_id = get_or_create_user(email, display_name)
  3. threads    = get_thread_history(student_id)   → sidebar
  4. For new problem: thread_id = create_thread(student_id)
  5. config = {"configurable": {"thread_id": thread_id}}
  6. graph.invoke({"student_id": student_id, "thread_id": thread_id, ...}, config)
  7. Inside solver_agent: call trim_messages_if_needed() before every LLM call
  8. After session: store LTM + update thread meta

Dependencies:
  pip install langgraph-checkpoint-redis redisvl redis tiktoken
"""

from __future__ import annotations

import hashlib
import math
import os
import time
import logging
from typing import Any

import redis as syncredis
import tiktoken
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage,
)
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.utils.vectorize import OpenAITextVectorizer
from langgraph.checkpoint.redis import RedisSaver
from langgraph.store.redis import RedisStore

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

REDIS_URL         = os.getenv("REDIS_URL", "redis://:jee_secret@localhost:6379")
EMBED_MODEL       = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM         = 1536

# STM trimming
TOKEN_LIMIT       = 6_000    # trim when messages exceed this
TOKEN_TARGET      = 3_000    # trim down to approximately this
KEEP_LAST_N       = 6        # always keep last N messages verbatim
TIKTOKEN_MODEL    = "gpt-4o" # for counting only

# TTLs
STM_SUMMARY_TTL   = 2  * 60 * 60   # 2 hours
THREAD_TTL        = 7  * 24 * 3600  # 7 days
EPISODIC_TTL      = 90 * 24 * 3600  # 90 days

# LTM
DECAY_THRESHOLD   = 0.05
TOP_K_EPISODES    = 3
MAX_THREADS_SHOWN = 20


# ──────────────────────────────────────────────────────────────────────────────
# KEY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _student_id_from_email(email: str) -> str:
    return hashlib.sha256(email.lower().strip().encode()).hexdigest()[:16]

def _user_key(sid: str)        -> str: return f"user:{sid}"
def _threads_key(sid: str)     -> str: return f"threads:{sid}"
def _thread_meta_key(tid: str) -> str: return f"thread:{tid}:meta"
def _stm_summary_key(tid: str) -> str: return f"stm:summary:{tid}"
def _ep_key(sid: str, eid: str)-> str: return f"episodic:{sid}:{eid}"
def _sem_key(sid: str)         -> str: return f"semantic:{sid}"
def _proc_key(sid: str)        -> str: return f"procedural:{sid}"
def _ep_index_name()           -> str: return "idx:episodic"


# ──────────────────────────────────────────────────────────────────────────────
# SINGLETONS
# ──────────────────────────────────────────────────────────────────────────────

_sync_client: syncredis.Redis | None = None
_vectorizer:  OpenAITextVectorizer | None = None
_tokenizer    = None


def get_sync_client() -> syncredis.Redis:
    global _sync_client
    if _sync_client is None:
        _sync_client = syncredis.from_url(REDIS_URL, decode_responses=True)
    return _sync_client

def get_vectorizer() -> OpenAITextVectorizer:
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = OpenAITextVectorizer(
            model=EMBED_MODEL,
            api_config={"api_key": os.getenv("OPENAI_API_KEY")},
        )
    return _vectorizer

def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model(TIKTOKEN_MODEL)
    return _tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# GRAPH INFRASTRUCTURE
# ──────────────────────────────────────────────────────────────────────────────

def build_stm_checkpointer() -> RedisSaver:
    """
    Returns RedisSaver for workflow.compile(checkpointer=...).
    Handles full AgentState persistence at every graph step automatically.
    Each thread_id is its own isolated namespace.
    """
    saver = RedisSaver(redis_url=REDIS_URL)
    saver.setup()
    logger.info("[STM] RedisSaver ready")
    return saver


def build_ltm_store() -> RedisStore:
    """Returns RedisStore for workflow.compile(store=...)."""
    store = RedisStore(redis_url=REDIS_URL)
    store.setup()
    logger.info("[LTM] RedisStore ready")
    return store


# ──────────────────────────────────────────────────────────────────────────────
# USER REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

def get_or_create_user(email: str, display_name: str) -> str:
    """
    Called on every Google OAuth login.
    Creates record on first login, updates last_login on subsequent ones.
    Returns student_id.

    Streamlit usage:
        student_id = get_or_create_user(info["email"], info["name"])
        st.session_state["student_id"] = student_id
    """
    client     = get_sync_client()
    student_id = _student_id_from_email(email)
    key        = _user_key(student_id)
    now        = time.time()

    if not client.exists(key):
        client.hset(key, mapping={
            "student_id":            student_id,
            "email":                 email,
            "display_name":          display_name,
            "created_at":            now,
            "last_login":            now,
            "total_problems_solved": 0,
        })
        logger.info(f"[User] New user | id={student_id}")
    else:
        client.hset(key, mapping={"last_login": now, "display_name": display_name})
        logger.info(f"[User] Login | id={student_id}")

    return student_id


def get_user_profile(student_id: str) -> dict | None:
    client = get_sync_client()
    data   = client.hgetall(_user_key(student_id))
    return data if data else None


def increment_problems_solved(student_id: str) -> None:
    get_sync_client().hincrby(_user_key(student_id), "total_problems_solved", 1)


# ──────────────────────────────────────────────────────────────────────────────
# THREAD REGISTRY
# ──────────────────────────────────────────────────────────────────────────────

def create_thread(student_id: str) -> str:
    """
    Creates a new thread for one problem-solving session.
    Returns thread_id.

    Streamlit usage:
        thread_id = create_thread(student_id)
        config    = {"configurable": {"thread_id": thread_id}}
        graph.invoke(initial_state | {"student_id": student_id,
                                       "thread_id":  thread_id}, config)
    """
    client    = get_sync_client()
    thread_id = f"{student_id}:{int(time.time() * 1000)}"
    now       = time.time()

    client.zadd(_threads_key(student_id), {thread_id: now})
    client.hset(_thread_meta_key(thread_id), mapping={
        "thread_id":       thread_id,
        "student_id":      student_id,
        "created_at":      now,
        "last_active":     now,
        "problem_summary": "",
        "topic":           "",
        "outcome":         "in_progress",
    })
    client.expire(_thread_meta_key(thread_id), THREAD_TTL)
    logger.info(f"[Thread] Created | id={thread_id}")
    return thread_id


def update_thread_meta(
    thread_id: str, problem_summary: str, topic: str, outcome: str
) -> None:
    """Updates the thread card shown in the sidebar after session ends."""
    client     = get_sync_client()
    student_id = thread_id.split(":")[0]
    now        = time.time()

    client.hset(_thread_meta_key(thread_id), mapping={
        "last_active":     now,
        "problem_summary": problem_summary[:120],
        "topic":           topic,
        "outcome":         outcome,
    })
    client.expire(_thread_meta_key(thread_id), THREAD_TTL)
    client.zadd(_threads_key(student_id), {thread_id: now}, xx=True)


def get_thread_history(student_id: str) -> list[dict]:
    """
    Returns past threads for the UI sidebar, newest first.
    Called on login.

    Returns:
        [{"thread_id": ..., "problem_summary": ..., "topic": ...,
          "outcome": ..., "last_active": ..., "created_at": ...}, ...]
    """
    client     = get_sync_client()
    thread_ids = client.zrevrange(_threads_key(student_id), 0, MAX_THREADS_SHOWN - 1)
    threads    = []
    for tid in thread_ids:
        meta = client.hgetall(_thread_meta_key(tid))
        if meta:
            threads.append(meta)
    logger.info(f"[Thread] Loaded {len(threads)} threads | student={student_id}")
    return threads


def load_thread_state(thread_id: str, checkpointer: RedisSaver) -> dict | None:
    """
    Loads AgentState snapshot for a past thread.
    Used when student clicks a past conversation to view/resume it.
    Returns None if the checkpoint has expired (>2h old).
    """
    config   = {"configurable": {"thread_id": thread_id}}
    snapshot = checkpointer.get(config)
    return snapshot.values if snapshot else None


# ──────────────────────────────────────────────────────────────────────────────
# STM TRIMMING + SUMMARIZATION
# ──────────────────────────────────────────────────────────────────────────────

def _count_tokens(messages: list[BaseMessage]) -> int:
    enc   = _get_tokenizer()
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total  += len(enc.encode(content)) + 4   # +4 per message overhead
    return total


def _summarize_messages(messages: list[BaseMessage], llm) -> str:
    """Calls the LLM to summarize a list of messages into 3-5 sentences."""
    lines = []
    for msg in messages:
        role    = type(msg).__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content.strip():
            lines.append(f"{role}: {content[:500]}")

    prompt = f"""Summarize this JEE math tutoring conversation in 3-5 sentences.
Include: what problem was being solved, what approach was taken, any errors found,
and what was resolved. Be specific — include key expressions and answers.

Conversation:
{chr(10).join(lines)}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return (response.content or "").strip()


def trim_messages_if_needed(
    messages:  list[BaseMessage],
    thread_id: str,
    llm,
) -> list[BaseMessage]:
    """
    Call this at the START of solver_agent, BEFORE building the message list
    to pass to the LLM.

    What it does:
        1. Count tokens across all messages.
        2. If under TOKEN_LIMIT → return as-is, no LLM call made.
        3. If over TOKEN_LIMIT:
             a. Split: [SystemMessages] + [middle] + [last KEEP_LAST_N]
             b. Summarize middle via LLM (absorbs any prior summary too).
             c. Persist new summary to Redis (stm:summary:<thread_id>, TTL 2h).
             d. Return [SystemMessages] + [AIMessage(summary)] + [last_N]

    The SystemMessage is always preserved at [0].
    The last KEEP_LAST_N messages are always verbatim (fresh context).
    Everything between them collapses into one AIMessage summary.

    Args:
        messages:  state["messages"] from AgentState.
        thread_id: used to persist and retrieve the running summary.
        llm:       the agent's LLM (same instance used by SolverAgent).

    Returns:
        Trimmed message list, safe to pass to any LLM call.

    Usage inside solver_agent:
        messages = trim_messages_if_needed(
            messages  = existing_msgs or [],
            thread_id = state.get("thread_id", ""),
            llm       = self.llm,
        )
    """
    if not messages:
        return messages

    total_tokens = _count_tokens(messages)
    logger.debug(f"[STM] tokens={total_tokens} thread={thread_id}")

    if total_tokens <= TOKEN_LIMIT:
        return messages

    logger.info(f"[STM] Trimming | {total_tokens} tokens > {TOKEN_LIMIT} limit")

    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    non_system  = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(non_system) <= KEEP_LAST_N:
        # Can't trim further — log and return as-is
        logger.warning("[STM] Cannot trim — too few non-system messages")
        return messages

    tail_msgs    = non_system[-KEEP_LAST_N:]
    to_summarize = non_system[:-KEEP_LAST_N]

    # Load existing summary from Redis so we build on it, not replace it
    client           = get_sync_client()
    existing_summary = client.get(_stm_summary_key(thread_id)) or ""

    if existing_summary:
        # Inject existing summary as first message so the new summary absorbs it
        to_summarize = [
            AIMessage(content=f"[Prior summary: {existing_summary}]")
        ] + to_summarize

    new_summary = _summarize_messages(to_summarize, llm)

    # Persist to Redis
    client.setex(_stm_summary_key(thread_id), STM_SUMMARY_TTL, new_summary)

    trimmed = system_msgs + [
        AIMessage(content=f"[Context summary — what happened before this point]\n{new_summary}")
    ] + tail_msgs

    logger.info(
        f"[STM] Trimmed {total_tokens} → {_count_tokens(trimmed)} tokens | "
        f"thread={thread_id}"
    )
    return trimmed


def get_stm_summary(thread_id: str) -> str | None:
    """Returns stored summary for a thread. Used when resuming past sessions."""
    return get_sync_client().get(_stm_summary_key(thread_id))


# ──────────────────────────────────────────────────────────────────────────────
# EPISODIC INDEX
# ──────────────────────────────────────────────────────────────────────────────

EPISODIC_INDEX_SCHEMA = {
    "index": {"name": _ep_index_name(), "prefix": "episodic:", "storage_type": "json"},
    "fields": [
        {"name": "$.student_id",  "type": "tag",     "as": "student_id"},
        {"name": "$.topic",       "type": "tag",     "as": "topic"},
        {"name": "$.difficulty",  "type": "tag",     "as": "difficulty"},
        {"name": "$.outcome",     "type": "tag",     "as": "outcome"},
        {"name": "$.timestamp",   "type": "numeric", "as": "timestamp"},
        {"name": "$.decay_score", "type": "numeric", "as": "decay_score"},
        {"name": "$.embedding",   "type": "vector",
         "attrs": {"algorithm": "HNSW", "datatype": "FLOAT32",
                   "dims": EMBED_DIM, "distance_metric": "COSINE"}},
    ],
}


def ensure_episodic_index() -> None:
    client = get_sync_client()
    try:
        client.ft(_ep_index_name()).info()
    except Exception:
        index = SearchIndex.from_dict(EPISODIC_INDEX_SCHEMA)
        index.connect(redis_url=REDIS_URL)
        index.create(overwrite=False)
        logger.info("[LTM] Episodic vector index created")


# ──────────────────────────────────────────────────────────────────────────────
# LTM WRITE
# ──────────────────────────────────────────────────────────────────────────────

def store_episodic_memory(
    student_id: str, topic: str, difficulty: str,
    problem_summary: str, final_answer: str,
    outcome: str, solve_attempts: int,
) -> str:
    ensure_episodic_index()
    client     = get_sync_client()
    vectorizer = get_vectorizer()
    ep_id      = f"{int(time.time() * 1000)}"
    now        = time.time()
    embedding  = vectorizer.embed(f"{topic} {difficulty} {problem_summary}", as_buffer=True)

    doc = {
        "student_id": student_id, "episode_id": ep_id,
        "topic": topic, "difficulty": difficulty,
        "problem_summary": problem_summary[:200], "final_answer": final_answer[:100],
        "outcome": outcome, "solve_attempts": solve_attempts,
        "timestamp": now, "access_count": 0, "decay_score": 1.0,
        "embedding": embedding,
    }
    key = _ep_key(student_id, ep_id)
    client.json().set(key, "$", doc)
    client.expire(key, EPISODIC_TTL)
    logger.info(f"[LTM] Episodic | student={student_id} topic={topic} outcome={outcome}")
    return ep_id


def update_semantic_memory(
    student_id: str, topic: str, outcome: str, mistake_pattern: str | None = None,
) -> None:
    client = get_sync_client()
    key    = _sem_key(student_id)
    ex     = client.json().get(key, "$")
    doc    = ex[0] if ex else {
        "student_id": student_id, "weak_topics": {}, "strong_topics": {},
        "mistake_patterns": [], "last_updated": time.time(),
    }
    if outcome == "correct":
        doc["strong_topics"][topic] = doc["strong_topics"].get(topic, 0) + 1
        doc["weak_topics"][topic]   = max(0, doc["weak_topics"].get(topic, 0) - 1)
    else:
        doc["weak_topics"][topic]   = doc["weak_topics"].get(topic, 0) + 1

    if mistake_pattern:
        found = False
        for e in doc["mistake_patterns"]:
            if e["pattern"] == mistake_pattern and e["topic"] == topic:
                e["count"] += 1; found = True; break
        if not found:
            doc["mistake_patterns"].append({"pattern": mistake_pattern, "topic": topic, "count": 1})

    doc["last_updated"] = time.time()
    client.json().set(key, "$", doc)


def update_procedural_memory(
    student_id: str, topic: str, strategy: str, success: bool, attempts: int,
) -> None:
    client = get_sync_client()
    key    = _proc_key(student_id)
    ex     = client.json().get(key, "$")
    doc    = ex[0] if ex else {"student_id": student_id, "strategy_success": {}, "last_updated": time.time()}

    doc["strategy_success"].setdefault(topic, {}).setdefault(strategy, {
        "success_count": 0, "total_count": 0, "attempts_sum": 0,
    })
    e = doc["strategy_success"][topic][strategy]
    e["total_count"]  += 1; e["attempts_sum"] += attempts
    if success: e["success_count"] += 1
    e["success_rate"] = round(e["success_count"] / e["total_count"], 2)
    e["attempts_avg"] = round(e["attempts_sum"]  / e["total_count"], 2)
    doc["last_updated"] = time.time()
    client.json().set(key, "$", doc)


# ──────────────────────────────────────────────────────────────────────────────
# LTM READ
# ──────────────────────────────────────────────────────────────────────────────

def retrieve_ltm(student_id: str, problem_text: str, topic: str) -> dict:
    """
    Fetches all LTM context for the current problem.
    Written to state["ltm_context"] by memory_manager_node (retrieve mode).
    """
    ensure_episodic_index()
    client     = get_sync_client()
    vectorizer = get_vectorizer()
    result: dict = {
        "similar_episodes": [], "weak_topics": {}, "strong_topics": {},
        "mistake_patterns": [], "best_strategy": None, "avg_attempts": None,
    }

    # Vector search
    try:
        q_emb = vectorizer.embed(f"{topic} {problem_text[:200]}", as_buffer=True)
        query = VectorQuery(
            vector=q_emb, vector_field_name="embedding",
            return_fields=["student_id", "episode_id", "topic", "difficulty",
                           "problem_summary", "final_answer", "outcome", "solve_attempts"],
            num_results=TOP_K_EPISODES,
            filter_expression=f"@student_id:{{{student_id}}}",
        )
        index = SearchIndex.from_dict(EPISODIC_INDEX_SCHEMA)
        index.connect(redis_url=REDIS_URL)
        eps   = index.query(query)

        for ep in eps:
            k = _ep_key(student_id, ep.get("episode_id", ""))
            if client.exists(k):
                client.json().numincrby(k, "$.access_count", 1)
                _refresh_decay_score(client, k)

        result["similar_episodes"] = [
            {f: ep.get(f) for f in ["topic","difficulty","problem_summary","final_answer","outcome","solve_attempts"]}
            for ep in eps
        ]
    except Exception as e:
        logger.warning(f"[LTM] Vector search failed: {e}")

    # Semantic
    try:
        sem = client.json().get(_sem_key(student_id), "$")
        if sem:
            d = sem[0]
            result["weak_topics"]      = d.get("weak_topics", {})
            result["strong_topics"]    = d.get("strong_topics", {})
            result["mistake_patterns"] = d.get("mistake_patterns", [])
    except Exception as e:
        logger.warning(f"[LTM] Semantic fetch failed: {e}")

    # Procedural
    try:
        proc = client.json().get(_proc_key(student_id), "$")
        if proc:
            strategies = proc[0].get("strategy_success", {}).get(topic, {})
            if strategies:
                best = max(strategies.items(),
                           key=lambda kv: (kv[1].get("success_rate",0), -kv[1].get("attempts_avg",99)))
                result["best_strategy"] = best[0]
                result["avg_attempts"]  = best[1].get("attempts_avg")
    except Exception as e:
        logger.warning(f"[LTM] Procedural fetch failed: {e}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# DECAY
# ──────────────────────────────────────────────────────────────────────────────

def _refresh_decay_score(client: syncredis.Redis, key: str) -> None:
    try:
        doc = client.json().get(key, "$")
        if not doc: return
        doc       = doc[0]
        days_old  = (time.time() - doc.get("timestamp", time.time())) / 86400
        score     = math.exp(-days_old / 30) * math.log(1 + doc.get("access_count", 0) + 1)
        client.json().set(key, "$.decay_score", round(score, 4))
    except Exception as e:
        logger.warning(f"[LTM] Decay refresh failed: {e}")


def prune_stale_episodic(student_id: str | None = None) -> int:
    """Run nightly via APScheduler or cron. Removes decayed episodic entries."""
    client  = get_sync_client()
    pattern = f"episodic:{student_id}:*" if student_id else "episodic:*:*"
    cutoff  = time.time() - (30 * 86400)
    pruned  = 0
    for key in client.keys(pattern):
        try:
            doc = client.json().get(key, "$")
            if not doc: continue
            doc = doc[0]
            _refresh_decay_score(client, key)
            doc = client.json().get(key, "$")[0]
            if doc.get("decay_score", 1.0) < DECAY_THRESHOLD and doc.get("timestamp", 0) < cutoff:
                client.delete(key); pruned += 1
        except Exception as e:
            logger.warning(f"[LTM] Prune error {key}: {e}")
    logger.info(f"[LTM] Pruned {pruned} stale entries")
    return pruned


# ──────────────────────────────────────────────────────────────────────────────
# LANGGRAPH NODE
# ──────────────────────────────────────────────────────────────────────────────

def memory_manager_node(state: dict) -> dict:
    """
    Two modes controlled by state["ltm_mode"]:

    "retrieve" — before solving:
        Fetches LTM context, writes state["ltm_context"].

    "store" — after student satisfied:
        Writes episodic/semantic/procedural, updates thread meta + user counter.
    """
    ltm_mode   = state.get("ltm_mode", "retrieve")
    student_id = state.get("student_id") or "anonymous"
    thread_id  = state.get("thread_id")  or ""

    if ltm_mode == "retrieve":
        parsed       = state.get("parsed_data") or {}
        problem_text = parsed.get("problem_text") or state.get("raw_text") or ""
        topic        = parsed.get("topic") or ""
        ltm_context  = retrieve_ltm(student_id=student_id, problem_text=problem_text, topic=topic)
        logger.info(f"[MemMgr] LTM retrieved | student={student_id}")
        return {"ltm_context": ltm_context}

    elif ltm_mode == "store":
        parsed          = state.get("parsed_data") or {}
        plan            = state.get("solution_plan") or {}
        solver_out      = state.get("solver_output") or {}
        verifier_out    = state.get("verifier_output") or {}
        topic           = parsed.get("topic") or ""
        difficulty      = plan.get("difficulty") or "medium"
        problem_text    = parsed.get("problem_text") or ""
        final_answer    = solver_out.get("final_answer") or ""
        solve_attempts  = state.get("solve_iterations") or 1
        verifier_status = verifier_out.get("status") or "incorrect"
        strategy        = plan.get("solver_strategy") or ""
        suggested_fix   = verifier_out.get("suggested_fix") or ""

        outcome = ("correct" if verifier_status == "correct"
                   else "hitl" if verifier_status == "needs_human"
                   else "incorrect")
        mistake = suggested_fix[:80].strip() if (suggested_fix and outcome != "correct") else None

        store_episodic_memory(student_id=student_id, topic=topic, difficulty=difficulty,
                              problem_summary=problem_text[:200], final_answer=final_answer,
                              outcome=outcome, solve_attempts=solve_attempts)
        update_semantic_memory(student_id=student_id, topic=topic, outcome=outcome, mistake_pattern=mistake)
        update_procedural_memory(student_id=student_id, topic=topic, strategy=strategy,
                                 success=(outcome=="correct"), attempts=solve_attempts)
        update_thread_meta(thread_id=thread_id, problem_summary=problem_text[:120],
                           topic=topic, outcome=outcome)
        if outcome == "correct":
            increment_problems_solved(student_id)

        logger.info(f"[MemMgr] LTM stored | student={student_id} outcome={outcome}")
        return {"ltm_stored": True}

    return {}


# ──────────────────────────────────────────────────────────────────────────────
# LTM CONTEXT FORMATTERS
# ──────────────────────────────────────────────────────────────────────────────

def format_ltm_for_solver(ltm_context: dict, topic: str) -> str:
    if not ltm_context: return ""
    lines = []
    best = ltm_context.get("best_strategy")
    avg  = ltm_context.get("avg_attempts")
    if best:
        lines.append(f"STUDENT HISTORY — for {topic} this student responds best to: {best} (avg {avg:.1f} attempts).")
    weak = ltm_context.get("weak_topics", {})
    if topic in weak and weak[topic] > 1:
        lines.append(f"WEAK AREA — struggled with {topic} {weak[topic]} times. Be especially clear on each step.")
    patterns = [p for p in ltm_context.get("mistake_patterns", []) if p.get("topic")==topic and p.get("count",0)>1]
    if patterns:
        lines.append(f"KNOWN MISTAKES in {topic}: " + "; ".join(p["pattern"] for p in patterns[:2]) + ".")
    eps = ltm_context.get("similar_episodes", [])
    if eps:
        ep = eps[0]
        lines.append(f"SIMILAR PAST PROBLEM — {ep['topic']} ({ep['difficulty']}), outcome={ep['outcome']}, answer={ep['final_answer']}.")
    return "\n".join(lines)


def format_ltm_for_explainer(ltm_context: dict, topic: str) -> str:
    if not ltm_context: return ""
    lines = []
    weak = ltm_context.get("weak_topics", {})
    if topic in weak and weak[topic] > 0:
        lines.append(f"PERSONALISATION — this student struggles with {topic}. Add extra intuition-building.")
    patterns = [p for p in ltm_context.get("mistake_patterns", []) if p.get("topic")==topic]
    if patterns:
        lines.append("COMMON MISTAKES THIS STUDENT MAKES in " + topic + ": "
                     + "; ".join(p["pattern"] for p in patterns[:3]) + ". Address in common_mistakes field.")
    return "\n".join(lines)