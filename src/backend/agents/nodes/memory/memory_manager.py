from __future__ import annotations

import math
import time
import numpy as np
from backend.agents import logger, BaseMessage, SystemMessage, HumanMessage, AIMessage
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery

from backend.agents.nodes.tools.tools import _embed_texts
from backend.agents.utils.db_utils import (
    get_sync_client,
    get_tokenizer,
    episodic_key,
    semantic_key,
    procedural_key,
    save_stm_summary,
    load_stm_summary,
    update_thread_meta,
    increment_problems_solved,
)

from backend.agents.nodes.memory import (
    REDIS_URL,
    EMBED_INPUT_TYPE_DOC,
    EMBED_INPUT_TYPE_QUERY,
    TOKEN_LIMIT,
    KEEP_LAST_N,
    EPISODIC_TTL,
    DECAY_THRESHOLD,
    TOP_K_EPISODES,
    EPISODIC_INDEX_NAME,
    EPISODIC_INDEX_SCHEMA,
)

# ══════════════════════════════════════════════════════════════════════════════
#  STM TRIMMING + SUMMARISATION
#  Called at the START of solver_agent, before building the message list.
# ══════════════════════════════════════════════════════════════════════════════

def _count_tokens(messages: list[BaseMessage]) -> int:
    enc   = get_tokenizer()
    total = 0
    for msg in messages:
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        total  += len(enc.encode(content)) + 4
    return total


def _summarize_messages(messages: list[BaseMessage], llm) -> str:
    lines = []
    for msg in messages:
        role    = type(msg).__name__.replace("Message", "")
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if content.strip():
            lines.append(f"{role}: {content[:500]}")

    prompt = (
        "Summarize this JEE math tutoring conversation in 3-5 sentences. "
        "Include: what problem was being solved, what approach was taken, "
        "any errors found, and what was resolved. "
        "Be specific — include key expressions and answers.\n\n"
        "Conversation:\n" + "\n".join(lines)
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return (response.content or "").strip()


def trim_messages_if_needed(
    messages:  list[BaseMessage],
    thread_id: str,
    llm,
) -> list[BaseMessage]:
    """
    Purpose: keep the message list under TOKEN_LIMIT tokens by summarising
    older context and replacing it with a single AIMessage summary.
    Result: the LLM in solver_agent always sees:
        [system prompt] + [rolling summary of all prior context] + [last 6 messages]
    Memory is never lost — it's compressed, not deleted.

    Redis key: stm:summary:<thread_id>  (plain string, TTL 2h)
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
        logger.warning("[STM] Cannot trim — too few non-system messages")
        return messages

    tail_msgs    = non_system[-KEEP_LAST_N:]
    to_summarize = non_system[:-KEEP_LAST_N]

    # Load existing rolling summary from Redis
    existing_summary = load_stm_summary(thread_id)
    if existing_summary:
        # Prepend so the new summary absorbs the previous one
        to_summarize = [
            AIMessage(content=f"[Prior summary: {existing_summary}]")
        ] + to_summarize

    new_summary = _summarize_messages(to_summarize, llm)

    # Persist to Redis (resets the 2h TTL)
    save_stm_summary(thread_id, new_summary)

    trimmed = system_msgs + [
        AIMessage(
            content=(
                "[Context summary — what happened before this point]\n"
                + new_summary
            )
        )
    ] + tail_msgs

    logger.info(
        f"[STM] Trimmed {total_tokens} → {_count_tokens(trimmed)} tokens "
        f"| thread={thread_id}"
    )
    return trimmed


# ══════════════════════════════════════════════════════════════════════════════
#  EPISODIC INDEX SETUP
# ══════════════════════════════════════════════════════════════════════════════

def ensure_episodic_index() -> None:
    """
    Creates the RedisVL vector search index on episodic:* keys if it doesn't exist.
    Called at the start of store_episodic_memory() and retrieve_ltm().
    """
    client = get_sync_client()
    try:
        client.ft(EPISODIC_INDEX_NAME).info()
    except Exception:
        index = SearchIndex.from_dict(EPISODIC_INDEX_SCHEMA)
        index.connect(redis_url=REDIS_URL)
        index.create(overwrite=False)
        logger.info("[LTM] Episodic vector index created")


# ══════════════════════════════════════════════════════════════════════════════
#  LTM WRITE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def store_episodic_memory(
    student_id:      str,
    topic:           str,
    difficulty:      str,
    problem_summary: str,
    final_answer:    str,
    outcome:         str,
    solve_attempts:  int,
) -> str:
    """
    Stores one solved-problem record in Redis as a JSON doc + vector embedding.

    What gets stored (Redis JSON key: episodic:<student_id>:<ep_id>):
        student_id      : hashed student identifier
        episode_id      : timestamp-based unique ID
        topic           : "calculus" | "probability" | etc.
        difficulty      : "easy" | "medium" | "hard"
        problem_summary : first 200 chars of the problem text
        final_answer    : "π/4" | "x = 3" | etc.
        outcome         : "correct" | "incorrect" | "hitl"
        solve_attempts  : how many solver retries were needed (1-3)
        timestamp       : Unix time of storage
        access_count    : starts at 0, incremented each time this episode is retrieved
        decay_score     : starts at 1.0, computed by forgetting curve on retrieval
        embedding       : 1024-dim Cohere vector of "{topic} {difficulty} {summary}"

    The embedding encodes WHAT THE PROBLEM WAS ABOUT so that future similar
    problems find this episode via vector similarity search.

    TTL: EPISODIC_TTL (90 days). Decay pruning can remove earlier.
    Returns the episode_id.
    """
    ensure_episodic_index()
    client    = get_sync_client()
    ep_id     = f"{int(time.time() * 1000)}"
    now       = time.time()

    # Embed "{topic} {difficulty} {problem_summary}" — one API call
    embedding = _embed_texts(
        [f"{topic} {difficulty} {problem_summary}"],
        EMBED_INPUT_TYPE_DOC,
    )
    # as_buffer=True → bytes for RedisVL storage; we convert the numpy row
    emb_bytes = embedding[0].astype(np.float32).tobytes()

    doc = {
        "student_id":      student_id,
        "episode_id":      ep_id,
        "topic":           topic,
        "difficulty":      difficulty,
        "problem_summary": problem_summary[:200],
        "final_answer":    final_answer[:100],
        "outcome":         outcome,
        "solve_attempts":  solve_attempts,
        "timestamp":       now,
        "access_count":    0,
        "decay_score":     1.0,
        "embedding":       emb_bytes.hex(),  # hex-encode bytes for JSON storage
    }
    key = episodic_key(student_id, ep_id)
    client.json().set(key, "$", doc)
    client.expire(key, EPISODIC_TTL)

    logger.info(
        f"[LTM] Episodic stored | student={student_id} "
        f"topic={topic} outcome={outcome}"
    )
    return ep_id


def update_semantic_memory(
    student_id:      str,
    topic:           str,
    outcome:         str,
    mistake_pattern: str | None = None,
) -> None:
    """
    Maintains a running profile of this student's topic strengths and weaknesses.

    Redis JSON key: semantic:<student_id>
    No TTL — this is permanent student profile data.

    Schema:
        weak_topics      : {topic: fail_count}
        strong_topics    : {topic: success_count}
        mistake_patterns : [{pattern, topic, count}]

    Correct answer  → strong_topics[topic] += 1, weak_topics[topic] -= 1 (floor 0)
    Wrong answer    → weak_topics[topic] += 1
    mistake_pattern → if provided, logged and deduplicated by (pattern, topic)

    Used by format_ltm_for_solver to warn the solver:
        "WEAK AREA — struggled with calculus 3 times. Be especially clear."
    """
    client = get_sync_client()
    key    = semantic_key(student_id)
    ex     = client.json().get(key, "$")
    doc    = ex[0] if ex else {
        "student_id":      student_id,
        "weak_topics":     {},
        "strong_topics":   {},
        "mistake_patterns": [],
        "last_updated":    time.time(),
    }

    if outcome == "correct":
        doc["strong_topics"][topic] = doc["strong_topics"].get(topic, 0) + 1
        doc["weak_topics"][topic]   = max(0, doc["weak_topics"].get(topic, 0) - 1)
    else:
        doc["weak_topics"][topic] = doc["weak_topics"].get(topic, 0) + 1

    if mistake_pattern:
        found = False
        for entry in doc["mistake_patterns"]:
            if entry["pattern"] == mistake_pattern and entry["topic"] == topic:
                entry["count"] += 1
                found = True
                break
        if not found:
            doc["mistake_patterns"].append({
                "pattern": mistake_pattern, "topic": topic, "count": 1
            })

    doc["last_updated"] = time.time()
    client.json().set(key, "$", doc)


def update_procedural_memory(
    student_id: str,
    topic:      str,
    strategy:   str,
    success:    bool,
    attempts:   int,
) -> None:
    """
    Tracks which solving strategies work for this student on which topics.

    Redis JSON key: procedural:<student_id>
    No TTL — permanent student profile data.

    Schema:
        strategy_success: {
            topic: {
                strategy_name: {
                    success_count, total_count, attempts_sum,
                    success_rate (computed), attempts_avg (computed)
                }
            }
        }

    Used by retrieve_ltm to find the best strategy for the current topic:
        best = max(strategies, key=lambda k: (success_rate, -attempts_avg))
    Then injected into solver prompt:
        "STUDENT HISTORY — for calculus this student responds best to:
         integration by substitution (avg 1.6 attempts)."
    """
    client = get_sync_client()
    key    = procedural_key(student_id)
    ex     = client.json().get(key, "$")
    doc    = ex[0] if ex else {
        "student_id":      student_id,
        "strategy_success": {},
        "last_updated":    time.time(),
    }

    doc["strategy_success"].setdefault(topic, {}).setdefault(strategy, {
        "success_count": 0,
        "total_count":   0,
        "attempts_sum":  0,
    })
    e = doc["strategy_success"][topic][strategy]
    e["total_count"]  += 1
    e["attempts_sum"] += attempts
    if success:
        e["success_count"] += 1
    e["success_rate"] = round(e["success_count"] / e["total_count"], 2)
    e["attempts_avg"] = round(e["attempts_sum"]  / e["total_count"], 2)

    doc["last_updated"] = time.time()
    client.json().set(key, "$", doc)


# ══════════════════════════════════════════════════════════════════════════════
#  LTM READ
#  Called from _retrieve_ltm_node (graph.py) → memory_manager_node("retrieve")
#  → runs BEFORE parser_agent so the solver always has student context.
# ══════════════════════════════════════════════════════════════════════════════

def retrieve_ltm(student_id: str, problem_text: str, topic: str) -> dict:
    """
    Fetches all LTM context for the current problem.
    Returns a single dict written to state["ltm_context"].

    Three independent lookups (any can fail gracefully):

    1. VECTOR SEARCH (episodic)
       Embeds "{topic} {problem_text[:200]}" via Cohere.
       Queries the HNSW index for TOP_K_EPISODES (3) most similar past problems
       for this student.
       Also increments access_count and refreshes decay_score for retrieved episodes.

    2. SEMANTIC LOOKUP
       Reads weak_topics, strong_topics, mistake_patterns from semantic:<student_id>.

    3. PROCEDURAL LOOKUP
       Reads strategy_success from procedural:<student_id>.
       Picks the best strategy for the current topic (highest success_rate,
       lowest attempts_avg as tiebreaker).

    The returned dict is formatted by format_ltm_for_solver() into a text block
    injected into the solver's system prompt.
    """
    ensure_episodic_index()
    client = get_sync_client()
    result: dict = {
        "similar_episodes": [],
        "weak_topics":      {},
        "strong_topics":    {},
        "mistake_patterns": [],
        "best_strategy":    None,
        "avg_attempts":     None,
    }

    # ── 1. Vector search (episodic) ────────────────────────────────────────────
    try:
        q_emb = _embed_texts(
            [f"{topic} {problem_text[:200]}"],
            EMBED_INPUT_TYPE_QUERY,
        )
        query = VectorQuery(
            vector           = q_emb[0].tobytes(),
            vector_field_name= "embedding",
            return_fields    = [
                "student_id", "episode_id", "topic", "difficulty",
                "problem_summary", "final_answer", "outcome", "solve_attempts",
            ],
            num_results      = TOP_K_EPISODES,
            filter_expression= f"@student_id:{{{student_id}}}",
        )
        index = SearchIndex.from_dict(EPISODIC_INDEX_SCHEMA)
        index.connect(redis_url=REDIS_URL)
        eps = index.query(query)

        for ep in eps:
            k = episodic_key(student_id, ep.get("episode_id", ""))
            if client.exists(k):
                client.json().numincrby(k, "$.access_count", 1)
                _refresh_decay_score(client, k)

        result["similar_episodes"] = [
            {
                f: ep.get(f)
                for f in [
                    "topic", "difficulty", "problem_summary",
                    "final_answer", "outcome", "solve_attempts",
                ]
            }
            for ep in eps
        ]
    except Exception as e:
        logger.warning(f"[LTM] Vector search failed: {e}")

    # ── 2. Semantic lookup ─────────────────────────────────────────────────────
    try:
        sem = client.json().get(semantic_key(student_id), "$")
        if sem:
            d = sem[0]
            result["weak_topics"]      = d.get("weak_topics", {})
            result["strong_topics"]    = d.get("strong_topics", {})
            result["mistake_patterns"] = d.get("mistake_patterns", [])
    except Exception as e:
        logger.warning(f"[LTM] Semantic fetch failed: {e}")

    # ── 3. Procedural lookup ───────────────────────────────────────────────────
    try:
        proc = client.json().get(procedural_key(student_id), "$")
        if proc:
            strategies = proc[0].get("strategy_success", {}).get(topic, {})
            if strategies:
                best = max(
                    strategies.items(),
                    key=lambda kv: (
                        kv[1].get("success_rate", 0),
                        -kv[1].get("attempts_avg", 99),
                    ),
                )
                result["best_strategy"] = best[0]
                result["avg_attempts"]  = best[1].get("attempts_avg")
    except Exception as e:
        logger.warning(f"[LTM] Procedural fetch failed: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MEMORY DECAY
# ══════════════════════════════════════════════════════════════════════════════

def _refresh_decay_score(client, key: str) -> None:
    """
    Recomputes and saves the decay_score for an episodic record.
    Called every time an episode is retrieved (so accessed memories decay slower).

    Formula:
        decay_score = e^(-days_old / 30) × log(1 + access_count + 1)
    The log(access_count) term is the "spaced repetition" effect —
    episodes that are retrieved often decay much more slowly.
    """
    try:
        doc = client.json().get(key, "$")
        if not doc:
            return
        doc       = doc[0]
        days_old  = (time.time() - doc.get("timestamp", time.time())) / 86400
        score     = math.exp(-days_old / 30) * math.log(1 + doc.get("access_count", 0) + 1)
        client.json().set(key, "$.decay_score", round(score, 4))
    except Exception as e:
        logger.warning(f"[LTM] Decay refresh failed: {e}")


def prune_stale_episodic(student_id: str | None = None) -> int:
    """
    Deletes episodic records with decay_score < DECAY_THRESHOLD (0.05)
    AND age > 30 days.
    Or call manually:
        prune_stale_episodic()           # all students    
        prune_stale_episodic("a3f9...")  # one student

    Returns count of pruned records.
    """
    ## right now we will be using the pruning manually only 
    client  = get_sync_client()
    pattern = f"episodic:{student_id}:*" if student_id else "episodic:*:*"
    cutoff  = time.time() - (30 * 86400)
    pruned  = 0
    for key in client.keys(pattern):
        try:
            doc = client.json().get(key, "$")
            if not doc:
                continue
            doc = doc[0]
            _refresh_decay_score(client, key)
            doc = client.json().get(key, "$")[0]
            if (
                doc.get("decay_score", 1.0) < DECAY_THRESHOLD
                and doc.get("timestamp", 0) < cutoff
            ):
                client.delete(key)
                pruned += 1
        except Exception as e:
            logger.warning(f"[LTM] Prune error {key}: {e}")
    logger.info(f"[LTM] Pruned {pruned} stale entries")
    return pruned


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH NODE — the actual node function wired into graph.py
# ══════════════════════════════════════════════════════════════════════════════

def memory_manager_node(state: dict) -> dict:
    """
    Router node — reads state["ltm_mode"] and dispatches to retrieve or store.

    RETRIEVE mode (runs before parser_agent via _retrieve_ltm_node in graph.py):
        Reads: state["parsed_data"]["problem_text"], state["parsed_data"]["topic"]
               state["student_id"]
        Calls: retrieve_ltm()
        Writes: state["ltm_context"]

    STORE mode (runs after student satisfaction via _store_ltm_node in graph.py):
        Reads: parsed_data, solution_plan, solver_output, verifier_output,
               solve_iterations, student_id, thread_id
        Calls: store_episodic_memory()
               update_semantic_memory()
               update_procedural_memory()
               update_thread_meta()
               increment_problems_solved() (if correct)
        Writes: state["ltm_stored"] = True
    """
    ltm_mode   = state.get("ltm_mode", "retrieve")
    student_id = state.get("student_id") or "anonymous"
    thread_id  = state.get("thread_id") or ""

    if ltm_mode == "retrieve":
        parsed       = state.get("parsed_data") or {}
        problem_text = parsed.get("problem_text") or state.get("raw_text") or ""
        topic        = parsed.get("topic") or ""
        ltm_context  = retrieve_ltm(
            student_id   = student_id,
            problem_text = problem_text,
            topic        = topic,
        )
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

        outcome = (
            "correct"   if verifier_status == "correct"      else
            "hitl"      if verifier_status == "needs_human"  else
            "incorrect"
        )
        mistake = (
            suggested_fix[:80].strip()
            if (suggested_fix and outcome != "correct")
            else None
        )

        store_episodic_memory(
            student_id      = student_id,
            topic           = topic,
            difficulty      = difficulty,
            problem_summary = problem_text[:200],
            final_answer    = final_answer,
            outcome         = outcome,
            solve_attempts  = solve_attempts,
        )
        update_semantic_memory(
            student_id      = student_id,
            topic           = topic,
            outcome         = outcome,
            mistake_pattern = mistake,
        )
        update_procedural_memory(
            student_id = student_id,
            topic      = topic,
            strategy   = strategy,
            success    = (outcome == "correct"),
            attempts   = solve_attempts,
        )
        update_thread_meta(
            thread_id       = thread_id,
            problem_summary = problem_text[:120],
            topic           = topic,
            outcome         = outcome,
        )
        if outcome == "correct":
            increment_problems_solved(student_id)

        logger.info(
            f"[MemMgr] LTM stored | student={student_id} outcome={outcome}"
        )
        return {"ltm_stored": True}

    return {}


# ══════════════════════════════════════════════════════════════════════════════
#  LTM CONTEXT FORMATTERS
#  Convert raw ltm_context dict into prompt-ready text strings.
# ══════════════════════════════════════════════════════════════════════════════

def format_ltm_for_solver(ltm_context: dict, topic: str) -> str:
    """
    USED: called in solver_agent (solver.py) to inject LTM into the system prompt.

    Converts ltm_context into 1-4 sentences of student-specific context.
    Each sentence starts with a CAPS label so the LLM knows it's metadata.

    Example output:
        STUDENT HISTORY — for calculus this student responds best to:
          integration by substitution (avg 1.6 attempts).
        WEAK AREA — struggled with calculus 3 times. Be especially clear on each step.
        KNOWN MISTAKES in calculus: forgot to change limits after substitution.
        SIMILAR PAST PROBLEM — calculus (hard), outcome=correct, answer=π/4.
    """
    if not ltm_context:
        return ""
    lines = []

    best = ltm_context.get("best_strategy")
    avg  = ltm_context.get("avg_attempts")
    if best:
        avg_str = f"{avg:.1f}" if avg is not None else "?"
        lines.append(
            f"STUDENT HISTORY — for {topic} this student responds best to: "
            f"{best} (avg {avg_str} attempts)."
        )

    weak = ltm_context.get("weak_topics", {})
    if topic in weak and weak[topic] > 1:
        lines.append(
            f"WEAK AREA — struggled with {topic} {weak[topic]} times. "
            "Be especially clear on each step."
        )

    patterns = [
        p for p in ltm_context.get("mistake_patterns", [])
        if p.get("topic") == topic and p.get("count", 0) > 1
    ]
    if patterns:
        lines.append(
            f"KNOWN MISTAKES in {topic}: "
            + "; ".join(p["pattern"] for p in patterns[:2]) + "."
        )

    eps = ltm_context.get("similar_episodes", [])
    if eps:
        ep = eps[0]
        lines.append(
            f"SIMILAR PAST PROBLEM — {ep['topic']} ({ep['difficulty']}), "
            f"outcome={ep['outcome']}, answer={ep['final_answer']}."
        )

    return "\n".join(lines)


def format_ltm_for_explainer(ltm_context: dict, topic: str) -> str:
    """
    NOT YET WIRED: defined here for future use in explainer_agent.

    Would personalise the explanation based on known weak spots:
        PERSONALISATION — this student struggles with calculus.
          Add extra intuition-building.
        COMMON MISTAKES THIS STUDENT MAKES in calculus:
          forgot to change limits after substitution; sign errors in integration.
    """
    if not ltm_context:
        return ""
    lines = []

    weak = ltm_context.get("weak_topics", {})
    if topic in weak and weak[topic] > 0:
        lines.append(
            f"PERSONALISATION — this student struggles with {topic}. "
            "Add extra intuition-building."
        )

    patterns = [
        p for p in ltm_context.get("mistake_patterns", [])
        if p.get("topic") == topic
    ]
    if patterns:
        lines.append(
            f"COMMON MISTAKES THIS STUDENT MAKES in {topic}: "
            + "; ".join(p["pattern"] for p in patterns[:3]) + ". "
            "Address these in common_mistakes field."
        )

    return "\n".join(lines)