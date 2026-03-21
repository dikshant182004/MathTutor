import os

REDIS_URL = os.getenv("REDIS_URL", "redis://:jee_secret@localhost:6379")

EMBED_DIM = 1024

# ── STM trimming thresholds ────────────────────────────────────────────────────
# Tokens are counted with tiktoken (gpt-4o encoding) as a fast local proxy.
# LLaMA's tokenizer would differ by ~5-10% — acceptable for a soft threshold.
TOKEN_LIMIT    = 8_000   # trim when messages exceed this many tokens
TOKEN_TARGET   = 5_000   # aim for approximately this after trimming
KEEP_LAST_N    = 6       # always keep last N messages verbatim (never summarised)
TIKTOKEN_MODEL = "gpt-4o"  # encoding used for counting only — no OpenAI call made

# ── Redis TTLs (all in seconds) ────────────────────────────────────────────────
STM_SUMMARY_TTL = 2  * 60 * 60    # 2 hours  — running message summary per thread
THREAD_TTL      = 7  * 24 * 3600  # 7 days   — thread metadata card in sidebar
EPISODIC_TTL    = 30 * 24 * 3600  # 90 days  — one record per solved problem

# ── LTM retrieval settings ────────────────────────────────────────────────────
DECAY_THRESHOLD   = 0.05   # episodes with decay_score below this get pruned
TOP_K_EPISODES    = 3      # how many past similar problems to retrieve
MAX_THREADS_SHOWN = 20     # max threads shown in sidebar history

# ── Episodic vector index name (RedisVL) ─────────────────────────────────────
EPISODIC_INDEX_NAME = "idx:episodic"

# ── Episodic vector index schema (used by ensure_episodic_index) ──────────────
EPISODIC_INDEX_SCHEMA = {
    "index": {
        "name":         EPISODIC_INDEX_NAME,
        "prefix":       "episodic:",
        "storage_type": "json",
    },
    "fields": [
        {"name": "student_id",  "type": "tag",     "path": "$.student_id"},
        {"name": "topic",       "type": "tag",     "path": "$.topic"},
        {"name": "difficulty",  "type": "tag",     "path": "$.difficulty"},
        {"name": "outcome",     "type": "tag",     "path": "$.outcome"},
        {"name": "timestamp",   "type": "numeric", "path": "$.timestamp"},
        {"name": "decay_score", "type": "numeric", "path": "$.decay_score"},
        {
            "name": "embedding",
            "type": "vector",
            "path": "$.embedding",
            "attrs": {
                "algorithm":       "HNSW",
                "datatype":        "FLOAT32",
                "dims":            EMBED_DIM,
                "distance_metric": "COSINE",
            },
        },
    ],
}