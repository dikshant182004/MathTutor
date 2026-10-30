"""Postgres persistence for durable, student-scoped conversation history and analytics.

Redis remains responsible for LangGraph checkpoints and short-lived coordination.
This module stores the durable product history independently of checkpoint expiry.
"""
from __future__ import annotations

import os
import threading
import uuid
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

_MIGRATION_LOCK = threading.Lock()
_MIGRATED_DSN: str | None = None
_MIGRATION_PATH = Path(__file__).with_name("migrations") / "001_conversation_history_analytics.sql"


def database_url() -> str:
    value = os.getenv("DATABASE_URL", "").strip()
    if not value:
        raise RuntimeError("DATABASE_URL is required for durable conversation history.")
    return value


class PostgresStore:
    """Small connection-per-operation repository; safe across API worker processes."""

    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or database_url()

    def connect(self):
        return psycopg.connect(self.dsn, row_factory=dict_row, connect_timeout=5)

    def migrate(self) -> None:
        global _MIGRATED_DSN
        with _MIGRATION_LOCK:
            if _MIGRATED_DSN == self.dsn:
                return
            sql = _MIGRATION_PATH.read_text(encoding="utf-8")
            with self.connect() as conn:
                conn.execute(sql, prepare=False)
            _MIGRATED_DSN = self.dsn

    def ensure_student(self, student_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO students(student_id) VALUES (%s) ON CONFLICT DO NOTHING",
                (student_id,),
            )

    def create_thread(self, student_id: str, thread_id: str, title: str = "", *, created_at=None) -> dict:
        with self.connect() as conn:
            conn.execute("INSERT INTO students(student_id) VALUES (%s) ON CONFLICT DO NOTHING", (student_id,))
            row = conn.execute(
                """INSERT INTO threads(thread_id, student_id, title, created_at, last_active)
                   VALUES (%s, %s, %s, COALESCE(%s, now()), COALESCE(%s, now()))
                   RETURNING thread_id, student_id, title, topic, outcome, created_at, last_active""",
                (thread_id, student_id, title[:120], created_at, created_at),
            ).fetchone()
        return row

    def list_threads(self, student_id: str, limit: int = 20) -> list[dict]:
        with self.connect() as conn:
            return conn.execute(
                """SELECT thread_id, title, topic, outcome, created_at, last_active
                   FROM threads WHERE student_id = %s
                   ORDER BY last_active DESC LIMIT %s""",
                (student_id, min(max(int(limit), 1), 100)),
            ).fetchall()

    def get_thread(self, student_id: str, thread_id: str) -> dict | None:
        with self.connect() as conn:
            return conn.execute(
                """SELECT thread_id, student_id, title, topic, outcome, created_at, last_active
                   FROM threads WHERE student_id = %s AND thread_id = %s""",
                (student_id, thread_id),
            ).fetchone()

    def touch_thread(self, student_id: str, thread_id: str, *, topic: str | None = None, outcome: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE threads SET last_active = now(),
                   topic = COALESCE(%s, topic), outcome = COALESCE(%s, outcome)
                   WHERE student_id = %s AND thread_id = %s""",
                (topic, outcome, student_id, thread_id),
            )

    def delete_thread(self, student_id: str, thread_id: str) -> bool:
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM threads WHERE student_id = %s AND thread_id = %s",
                (student_id, thread_id),
            )
            return cur.rowcount > 0

    def append_message(self, student_id: str, thread_id: str, role: str, content: str, message_id: str | None = None) -> str:
        message_id = str(uuid.UUID(message_id)) if message_id else str(uuid.uuid4())
        if role not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported conversation message role.")
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO conversation_messages(message_id, thread_id, student_id, role, content)
                   VALUES (%s, %s, %s, %s, %s) ON CONFLICT (message_id) DO NOTHING""",
                (message_id, thread_id, student_id, role, content),
            )
            conn.execute(
                "UPDATE threads SET last_active = now() WHERE student_id = %s AND thread_id = %s",
                (student_id, thread_id),
            )
        return message_id

    def list_messages(self, student_id: str, thread_id: str, limit: int = 500) -> list[dict]:
        with self.connect() as conn:
            return conn.execute(
                """SELECT message_id, role, content, created_at
                   FROM conversation_messages
                   WHERE student_id = %s AND thread_id = %s
                   ORDER BY created_at, message_id LIMIT %s""",
                (student_id, thread_id, min(max(int(limit), 1), 1000)),
            ).fetchall()

    def start_run(self, student_id: str, thread_id: str, run_id: str) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO tutoring_runs(run_id, thread_id, student_id, status)
                   VALUES (%s, %s, %s, 'running')""",
                (run_id, thread_id, student_id),
            )

    def finish_run(self, student_id: str, thread_id: str, run_id: str, *, status: str,
                   route: str | None = None, complexity: str | None = None,
                   llm_calls: int = 0, tool_calls: int = 0, duration_ms: int | None = None,
                   error_code: str | None = None) -> None:
        if status not in {"completed", "interrupted", "failed"}:
            raise ValueError("Invalid run status.")
        with self.connect() as conn:
            conn.execute(
                """UPDATE tutoring_runs SET status = %s, route = %s, complexity = %s,
                   llm_calls = %s, tool_calls = %s, duration_ms = %s, error_code = %s,
                   completed_at = now()
                   WHERE run_id = %s AND student_id = %s AND thread_id = %s""",
                (status, route, complexity, max(0, int(llm_calls)), max(0, int(tool_calls)),
                 duration_ms, error_code, run_id, student_id, thread_id),
            )

    def analytics_summary(self, student_id: str) -> dict:
        with self.connect() as conn:
            row = conn.execute(
                """SELECT
                     (SELECT count(*) FROM threads WHERE student_id = %s) AS total_threads,
                     (SELECT count(*) FROM conversation_messages WHERE student_id = %s) AS total_messages,
                     (SELECT count(*) FROM tutoring_runs WHERE student_id = %s) AS total_runs,
                     (SELECT count(*) FROM tutoring_runs WHERE student_id = %s AND status = 'completed') AS completed_runs,
                     (SELECT count(*) FROM tutoring_runs WHERE student_id = %s AND status = 'interrupted') AS interrupted_runs,
                     (SELECT count(*) FROM tutoring_runs WHERE student_id = %s AND status = 'failed') AS failed_runs,
                     (SELECT COALESCE(sum(llm_calls), 0) FROM tutoring_runs WHERE student_id = %s) AS llm_calls,
                     (SELECT COALESCE(sum(tool_calls), 0) FROM tutoring_runs WHERE student_id = %s) AS tool_calls""",
                (student_id,) * 8,
            ).fetchone()
        return {key: int(value or 0) for key, value in row.items()}


def get_postgres_store() -> PostgresStore:
    store = PostgresStore()
    store.migrate()
    return store
