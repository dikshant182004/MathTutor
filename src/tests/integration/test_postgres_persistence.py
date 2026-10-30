from __future__ import annotations

import uuid

import pytest

from backend.api.persistence import PostgresStore


@pytest.fixture(scope="module")
def store():
    repository = PostgresStore()
    repository.migrate()
    yield repository


def test_postgres_persists_student_scoped_threads_messages_and_analytics(store):
    student_a = uuid.uuid4().hex[:16]
    student_b = uuid.uuid4().hex[:16]
    thread_a = f"{student_a}:1000000000001"
    thread_b = f"{student_b}:1000000000002"
    run_id = str(uuid.uuid4())
    try:
        store.create_thread(student_a, thread_a, "Algebra practice")
        store.create_thread(student_b, thread_b, "Geometry practice")
        store.append_message(student_a, thread_a, "user", "Solve x + 2 = 5")
        store.append_message(student_a, thread_a, "assistant", "x = 3")
        store.start_run(student_a, thread_a, run_id)
        store.finish_run(
            student_a, thread_a, run_id, status="completed",
            route="solver_agent", complexity="trivial",
            llm_calls=2, tool_calls=1, duration_ms=125,
        )

        assert [row["thread_id"] for row in store.list_threads(student_a)] == [thread_a]
        assert [row["thread_id"] for row in store.list_threads(student_b)] == [thread_b]
        messages = store.list_messages(student_a, thread_a)
        assert [(row["role"], row["content"]) for row in messages] == [
            ("user", "Solve x + 2 = 5"),
            ("assistant", "x = 3"),
        ]
        assert store.list_messages(student_b, thread_a) == []
        assert store.analytics_summary(student_a) == {
            "total_threads": 1, "total_messages": 2, "total_runs": 1,
            "completed_runs": 1, "interrupted_runs": 0, "failed_runs": 0,
            "llm_calls": 2, "tool_calls": 1,
        }
        assert store.analytics_summary(student_b)["total_threads"] == 1
        assert store.analytics_summary(student_b)["total_messages"] == 0
    finally:
        store.delete_thread(student_a, thread_a)
        store.delete_thread(student_b, thread_b)
