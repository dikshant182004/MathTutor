"""Regression tests for durable HITL resume history and analytics."""
from __future__ import annotations

import json
import asyncio
from types import SimpleNamespace

from backend.api.routes.hitl import _resume_stream


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.values = {}

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        return True

    def hdel(self, key, field):
        self.hashes.get(key, {}).pop(field, None)

    def set(self, key, value, **kwargs):
        self.values[key] = value
        return True

    def eval(self, script, numkeys, key, token):
        self.values.pop(key, None)
        return 1


class FakeStore:
    def __init__(self):
        self.messages = []
        self.runs = []

    def start_run(self, student_id, thread_id, run_id):
        self.runs.append(("start", student_id, thread_id, run_id))

    def append_message(self, student_id, thread_id, role, content):
        self.messages.append((student_id, thread_id, role, content))

    def finish_run(self, student_id, thread_id, run_id, **kwargs):
        self.runs.append(("finish", student_id, thread_id, run_id, kwargs))


class FakeGraph:
    async def astream(self, *args, **kwargs):
        if False:
            yield None

    def get_state(self, config):
        return SimpleNamespace(
            values={"final_response": "The answer is 42.", "llm_calls_used": 2, "tool_calls_used": 1},
            tasks=(),
        )


def test_resume_persists_human_and_final_messages_and_closes_run():
    student_id = "a" * 16
    thread_id = f"{student_id}:123"
    redis = FakeRedis()
    store = FakeStore()

    async def collect():
        return [
            item async for item in _resume_stream(
                FakeGraph(),
                redis,
                store,
                student_id,
                thread_id,
                {"corrected_text": "I meant x squared."},
                "run-resume-1",
                f"idem:{thread_id}:key-1",
                f"resume-lock:{thread_id}",
                "lock-token",
            )
        ]

    events = asyncio.run(collect())

    assert any("event: solution" in event for event in events)
    assert any("event: done" in event for event in events)
    assert (student_id, thread_id, "user", "I meant x squared.") in store.messages
    assert (student_id, thread_id, "assistant", "The answer is 42.") in store.messages
    assert store.runs[0] == ("start", student_id, thread_id, "run-resume-1")
    finish = store.runs[-1]
    assert finish[0] == "finish"
    assert finish[3] == "run-resume-1"
    assert finish[4]["status"] == "completed"
    cached = json.loads(redis.values[f"idem:{thread_id}:key-1"])
    assert cached["status"] == "complete"
    assert any("event: done" in event for event in cached["events"])
