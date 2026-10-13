from __future__ import annotations

import fnmatch
import json
import os

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from backend.agents.state import AgentState
from backend.api.deps import get_current_student, get_graph_app, get_redis, issue_access_token
from backend.api.main import app


STUDENT_A = "aaaaaaaaaaaaaaaa"
STUDENT_B = "bbbbbbbbbbbbbbbb"


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.strings = {}
        self.sorted_sets = {}
        self.ttls = {}

    def ping(self):
        return True

    def exists(self, key):
        return int(key in self.hashes or key in self.strings)

    def hset(self, name, mapping=None, key=None, value=None):
        target = self.hashes.setdefault(name, {})
        if mapping:
            target.update({str(k): str(v) for k, v in mapping.items()})
        if key is not None:
            target[str(key)] = str(value)
        return 1

    def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    def hget(self, name, key):
        return self.hashes.get(name, {}).get(key)

    def hdel(self, name, *keys):
        target = self.hashes.get(name, {})
        removed = 0
        for key in keys:
            if key in target:
                del target[key]
                removed += 1
        return removed

    def zadd(self, name, mapping):
        target = self.sorted_sets.setdefault(name, {})
        target.update({str(k): float(v) for k, v in mapping.items()})
        return 1

    def zrevrange(self, name, start, end):
        ordered = sorted(self.sorted_sets.get(name, {}).items(), key=lambda pair: pair[1], reverse=True)
        stop = None if end == -1 else end + 1
        return [key for key, _ in ordered[start:stop]]

    def zrem(self, name, *members):
        target = self.sorted_sets.get(name, {})
        count = 0
        for member in members:
            if member in target:
                del target[member]
                count += 1
        return count

    def delete(self, *keys):
        count = 0
        for key in keys:
            existed = self.hashes.pop(key, None) is not None
            existed = self.strings.pop(key, None) is not None or existed
            self.sorted_sets.pop(key, None)
            self.ttls.pop(key, None)
            count += int(existed)
        return count

    def expire(self, name, seconds):
        self.ttls[name] = seconds
        return True

    def set(self, name, value, ex=None, nx=False):
        if nx and name in self.strings:
            return None
        self.strings[name] = value
        if ex is not None:
            self.ttls[name] = ex
        return True

    def get(self, name):
        return self.strings.get(name)

    def scan_iter(self, match="*"):
        keys = set(self.hashes) | set(self.strings) | set(self.sorted_sets)
        return iter(key for key in keys if fnmatch.fnmatch(key, match))


@pytest.fixture
def api_client(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "phase-d-test-secret-with-at-least-32-characters")
    redis = FakeRedis()
    counters = {"resumed": 0}

    def wait_for_human(state: AgentState):
        response = interrupt({"type": "clarification", "prompt": "Please confirm the answer."})
        counters["resumed"] += 1
        answer = str(response.get("answer", ""))
        return {"answer": answer, "final_response": answer}

    builder = StateGraph(AgentState)
    builder.add_node("wait_for_human", wait_for_human)
    builder.set_entry_point("wait_for_human")
    builder.add_edge("wait_for_human", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    app.dependency_overrides[get_redis] = lambda: redis
    app.dependency_overrides[get_graph_app] = lambda: graph
    try:
        with TestClient(app) as client:
            yield client, redis, counters
    finally:
        app.dependency_overrides.clear()


def bearer(student_id=STUDENT_A):
    token, _ = issue_access_token(student_id)
    return {"Authorization": f"Bearer {token}"}


def create_thread(client):
    response = client.post("/threads", headers=bearer(), json={})
    assert response.status_code == 201
    return response.json()["thread_id"]


def test_invalid_jwt_is_rejected(api_client):
    client, _, _ = api_client
    response = client.get("/models", headers={"Authorization": "Bearer not-a-valid-jwt"})
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_token"


def test_another_students_thread_is_rejected_before_checkpoint_access(api_client):
    client, _, _ = api_client
    foreign_thread = f"{STUDENT_B}:1730000000000"
    response = client.get(f"/threads/{foreign_thread}/messages", headers=bearer(STUDENT_A))
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "thread_not_found"


def test_expired_checkpoint_returns_410(api_client):
    client, redis, _ = api_client
    thread_id = create_thread(client)
    redis.hset(f"thread:{thread_id}:meta", mapping={"checkpoint_expires_at": "1"})
    response = client.post(
        f"/threads/{thread_id}/resume",
        headers={**bearer(), "Idempotency-Key": "expired-checkpoint"},
        json={"response": {"answer": "x=1"}},
    )
    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "checkpoint_expired"


def test_real_langgraph_interrupt_pauses_resumes_same_thread_and_duplicate_is_idempotent(api_client):
    client, _, counters = api_client
    thread_id = create_thread(client)

    first = client.post(
        f"/threads/{thread_id}/messages",
        headers=bearer(),
        json={"content": "Please solve this after clarification."},
    )
    assert first.status_code == 200
    assert "event: interrupt" in first.text
    assert "Please confirm the answer." in first.text
    assert counters["resumed"] == 0

    history = client.get(f"/threads/{thread_id}/messages", headers=bearer())
    assert history.status_code == 200
    pending = history.json()["pending_interrupt"]
    assert pending is not None
    assert pending["thread_id"] == thread_id

    headers = {**bearer(), "Idempotency-Key": "resume-once-01"}
    body = {"response": {"answer": "x = 1"}}
    resumed = client.post(f"/threads/{thread_id}/resume", headers=headers, json=body)
    assert resumed.status_code == 200
    assert "event: done" in resumed.text
    assert "x = 1" in resumed.text
    assert counters["resumed"] == 1

    duplicate = client.post(f"/threads/{thread_id}/resume", headers=headers, json=body)
    assert duplicate.status_code == 200
    assert duplicate.text == resumed.text
    assert counters["resumed"] == 1
