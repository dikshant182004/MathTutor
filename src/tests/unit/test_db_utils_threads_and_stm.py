import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "backend" / "agents" / "utils" / "db_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("db_utils_threads_module", _MODULE_PATH)
db_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(db_utils)


class _FakeRedis:
    """In-memory stand-in for a Redis client. Zero network calls."""

    def __init__(self):
        self.hashes  = {}
        self.strings = {}
        self.zsets   = {}
        self.expiry  = {}

    def exists(self, key):
        return key in self.hashes

    def hset(self, key, mapping):
        current = self.hashes.get(key, {})
        current.update(mapping)
        self.hashes[key] = current

    def hgetall(self, key):
        return self.hashes.get(key, {})

    def expire(self, key, ttl):
        self.expiry[key] = ttl

    def zadd(self, key, mapping, xx=False):
        bucket = self.zsets.get(key, {})
        for member, score in mapping.items():
            if xx and member not in bucket:
                continue
            bucket[member] = score
        self.zsets[key] = bucket

    def zrevrange(self, key, start, stop):
        bucket  = self.zsets.get(key, {})
        ordered = sorted(bucket.items(), key=lambda kv: kv[1], reverse=True)
        members = [m for m, _ in ordered]
        return members[start : stop + 1]

    def setex(self, key, ttl, value):
        self.expiry[key]  = ttl
        self.strings[key] = value

    def get(self, key):
        return self.strings.get(key)


# ── thread round-trip ─────────────────────────────────────────────────────────

@pytest.mark.unit
def test_create_thread_prefixes_student_id(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    tid = db_utils.create_thread("student1")
    assert tid.startswith("student1:")


@pytest.mark.unit
def test_update_thread_meta_truncates_long_summary(monkeypatch):
    """Summary must be truncated to the module's own SUMMARY_MAX constant."""
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid = "studentx"
    tid = db_utils.create_thread(sid)
    db_utils.update_thread_meta(tid, "a" * 300, "algebra", "correct")

    meta        = fake.hgetall(db_utils.thread_meta_key(tid))
    max_len     = getattr(db_utils, "SUMMARY_MAX", 120)   # fall back to 120 if constant renamed
    assert len(meta["problem_summary"]) <= max_len
    assert meta["topic"]   == "algebra"
    assert meta["outcome"] == "correct"


@pytest.mark.unit
def test_get_thread_history_returns_created_thread(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid = "studentx"
    tid = db_utils.create_thread(sid)
    db_utils.update_thread_meta(tid, "short summary", "calculus", "correct")

    history = db_utils.get_thread_history(sid)
    assert len(history) == 1
    assert history[0]["thread_id"] == tid


@pytest.mark.unit
def test_multiple_threads_appear_in_history(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid  = "student_multi"
    tid1 = db_utils.create_thread(sid)
    tid2 = db_utils.create_thread(sid)
    db_utils.update_thread_meta(tid1, "problem A", "algebra",  "correct")
    db_utils.update_thread_meta(tid2, "problem B", "calculus", "incorrect")

    history = db_utils.get_thread_history(sid)
    ids     = [h["thread_id"] for h in history]
    assert tid1 in ids
    assert tid2 in ids


# ── STM summary ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_stm_summary_save_and_load(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    db_utils.save_stm_summary("t1", "rolling summary")
    loaded = db_utils.load_stm_summary("t1")
    assert loaded == "rolling summary"


@pytest.mark.unit
def test_stm_summary_ttl_is_set(monkeypatch):
    """save_stm_summary must attach the module-level TTL to the key."""
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    db_utils.save_stm_summary("t1", "some summary")
    key = db_utils.stm_summary_key("t1")
    assert fake.expiry[key] == db_utils.STM_SUMMARY_TTL


@pytest.mark.unit
def test_stm_summary_overwrites_previous_value(monkeypatch):
    """Saving twice on the same thread should replace the first value."""
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    db_utils.save_stm_summary("t2", "first")
    db_utils.save_stm_summary("t2", "second")
    assert db_utils.load_stm_summary("t2") == "second"


@pytest.mark.unit
def test_stm_summary_ttl_refreshed_on_overwrite(monkeypatch):
    """TTL must be re-applied on every save, not just the first."""
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    db_utils.save_stm_summary("t3", "v1")
    db_utils.save_stm_summary("t3", "v2")
    key = db_utils.stm_summary_key("t3")
    assert fake.expiry[key] == db_utils.STM_SUMMARY_TTL