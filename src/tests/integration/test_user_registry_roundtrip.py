"""
tests/integration/test_user_registry_roundtrip.py
===================================================
Integration tests: create a user → fetch profile → update name.
All Redis calls go through _FakeRedis — no real Redis connection.
"""
import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "backend" / "agents" / "utils" / "db_utils.py"
)
_SPEC    = importlib.util.spec_from_file_location("integration_db_utils_module", _MODULE_PATH)
db_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(db_utils)


class _FakeRedis:
    def __init__(self):
        self.hashes = {}

    def exists(self, key):
        return key in self.hashes

    def hset(self, key, mapping):
        existing = self.hashes.get(key, {})
        existing.update(mapping)
        self.hashes[key] = existing

    def hgetall(self, key):
        return self.hashes.get(key, {})


@pytest.mark.integration
def test_user_create_then_profile_fetch(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid     = db_utils.get_or_create_user("student@example.com", "Student")
    profile = db_utils.get_user_profile(sid)

    assert profile is not None
    assert profile["student_id"] == sid
    assert profile["email"]      == "student@example.com"


@pytest.mark.integration
def test_user_rename_reflected_in_profile(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    db_utils.get_or_create_user("student@example.com", "Student")
    db_utils.get_or_create_user("student@example.com", "Student Renamed")

    sid     = db_utils.student_id_from_email("student@example.com")
    updated = db_utils.get_user_profile(sid)
    assert updated["display_name"] == "Student Renamed"


@pytest.mark.integration
def test_two_different_users_have_separate_profiles(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid_a = db_utils.get_or_create_user("alice@example.com", "Alice")
    sid_b = db_utils.get_or_create_user("bob@example.com",   "Bob")

    profile_a = db_utils.get_user_profile(sid_a)
    profile_b = db_utils.get_user_profile(sid_b)

    assert profile_a["email"] == "alice@example.com"
    assert profile_b["email"] == "bob@example.com"
    assert sid_a != sid_b