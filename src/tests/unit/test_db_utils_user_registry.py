import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "backend" / "agents" / "utils" / "db_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("db_utils_registry_module", _MODULE_PATH)
db_utils = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
_SPEC.loader.exec_module(db_utils)


class _FakeRedis:
    def __init__(self):
        self.hashes = {}

    def exists(self, key):
        return key in self.hashes

    def hset(self, key, mapping):
        current = self.hashes.get(key, {})
        current.update(mapping)
        self.hashes[key] = current

    def hgetall(self, key):
        return self.hashes.get(key, {})


# ── student_id_from_email ─────────────────────────────────────────────────────

@pytest.mark.unit
def test_student_id_from_email_is_deterministic(monkeypatch):
    """Same email must always produce the same student ID."""
    id1 = db_utils.student_id_from_email("alice@example.com")
    id2 = db_utils.student_id_from_email("alice@example.com")
    assert id1 == id2


@pytest.mark.unit
def test_student_id_from_email_is_unique_per_email(monkeypatch):
    """Different emails must produce different student IDs."""
    id_alice = db_utils.student_id_from_email("alice@example.com")
    id_bob   = db_utils.student_id_from_email("bob@example.com")
    assert id_alice != id_bob


# ── get_or_create_user ────────────────────────────────────────────────────────

@pytest.mark.unit
def test_get_or_create_user_creates_new_user(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid     = db_utils.get_or_create_user("alice@example.com", "Alice")
    profile = fake.hgetall(db_utils.user_key(sid))

    assert sid == db_utils.student_id_from_email("alice@example.com")
    assert profile["email"]        == "alice@example.com"
    assert profile["display_name"] == "Alice"
    assert profile["total_problems_solved"] == 0


@pytest.mark.unit
def test_get_or_create_user_updates_display_name_on_second_call(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid    = db_utils.get_or_create_user("bob@example.com", "Bob")
    before = dict(fake.hgetall(db_utils.user_key(sid)))

    db_utils.get_or_create_user("bob@example.com", "Bobby")
    after  = fake.hgetall(db_utils.user_key(sid))

    assert after["display_name"] == "Bobby"
    assert float(after["last_login"]) >= float(before["last_login"])


@pytest.mark.unit
def test_get_or_create_user_does_not_reset_problems_solved(monkeypatch):
    """Re-login must not zero out accumulated problem count."""
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid = db_utils.get_or_create_user("carol@example.com", "Carol")
    # Manually bump the counter as if problems were solved
    fake.hashes[db_utils.user_key(sid)]["total_problems_solved"] = 5

    db_utils.get_or_create_user("carol@example.com", "Carol")
    after = fake.hgetall(db_utils.user_key(sid))
    assert int(after["total_problems_solved"]) == 5


@pytest.mark.unit
def test_get_or_create_user_returns_same_sid_on_repeat(monkeypatch):
    fake = _FakeRedis()
    monkeypatch.setattr(db_utils, "get_sync_client", lambda: fake)

    sid1 = db_utils.get_or_create_user("dave@example.com", "Dave")
    sid2 = db_utils.get_or_create_user("dave@example.com", "Dave")
    assert sid1 == sid2