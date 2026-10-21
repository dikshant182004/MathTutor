from __future__ import annotations

from backend.agents.utils import db_utils


def test_checkpointer_configures_explicit_two_hour_ttl(monkeypatch):
    captured = {}

    class FakeRedisSaver:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.setup_called = False

        def setup(self):
            self.setup_called = True

    monkeypatch.setattr(db_utils, "RedisSaver", FakeRedisSaver)
    monkeypatch.setattr(db_utils, "REDIS_URL", "redis://example.invalid:6379")

    saver = db_utils.build_stm_checkpointer()

    assert captured["ttl"] == {"default_ttl": 120, "refresh_on_read": True}
    assert saver.setup_called is True
