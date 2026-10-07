from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from threading import RLock
from typing import Any


class JsonRepository:
    """Small local persistence adapter for development/evaluation.

    Production deployments can replace this adapter with Redis/Postgres without
    changing student-model or evaluation logic.
    """

    def __init__(self, path: str = ".data/math_tutor.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        with self._lock:
            if not self.path.exists():
                return None
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data.get(namespace, {}).get(key)

    def put(self, namespace: str, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            data = {}
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
            data.setdefault(namespace, {})[key] = value
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
