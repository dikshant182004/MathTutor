from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def load_cases(path: str | Path) -> list[dict]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def exact_answer_accuracy(cases: list[dict], predict: Callable[[dict], str]) -> float:
    if not cases:
        return 0.0
    correct = sum(predict(case).strip() == case["answer"].strip() for case in cases)
    return correct / len(cases)
