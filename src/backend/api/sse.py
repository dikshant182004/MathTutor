"""Small helpers for the documented server-sent event contract."""
from __future__ import annotations

import json
import re
from typing import Any


def encode_sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


_STEP_HEADER = re.compile(r"(?m)^\\s*Step\\s+(\\d+)\\s*[—–:-]\\s*")


def solution_event_data(text: str) -> dict[str, Any]:
    """Parse existing 'Step N —' blocks without inventing missing steps."""
    matches = list(_STEP_HEADER.finditer(text))
    steps: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        steps.append({"number": int(match.group(1)), "text": body})
    return {"steps": steps, "text": text}
