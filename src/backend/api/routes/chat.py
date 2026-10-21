"""SSE chat stream over the existing LangGraph workflow."""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.deps import (
    CHECKPOINT_TTL_SECONDS, THREAD_TTL_SECONDS, check_thread_owner, get_current_student,
    get_graph_app, get_redis, require_live_checkpoint, thread_meta_key,
)
from backend.api.schemas import MessageRequest
from backend.api.sse import encode_sse, solution_event_data, sse_headers

router = APIRouter(prefix="/threads", tags=["chat"])


def _interrupt_from_snapshot(snapshot: Any, thread_id: str) -> dict | None:
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            value = getattr(item, "value", {})
            return {
                "thread_id": thread_id,
                "interrupt_id": getattr(item, "id", None),
                **(value if isinstance(value, dict) else {"prompt": str(value)}),
            }
    return None


async def graph_event_stream(graph: Any, redis: Any, thread_id: str, student_id: str, content: str) -> AsyncIterator[str]:
    from backend.agents.state import make_initial_state

    config = {"configurable": {"thread_id": thread_id}}
    initial = make_initial_state(student_id=student_id, thread_id=thread_id, raw_text=content)
    try:
        async for mode, chunk in graph.astream(
            initial,
            config=config,
            stream_mode=["updates", "messages", "custom"],
        ):
            if mode == "messages":
                message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
                text = getattr(message, "content", "")
                if isinstance(text, str) and text and message.__class__.__name__.endswith("Chunk"):
                    yield encode_sse("token", {"text": text})
            elif mode == "updates" and isinstance(chunk, dict):
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    timings = update.get("node_timings_ms") or {}
                    node_event = {"name": str(node_name), "status": "interrupted" if "__interrupt__" in update else "complete"}
                    if node_name in timings:
                        node_event["ms"] = timings[node_name]
                    yield encode_sse("node", node_event)
                    if "route" in update and "route_reason" in update:
                        route_path, route_reason = update.get("route"), update.get("route_reason")
                        plan = update.get("solution_plan") or {}
                        parsed = update.get("parsed_data") or {}
                        yield encode_sse("route", {
                            "path": route_path,
                            "complexity": update.get("complexity"),
                            "intent": plan.get("intent_type") if isinstance(plan, dict) else None,
                            "topic": parsed.get("topic") if isinstance(parsed, dict) else None,
                            "reason": route_reason,
                        })
            elif mode == "custom" and isinstance(chunk, dict) and chunk.get("event"):
                yield encode_sse(str(chunk["event"]), chunk.get("data") or {})

        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        pending = _interrupt_from_snapshot(snapshot, thread_id)
        now = time.time()
        meta_key = thread_meta_key(thread_id)
        redis.hset(meta_key, mapping={"last_active": now, "checkpoint_expires_at": now + CHECKPOINT_TTL_SECONDS})
        redis.expire(meta_key, THREAD_TTL_SECONDS)
        if pending:
            redis.hset(meta_key, mapping={"pending_interrupt_id": pending.get("interrupt_id") or ""})
            yield encode_sse("interrupt", pending)
            return

        redis.hdel(meta_key, "pending_interrupt_id")
        verifier = values.get("verifier_output") or {}
        if verifier:
            yield encode_sse("verification", {
                "status": verifier.get("status"),
                "method": verifier.get("verification_method") or verifier.get("method") or "unavailable",
                "confidence": verifier.get("confidence"),
            })
        final_response = values.get("final_response")
        if final_response:
            yield encode_sse("solution", solution_event_data(str(final_response)))
        yield encode_sse("done", {
            "run_id": str(uuid.uuid4()),
            "message_id": str(uuid.uuid4()),
            "models": [],
            "usage": {
                "llm_calls": int(values.get("llm_calls_used") or 0),
                "tool_calls": int(values.get("tool_calls_used") or 0),
            },
        })
    except Exception:
        # Never serialize exception text, prompts, provider payloads, or credentials.
        yield encode_sse("error", {"code": "run_failed", "message": "The tutoring request could not be completed.", "retryable": True})


@router.post("/{thread_id}/messages")
def send_message(
    thread_id: str,
    body: MessageRequest,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
    graph=Depends(get_graph_app),
):
    metadata = check_thread_owner(redis, thread_id, student_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "message": "Thread state is temporarily unavailable.", "retryable": True}) from exc
    require_live_checkpoint(metadata, snapshot)
    if _interrupt_from_snapshot(snapshot, thread_id):
        raise HTTPException(status_code=409, detail={"code": "resume_required", "message": "This thread is waiting for a response. Resume the pending interrupt first.", "retryable": False})
    now = time.time()
    redis.hset(thread_meta_key(thread_id), mapping={"last_active": now, "checkpoint_expires_at": now + CHECKPOINT_TTL_SECONDS})
    return StreamingResponse(
        graph_event_stream(graph, redis, thread_id, student_id, body.content.strip()),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
