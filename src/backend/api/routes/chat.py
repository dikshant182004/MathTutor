"""SSE chat stream over LangGraph with durable Postgres conversation history."""
from __future__ import annotations

import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.deps import (
    CHECKPOINT_TTL_SECONDS, THREAD_TTL_SECONDS, check_thread_owner, get_current_student,
    get_graph_app, get_postgres, get_redis, require_live_checkpoint, thread_meta_key,
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


async def graph_event_stream(
    graph: Any, redis: Any, store: Any, thread_id: str, student_id: str,
    content: str, run_id: str,
) -> AsyncIterator[str]:
    from backend.agents.state import make_initial_state

    config = {"configurable": {"thread_id": thread_id}}
    initial = make_initial_state(student_id=student_id, thread_id=thread_id, raw_text=content)
    started = time.monotonic()
    status_value = "failed"
    try:
        async for mode, chunk in graph.astream(initial, config=config, stream_mode=["updates", "messages", "custom"]):
            if mode == "messages":
                message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
                token = getattr(message, "content", "")
                if isinstance(token, str) and token and message.__class__.__name__.endswith("Chunk"):
                    yield encode_sse("token", {"text": token})
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
                        plan = update.get("solution_plan") or {}
                        parsed = update.get("parsed_data") or {}
                        yield encode_sse("route", {
                            "path": update.get("route"), "complexity": update.get("complexity"),
                            "intent": plan.get("intent_type") if isinstance(plan, dict) else None,
                            "topic": parsed.get("topic") if isinstance(parsed, dict) else None,
                            "reason": update.get("route_reason"),
                        })
            elif mode == "custom" and isinstance(chunk, dict) and chunk.get("event"):
                yield encode_sse(str(chunk["event"]), chunk.get("data") or {})

        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        pending = _interrupt_from_snapshot(snapshot, thread_id)
        now = time.time()
        redis.hset(thread_meta_key(thread_id), mapping={"last_active": now, "checkpoint_expires_at": now + CHECKPOINT_TTL_SECONDS})
        redis.expire(thread_meta_key(thread_id), THREAD_TTL_SECONDS)
        if pending:
            status_value = "interrupted"
            redis.hset(thread_meta_key(thread_id), mapping={"pending_interrupt_id": pending.get("interrupt_id") or ""})
            store.append_message(student_id, thread_id, "assistant", str(pending.get("prompt") or "A response is required to continue."))
            yield encode_sse("interrupt", pending)
        else:
            status_value = "completed"
            redis.hdel(thread_meta_key(thread_id), "pending_interrupt_id")
            verifier = values.get("verifier_output") or {}
            if verifier:
                yield encode_sse("verification", {
                    "status": verifier.get("status"),
                    "method": verifier.get("verification_method") or verifier.get("method") or "unavailable",
                    "confidence": verifier.get("confidence"),
                })
            final_response = values.get("final_response")
            if final_response:
                store.append_message(student_id, thread_id, "assistant", str(final_response))
                yield encode_sse("solution", solution_event_data(str(final_response)))
        usage = {
            "llm_calls": int(values.get("llm_calls_used") or 0),
            "tool_calls": int(values.get("tool_calls_used") or 0),
        }
        store.finish_run(
            student_id, thread_id, run_id, status=status_value,
            route=values.get("route"), complexity=values.get("complexity"),
            llm_calls=usage["llm_calls"], tool_calls=usage["tool_calls"],
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        yield encode_sse("done", {"run_id": run_id, "message_id": str(uuid.uuid4()), "models": [], "usage": usage})
    except Exception:
        try:
            store.finish_run(
                student_id, thread_id, run_id, status="failed",
                duration_ms=int((time.monotonic() - started) * 1000), error_code="run_failed",
            )
        except Exception:
            pass
        yield encode_sse("error", {"code": "run_failed", "message": "The tutoring request could not be completed.", "retryable": True})


@router.post("/{thread_id}/messages")
def send_message(
    thread_id: str,
    body: MessageRequest,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
    graph=Depends(get_graph_app),
    store=Depends(get_postgres),
):
    metadata = check_thread_owner(redis, thread_id, student_id)
    if redis.get(f"resume-lock:{thread_id}"):
        raise HTTPException(status_code=409, detail={"code": "thread_resume_in_progress", "message": "This thread is currently being resumed. Retry after it completes.", "retryable": True})
    try:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "message": "Thread state is temporarily unavailable.", "retryable": True}) from exc
    require_live_checkpoint(metadata, snapshot)
    if _interrupt_from_snapshot(snapshot, thread_id):
        raise HTTPException(status_code=409, detail={"code": "resume_required", "message": "This thread is waiting for a response. Resume the pending interrupt first.", "retryable": False})
    content = body.content.strip()
    run_id = str(uuid.uuid4())
    try:
        store.append_message(student_id, thread_id, "user", content)
        store.start_run(student_id, thread_id, run_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "history_write_failed", "message": "The message could not be safely persisted.", "retryable": True}) from exc
    now = time.time()
    redis.hset(thread_meta_key(thread_id), mapping={"last_active": now, "checkpoint_expires_at": now + CHECKPOINT_TTL_SECONDS})
    return StreamingResponse(
        graph_event_stream(graph, redis, store, thread_id, student_id, content, run_id),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
