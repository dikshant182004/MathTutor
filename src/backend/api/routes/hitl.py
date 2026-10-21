"""Idempotent HITL resume on the exact checkpoint/thread that was interrupted."""
from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.types import Command

from backend.api.deps import (
    CHECKPOINT_TTL_SECONDS, IDEMPOTENCY_TTL_SECONDS, THREAD_TTL_SECONDS, check_thread_owner,
    get_current_student, get_graph_app, get_redis, require_live_checkpoint,
    thread_meta_key, validate_idempotency_key,
)
from backend.api.routes.chat import _interrupt_from_snapshot
from backend.api.schemas import ResumeRequest
from backend.api.sse import encode_sse, solution_event_data, sse_headers

router = APIRouter(prefix="/threads", tags=["hitl"])


async def _resume_stream(
    graph: Any, redis: Any, thread_id: str,
    response: dict, idempotency_storage_key: str,
    resume_lock_key: str, resume_lock_token: str,
) -> AsyncIterator[str]:
    events: list[str] = []
    completed = False
    config = {"configurable": {"thread_id": thread_id}}
    try:
        async for mode, chunk in graph.astream(
            Command(resume=response),
            config=config,
            stream_mode=["updates", "messages", "custom"],
        ):
            if mode == "messages":
                message = chunk[0] if isinstance(chunk, tuple) and chunk else chunk
                content = getattr(message, "content", "")
                if isinstance(content, str) and content and message.__class__.__name__.endswith("Chunk"):
                    event = encode_sse("token", {"text": content})
                    events.append(event)
                    yield event
            elif mode == "updates" and isinstance(chunk, dict):
                for node_name, update in chunk.items():
                    if isinstance(update, dict):
                        event = encode_sse("node", {"name": str(node_name), "status": "interrupted" if "__interrupt__" in update else "complete"})
                        events.append(event)
                        yield event
                        if any(k in update for k in ("route", "route_reason", "complexity", "intake_output")):
                            from backend.agents.routing_policy import route_with_reason
                            try:
                                route_path, route_reason = route_with_reason(update)
                            except Exception:
                                route_path, route_reason = update.get("route"), update.get("route_reason")
                            plan = update.get("solution_plan") or {}
                            parsed = update.get("parsed_data") or {}
                            event = encode_sse("route", {
                                "path": route_path,
                                "complexity": update.get("complexity"),
                                "intent": plan.get("intent_type") if isinstance(plan, dict) else None,
                                "topic": parsed.get("topic") if isinstance(parsed, dict) else None,
                                "reason": route_reason,
                            })
                            events.append(event)
                            yield event
            elif mode == "custom" and isinstance(chunk, dict) and chunk.get("event"):
                event = encode_sse(str(chunk["event"]), chunk.get("data") or {})
                events.append(event)
                yield event

        snapshot = graph.get_state(config)
        values = getattr(snapshot, "values", None) or {}
        now = time.time()
        redis.hset(thread_meta_key(thread_id), mapping={"last_active": now, "checkpoint_expires_at": now + CHECKPOINT_TTL_SECONDS})
        redis.expire(thread_meta_key(thread_id), THREAD_TTL_SECONDS)
        pending = _interrupt_from_snapshot(snapshot, thread_id)
        if pending:
            redis.hset(thread_meta_key(thread_id), mapping={"pending_interrupt_id": pending.get("interrupt_id") or ""})
            event = encode_sse("interrupt", pending)
            events.append(event)
            yield event
        else:
            redis.hdel(thread_meta_key(thread_id), "pending_interrupt_id")
            verifier = values.get("verifier_output") or {}
            if verifier:
                event = encode_sse("verification", {
                    "status": verifier.get("status"),
                    "method": verifier.get("verification_method") or verifier.get("method") or "unavailable",
                    "confidence": verifier.get("confidence"),
                })
                events.append(event)
                yield event
            final_response = values.get("final_response")
            if final_response:
                event = encode_sse("solution", solution_event_data(str(final_response)))
                events.append(event)
                yield event
            event = encode_sse("done", {
                "run_id": uuid.uuid4().hex,
                "message_id": __import__("uuid").uuid4().hex,
                "models": [],
                "usage": {
                    "llm_calls": int(values.get("llm_calls_used") or 0),
                    "tool_calls": int(values.get("tool_calls_used") or 0),
                },
            })
            events.append(event)
            yield event
        completed = True
    except Exception:
        event = encode_sse("error", {"code": "resume_failed", "message": "The interrupted request could not be resumed.", "retryable": True})
        events.append(event)
        yield event
        completed = True
    finally:
        # Cache the exact emitted events; the same key can never execute resume twice.
        if completed:
            redis.set(
                idempotency_storage_key,
                json.dumps({"status": "complete", "events": events}, separators=(",", ":")),
                ex=IDEMPOTENCY_TTL_SECONDS,
            )
        else:
            redis.set(
                idempotency_storage_key,
                json.dumps({"status": "failed", "events": [encode_sse("error", {"code": "resume_interrupted", "message": "Resume did not complete; do not reuse this request key.", "retryable": False})]}, separators=(",", ":")),
                ex=IDEMPOTENCY_TTL_SECONDS,
            )
        # Production Redis uses atomic compare-and-delete to avoid releasing a lock
        # that expired and was subsequently acquired by another request.
        try:
            redis.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
                1, resume_lock_key, resume_lock_token,
            )
        except AttributeError:
            if redis.get(resume_lock_key) == resume_lock_token:
                redis.delete(resume_lock_key)


@router.post("/{thread_id}/resume")
def resume_thread(
    thread_id: str,
    body: ResumeRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
    graph=Depends(get_graph_app),
):
    metadata = check_thread_owner(redis, thread_id, student_id)
    key = validate_idempotency_key(idempotency_key)
    storage_key = f"idem:{thread_id}:{key}"

    # Idempotency is checked before checkpoint state: a completed duplicate must
    # replay its first result even after the graph has moved past the interrupt.
    raw = redis.get(storage_key)
    if raw:
        try:
            cached = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except (ValueError, TypeError):
            cached = {}
        if cached.get("status") in {"complete", "failed"}:
            return StreamingResponse(
                iter(cached.get("events") or []),
                media_type="text/event-stream",
                headers=sse_headers(),
            )
        raise HTTPException(status_code=409, detail={"code": "resume_in_progress", "message": "A resume with this idempotency key has already started.", "retryable": False})

    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "message": "Thread state is temporarily unavailable.", "retryable": True}) from exc
    require_live_checkpoint(metadata, snapshot)
    pending = _interrupt_from_snapshot(snapshot, thread_id)
    if not pending:
        raise HTTPException(status_code=409, detail={"code": "no_pending_interrupt", "message": "This thread has no pending interrupt to resume.", "retryable": False})
    if body.interrupt_id != str(pending.get("interrupt_id") or ""):
        raise HTTPException(status_code=409, detail={"code": "stale_interrupt", "message": "The supplied interrupt ID does not match the thread's current pending interrupt.", "retryable": False})

    hitl_type = str(pending.get("hitl_type") or pending.get("type") or "clarification")
    decision = body.decision
    response = dict(body.response)
    allowed_decisions = {
        "verification": {"correct", "incorrect"},
        "clarification": {"submit", "retry"},
        "bad_input": {"submit", "retry"},
    }.get(hitl_type, {"submit", "retry"})
    if decision not in allowed_decisions:
        raise HTTPException(status_code=422, detail={"code": "invalid_decision", "message": "The decision is not valid for this interrupt type.", "retryable": False})
    if hitl_type == "verification":
        expected = decision == "correct"
        if "is_correct" in response and bool(response["is_correct"]) != expected:
            raise HTTPException(status_code=422, detail={"code": "decision_mismatch", "message": "The decision conflicts with the verification response.", "retryable": False})
        response["is_correct"] = expected
    if hitl_type == "clarification" and not any(
        isinstance(response.get(field), str) and response[field].strip()
        for field in ("corrected_text", "answer", "raw_text")
    ):
        raise HTTPException(status_code=422, detail={"code": "response_required", "message": "Provide clarification text before resuming.", "retryable": False})
    if hitl_type == "bad_input" and decision == "submit" and not any(
        isinstance(response.get(field), str) and response[field].strip()
        for field in ("raw_text", "new_image_path", "new_audio_path")
    ):
        raise HTTPException(status_code=422, detail={"code": "response_required", "message": "Provide replacement input before resuming.", "retryable": False})

    resume_lock_key = f"resume-lock:{thread_id}"
    resume_lock_token = uuid.uuid4().hex
    if not redis.set(resume_lock_key, resume_lock_token, ex=300, nx=True):
        raise HTTPException(status_code=409, detail={"code": "thread_resume_in_progress", "message": "Another request is already resuming this thread.", "retryable": True})

    reserved = redis.set(storage_key, json.dumps({"status": "processing"}), ex=IDEMPOTENCY_TTL_SECONDS, nx=True)
    if not reserved:
        try:
            redis.eval(
                "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end",
                1, resume_lock_key, resume_lock_token,
            )
        except AttributeError:
            redis.delete(resume_lock_key)
        raw = redis.get(storage_key)
        try:
            cached = json.loads(raw.decode() if isinstance(raw, bytes) else raw or "{}")
        except (ValueError, TypeError):
            cached = {}
        if cached.get("status") in {"complete", "failed"}:
            return StreamingResponse(
                iter(cached.get("events") or []),
                media_type="text/event-stream",
                headers=sse_headers(),
            )
        raise HTTPException(status_code=409, detail={"code": "resume_in_progress", "message": "A resume with this idempotency key has already started.", "retryable": False})
    return StreamingResponse(
        _resume_stream(graph, redis, thread_id, response, storage_key, resume_lock_key, resume_lock_token),
        media_type="text/event-stream",
        headers=sse_headers(),
    )
