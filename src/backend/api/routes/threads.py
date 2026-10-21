"""Thread registry and persisted graph-state history endpoints."""
from __future__ import annotations

import time
from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.deps import (
    check_thread_owner, get_current_student, get_graph_app, get_redis,
    stm_summary_key, thread_meta_key, threads_key, THREAD_TTL_SECONDS,
    require_live_checkpoint,
)
from backend.api.schemas import ThreadCreateRequest, ThreadSummary

router = APIRouter(prefix="/threads", tags=["threads"])


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _public_thread(meta: dict) -> dict:
    meta = {_text(k): _text(v) for k, v in (meta or {}).items()}
    def number(name):
        try:
            return float(meta[name]) if meta.get(name) not in (None, "") else None
        except (TypeError, ValueError):
            return None
    return {
        "thread_id": meta.get("thread_id", ""),
        "created_at": number("created_at"),
        "last_active": number("last_active"),
        "problem_summary": meta.get("problem_summary", ""),
        "topic": meta.get("topic", ""),
        "outcome": meta.get("outcome", "in_progress"),
    }


@router.get("")
def list_threads(student_id: str = Depends(get_current_student), redis=Depends(get_redis)) -> dict:
    thread_ids = redis.zrevrange(threads_key(student_id), 0, 19) or []
    threads = []
    for raw_id in thread_ids:
        thread_id = _text(raw_id)
        if not thread_id.startswith(student_id + ":"):
            continue
        meta = redis.hgetall(thread_meta_key(thread_id)) or {}
        if meta:
            public = _public_thread(meta)
            if public["thread_id"] == thread_id:
                threads.append(public)
    return {"threads": threads}


@router.post("", response_model=ThreadSummary, status_code=status.HTTP_201_CREATED)
def create_thread(
    body: ThreadCreateRequest,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
) -> ThreadSummary:
    now = time.time()
    thread_id = f"{student_id}:{int(now * 1000)}"
    # Avoid a collision if two create requests share a millisecond timestamp.
    while redis.exists(thread_meta_key(thread_id)):
        now += 0.001
        thread_id = f"{student_id}:{int(now * 1000)}"
    redis.zadd(threads_key(student_id), {thread_id: now})
    redis.hset(thread_meta_key(thread_id), mapping={
        "thread_id": thread_id,
        "student_id": student_id,
        "created_at": now,
        "last_active": now,
        "problem_summary": (body.title or "")[:120],
        "topic": "",
        "outcome": "in_progress",
    })
    redis.expire(thread_meta_key(thread_id), THREAD_TTL_SECONDS)
    return ThreadSummary(**_public_thread(redis.hgetall(thread_meta_key(thread_id))))


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: str,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
) -> None:
    check_thread_owner(redis, thread_id, student_id)
    if redis.get(f"resume-lock:{thread_id}"):
        raise HTTPException(status_code=409, detail={"code": "thread_resume_in_progress", "message": "This thread cannot be deleted while it is being resumed.", "retryable": True})
    redis.zrem(threads_key(student_id), thread_id)
    redis.delete(thread_meta_key(thread_id), stm_summary_key(thread_id))
    # Checkpoints are access-denied immediately and expire under the checkpointer TTL.
    # Delete resume idempotency records too; they must not survive a user-requested delete.
    try:
        keys = list(redis.scan_iter(match=f"idem:{thread_id}:*"))
        if keys:
            redis.delete(*keys)
    except Exception:
        pass


@router.get("/{thread_id}/messages")
def get_messages(
    thread_id: str,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
    graph=Depends(get_graph_app),
) -> dict:
    metadata = check_thread_owner(redis, thread_id, student_id)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"code": "checkpoint_unavailable", "message": "Thread history is temporarily unavailable.", "retryable": True}) from exc
    require_live_checkpoint(metadata, snapshot)
    values = getattr(snapshot, "values", None) or {}
    messages = []
    for item in values.get("messages", []) or []:
        role = getattr(item, "type", None) or ("assistant" if item.__class__.__name__.startswith("AI") else "user")
        content = getattr(item, "content", "")
        if isinstance(content, list):
            content = " ".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        messages.append({"id": getattr(item, "id", None), "role": role, "content": str(content)})
    pending = None
    for task in getattr(snapshot, "tasks", ()) or ():
        for item in getattr(task, "interrupts", ()) or ():
            pending = {
                "thread_id": thread_id,
                "interrupt_id": getattr(item, "id", None),
                **(getattr(item, "value", {}) if isinstance(getattr(item, "value", {}), dict) else {"prompt": str(getattr(item, "value", ""))}),
            }
            break
        if pending:
            break
    return {"thread_id": thread_id, "messages": messages, "pending_interrupt": pending}
