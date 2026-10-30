"""Thread registry and Postgres-backed conversation history endpoints."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.deps import (
    check_thread_owner, get_current_student, get_graph_app, get_postgres, get_redis,
    stm_summary_key, thread_meta_key, threads_key, THREAD_TTL_SECONDS,
    require_live_checkpoint,
)
from backend.api.schemas import ThreadCreateRequest, ThreadSummary

router = APIRouter(prefix="/threads", tags=["threads"])


def _epoch(value):
    if isinstance(value, datetime):
        return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _public_thread(row: dict) -> dict:
    return {
        "thread_id": row.get("thread_id", ""),
        "created_at": _epoch(row.get("created_at")),
        "last_active": _epoch(row.get("last_active")),
        "problem_summary": row.get("title", row.get("problem_summary", "")) or "",
        "topic": row.get("topic", "") or "",
        "outcome": row.get("outcome", "in_progress") or "in_progress",
    }


@router.get("")
def list_threads(
    student_id: str = Depends(get_current_student),
    store=Depends(get_postgres),
) -> dict:
    return {"threads": [_public_thread(row) for row in store.list_threads(student_id, limit=20)]}


@router.post("", response_model=ThreadSummary, status_code=status.HTTP_201_CREATED)
def create_thread(
    body: ThreadCreateRequest,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
    store=Depends(get_postgres),
) -> ThreadSummary:
    now = time.time()
    thread_id = f"{student_id}:{int(now * 1000)}"
    while store.get_thread(student_id, thread_id) is not None:
        now += 0.001
        thread_id = f"{student_id}:{int(now * 1000)}"
    row = store.create_thread(student_id, thread_id, (body.title or "")[:120])
    # Redis keeps only the short-lived graph/checkpoint coordination metadata.
    redis.zadd(threads_key(student_id), {thread_id: now})
    redis.hset(thread_meta_key(thread_id), mapping={
        "thread_id": thread_id, "student_id": student_id,
        "created_at": now, "last_active": now,
        "problem_summary": (body.title or "")[:120], "topic": "", "outcome": "in_progress",
    })
    redis.expire(thread_meta_key(thread_id), THREAD_TTL_SECONDS)
    return ThreadSummary(**_public_thread(row))


@router.delete("/{thread_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_thread(
    thread_id: str,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
    store=Depends(get_postgres),
) -> None:
    if store.get_thread(student_id, thread_id) is None:
        raise HTTPException(status_code=404, detail={"code": "thread_not_found", "message": "Thread not found.", "retryable": False})
    if redis.get(f"resume-lock:{thread_id}"):
        raise HTTPException(status_code=409, detail={"code": "thread_resume_in_progress", "message": "This thread cannot be deleted while it is being resumed.", "retryable": True})
    # Postgres cascades messages and run analytics only for this owned thread.
    store.delete_thread(student_id, thread_id)
    redis.zrem(threads_key(student_id), thread_id)
    redis.delete(thread_meta_key(thread_id), stm_summary_key(thread_id))
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
    store=Depends(get_postgres),
) -> dict:
    if store.get_thread(student_id, thread_id) is None:
        raise HTTPException(status_code=404, detail={"code": "thread_not_found", "message": "Thread not found.", "retryable": False})
    # The relational history remains available after Redis metadata/checkpoint expiry.
    messages = [
        {"id": str(row["message_id"]), "role": row["role"], "content": row["content"],
         "created_at": _epoch(row["created_at"])}
        for row in store.list_messages(student_id, thread_id)
    ]
    pending = None
    try:
        snapshot = graph.get_state({"configurable": {"thread_id": thread_id}})
        tasks = getattr(snapshot, "tasks", ()) or ()
        for task in tasks:
            for item in getattr(task, "interrupts", ()) or ():
                value = getattr(item, "value", {})
                pending = {
                    "thread_id": thread_id, "interrupt_id": getattr(item, "id", None),
                    **(value if isinstance(value, dict) else {"prompt": str(value)}),
                }
                break
            if pending:
                break
    except Exception:
        # History is durable even when the optional live checkpoint is unavailable.
        pending = None
    return {"thread_id": thread_id, "messages": messages, "pending_interrupt": pending}


@router.get("/analytics/summary")
def analytics_summary(
    student_id: str = Depends(get_current_student),
    store=Depends(get_postgres),
) -> dict:
    """Return aggregate usage for the authenticated student only."""
    return store.analytics_summary(student_id)
