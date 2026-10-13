"""Authentication, ownership checks, and lazy runtime dependencies."""
from __future__ import annotations

import hashlib
import os
import re
import time
from typing import Any

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)
_STUDENT_ID = re.compile(r"^[0-9a-f]{16}$")
_THREAD_ID = re.compile(r"^(?P<student>[0-9a-f]{16}):(?P<stamp>[0-9]{1,20})$")
_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
JWT_ISSUER = "mathtutor-api"
JWT_TTL_SECONDS = 3600
CHECKPOINT_TTL_SECONDS = 24 * 60 * 60
THREAD_TTL_SECONDS = 7 * 24 * 60 * 60
IDEMPOTENCY_TTL_SECONDS = 24 * 60 * 60


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET", "")
    if len(secret) < 32 or secret.lower().startswith(("replace-with-", "change-me-", "your-secret")):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "auth_not_configured", "message": "Authentication is not configured.", "retryable": False},
        )
    return secret


def issue_access_token(student_id: str, *, now: int | None = None) -> tuple[str, int]:
    if not _STUDENT_ID.fullmatch(student_id):
        raise ValueError("student_id must be the stable 16-character hex identifier")
    issued = int(time.time() if now is None else now)
    claims = {
        "sub": student_id,
        "iss": JWT_ISSUER,
        "iat": issued,
        "exp": issued + JWT_TTL_SECONDS,
    }
    token = jwt.encode(claims, _jwt_secret(), algorithm="HS256")
    return token, JWT_TTL_SECONDS


def decode_access_token(token: str) -> str:
    try:
        claims = jwt.decode(
            token,
            _jwt_secret(),
            algorithms=["HS256"],
            issuer=JWT_ISSUER,
            options={"require": ["sub", "iss", "iat", "exp"]},
        )
    except HTTPException:
        raise
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "The access token is invalid or expired.", "retryable": False},
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    student_id = claims.get("sub")
    if not isinstance(student_id, str) or not _STUDENT_ID.fullmatch(student_id):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_token", "message": "The access token is invalid or expired.", "retryable": False},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return student_id


def enforce_rate_limit(
    redis: Any, *, client_ip: str, student_id: str | None = None, include_ip: bool = True
) -> None:
    """Use Redis counters so limits work across stateless API instances."""
    window = int(time.time() // 60)
    ip_digest = hashlib.sha256((client_ip or "unknown").encode("utf-8")).hexdigest()[:16]
    checks = [(f"rl:ip:{ip_digest}:{window}", 120)] if include_ip else []
    if student_id:
        checks.append((f"rl:student:{student_id}:{window}", 60))
    for key, limit in checks:
        count = int(redis.incr(key))
        if count == 1:
            redis.expire(key, 120)
        if count > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"code": "rate_limited", "message": "Too many requests. Please wait before trying again.", "retryable": True},
                headers={"Retry-After": "60"},
            )


def get_redis():
    """Require the shared Redis service; API requests must not fall back to process memory."""
    try:
        from backend.agents.utils.db_utils import get_sync_client
        client = get_sync_client()
        client.ping()
        return client
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "storage_unavailable", "message": "Persistent storage is unavailable.", "retryable": True},
        ) from exc


def get_current_student(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    redis=Depends(get_redis),
) -> str:
    client_ip = request.client.host if request.client else "unknown"
    # Rate-limit even invalid/missing tokens so unauthenticated callers cannot bypass IP controls.
    enforce_rate_limit(redis, client_ip=client_ip)
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "missing_token", "message": "Bearer authentication is required.", "retryable": False},
            headers={"WWW-Authenticate": "Bearer"},
        )
    student_id = decode_access_token(credentials.credentials)
    enforce_rate_limit(redis, client_ip=client_ip, student_id=student_id, include_ip=False)
    return student_id


def get_graph_app():
    """Return the graph only when its checkpointer is durable, never InMemorySaver in the API."""
    try:
        from backend.agents.graph import chatbot, checkpointer
        if "redis" not in type(checkpointer).__name__.lower():
            raise RuntimeError("A durable Redis checkpointer is required for API execution.")
        return chatbot
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "graph_unavailable", "message": "The tutoring graph is not ready for durable execution.", "retryable": True},
        ) from exc


def thread_meta_key(thread_id: str) -> str:
    return f"thread:{thread_id}:meta"


def threads_key(student_id: str) -> str:
    return f"threads:{student_id}"


def stm_summary_key(thread_id: str) -> str:
    return f"stm:summary:{thread_id}"


def check_thread_owner(redis: Any, thread_id: str, student_id: str) -> dict[str, str]:
    """Check both the embedded owner prefix and persisted metadata before touching state."""
    match = _THREAD_ID.fullmatch(thread_id or "")
    if not match or match.group("student") != student_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "thread_not_found", "message": "Thread not found.", "retryable": False},
        )
    metadata = redis.hgetall(thread_meta_key(thread_id)) or {}
    metadata = {
        (key.decode() if isinstance(key, bytes) else str(key)):
        (value.decode() if isinstance(value, bytes) else str(value))
        for key, value in metadata.items()
    }
    if not metadata or metadata.get("student_id") != student_id or metadata.get("thread_id", thread_id) != thread_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "thread_not_found", "message": "Thread not found.", "retryable": False},
        )
    return metadata


def require_live_checkpoint(metadata: dict[str, str], snapshot: Any) -> None:
    values = getattr(snapshot, "values", None) or {}
    tasks = getattr(snapshot, "tasks", None) or ()
    has_interrupt = any(getattr(task, "interrupts", None) for task in tasks)
    has_checkpoint = bool(values) or bool(getattr(snapshot, "next", None)) or has_interrupt
    expiry = metadata.get("checkpoint_expires_at")
    try:
        expired = expiry is not None and float(expiry) <= time.time()
    except (TypeError, ValueError):
        expired = True
    # A checkpoint expected by metadata but missing from the saver is not resumable.
    missing_checkpoint = expiry is not None and not has_checkpoint
    if expired or missing_checkpoint:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail={"code": "checkpoint_expired", "message": "This thread's checkpoint has expired. Start a new thread to continue.", "retryable": False},
        )


def validate_idempotency_key(value: str | None) -> str:
    if value is None or not _IDEMPOTENCY_KEY.fullmatch(value):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "idempotency_key_required", "message": "Provide a valid Idempotency-Key header.", "retryable": False},
        )
    return value
