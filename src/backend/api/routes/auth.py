"""Google ID-token exchange for short-lived MathTutor API JWTs."""
from __future__ import annotations

import hashlib
import os
import time

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.deps import get_redis, issue_access_token
from backend.api.schemas import AuthResponse, GoogleAuthRequest

router = APIRouter(prefix="/auth", tags=["auth"])


def verify_google_id_token(id_token: str, client_id: str) -> dict:
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token as google_id_token

    return google_id_token.verify_oauth2_token(id_token, Request(), audience=client_id)


@router.post("/google", response_model=AuthResponse)
def google_login(body: GoogleAuthRequest, redis=Depends(get_redis)) -> AuthResponse:
    client_id = os.getenv("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "auth_not_configured", "message": "Google sign-in is not configured.", "retryable": False},
        )
    try:
        claims = verify_google_id_token(body.id_token, client_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "invalid_google_token", "message": "Google could not verify this sign-in token.", "retryable": False},
        ) from exc

    email = str(claims.get("email") or "").strip().lower()
    verified = claims.get("email_verified") is True or str(claims.get("email_verified", "")).lower() == "true"
    issuer = claims.get("iss")
    if (
        not email
        or not verified
        or issuer not in {"accounts.google.com", "https://accounts.google.com"}
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "unverified_google_identity", "message": "A verified Google email is required.", "retryable": False},
        )

    student_id = hashlib.sha256(email.encode("utf-8")).hexdigest()[:16]
    now = time.time()
    key = f"user:{student_id}"
    if not redis.exists(key):
        redis.hset(key, mapping={
            "student_id": student_id,
            "email": email,
            "display_name": str(claims.get("name") or claims.get("given_name") or email.split("@")[0])[:120],
            "created_at": now,
            "last_login": now,
            "total_problems_solved": 0,
        })
    else:
        redis.hset(key, mapping={
            "last_login": now,
            "display_name": str(claims.get("name") or claims.get("given_name") or email.split("@")[0])[:120],
        })
    display_name = str(claims.get("name") or claims.get("given_name") or email.split("@")[0])[:120]
    try:
        token, expires_in = issue_access_token(student_id)
    except HTTPException:
        raise
    return AuthResponse(
        access_token=token,
        expires_in=expires_in,
        student_id=student_id,
        display_name=display_name,
    )
