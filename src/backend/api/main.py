"""MathTutor FastAPI application; durable graph execution is required for chat."""
from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.auth import router as auth_router
from backend.api.routes.models import router as models_router
from backend.api.routes.threads import router as threads_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.hitl import router as hitl_router

app = FastAPI(title="MathTutor API", version="0.1.0")

_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(models_router)
app.include_router(threads_router)
app.include_router(chat_router)
app.include_router(hitl_router)
