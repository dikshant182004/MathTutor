"""MathTutor FastAPI application; durable graph execution is required for chat."""
from __future__ import annotations

import os
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes.auth import router as auth_router
from backend.api.routes.models import router as models_router
from backend.api.routes.threads import router as threads_router
from backend.api.routes.chat import router as chat_router
from backend.api.routes.hitl import router as hitl_router

app = FastAPI(title="MathTutor API", version="0.1.0", docs_url=None, redoc_url=None, openapi_url=None)

_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Idempotency-Key"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_request: Request, _exc: RequestValidationError):
    # Do not echo submitted prompts, files, or other input values in validation errors.
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "invalid_request", "message": "The request did not match the required schema.", "retryable": False}},
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(auth_router)
app.include_router(models_router)
app.include_router(threads_router)
app.include_router(chat_router)
app.include_router(hitl_router)
