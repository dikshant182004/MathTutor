"""Validated request and response contracts for the MathTutor API."""
from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(min_length=1, max_length=8192)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    student_id: str
    display_name: str


class ThreadCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ThreadSummary(BaseModel):
    thread_id: str
    created_at: float | None = None
    last_active: float | None = None
    problem_summary: str = ""
    topic: str = ""
    outcome: str = "in_progress"


class MessageRequest(BaseModel):
    content: str = Field(min_length=1, max_length=4000)


class ResumeRequest(BaseModel):
    interrupt_id: str = Field(min_length=1, max_length=256)
    decision: Literal["submit", "retry", "correct", "incorrect"] = "submit"
    response: dict[str, Any] = Field(min_length=1)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str
    message: str
    retryable: bool = False
