from __future__ import annotations

import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.agents.graph import chatbot
from backend.agents.state import make_initial_state
from backend.v2.routing import classify_problem


class SolveRequest(BaseModel):
    student_id: str = "local-student"
    thread_id: str | None = None
    problem: str = Field(min_length=1, max_length=12000)


class SolveResponse(BaseModel):
    thread_id: str
    execution_plan: dict | None = None
    final_response: str | None = None
    status: str


app = FastAPI(title="MathTutor V2 API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "mathtutor-v2"}


@app.post("/v2/plan")
def plan(request: SolveRequest) -> dict:
    return classify_problem(request.problem).__dict__


@app.post("/v2/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    thread_id = request.thread_id or str(uuid4())
    try:
        state = make_initial_state(
            student_id=request.student_id,
            thread_id=thread_id,
            raw_text=request.problem,
        )
        result = chatbot.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        return SolveResponse(
            thread_id=thread_id,
            execution_plan=result.get("execution_plan"),
            final_response=result.get("final_response"),
            status="ok" if result.get("final_response") else "needs_follow_up",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
