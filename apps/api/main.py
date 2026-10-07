from __future__ import annotations

import os
from io import BytesIO
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import re
from pypdf import PdfReader

from backend.agents.graph import chatbot
from backend.agents.state import make_initial_state
from backend.v2.routing import classify_problem
from backend.v2.socratic import next_socratic_step
from backend.v2.student_memory import student_memory
from backend.v2.document_store import document_store
from backend.v2.observability import active_trace, new_trace, trace_store
from backend.v2.practice import generate_problem, generate_session


class SolveRequest(BaseModel):
    student_id: str = Field(default="local-student", pattern=r"^[A-Za-z0-9._-]{1,80}$")
    thread_id: str | None = None
    problem: str = Field(min_length=1, max_length=12000)


class SolveResponse(BaseModel):
    thread_id: str
    execution_plan: dict | None = None
    final_response: str | None = None
    status: str


app = FastAPI(title="MathTutor V2 API", version="2.0.0")

_STUDENT_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

def validate_student_id(student_id: str) -> str:
    if not _STUDENT_ID.fullmatch(student_id):
        raise HTTPException(status_code=400, detail="Invalid student_id format.")
    return student_id

@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    expected = os.getenv("V2_API_KEY", "").strip()
    if expected and request.url.path != "/health":
        supplied = request.headers.get("X-V2-API-Key", "")
        if supplied != expected:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
    return await call_next(request)
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

@app.get("/v2/observability/traces")
def traces(limit: int = 50) -> dict:
    return {"items": trace_store.recent(limit)}


@app.post("/v2/plan")
def plan(request: SolveRequest) -> dict:
    return classify_problem(request.problem).__dict__


class DocumentTextRequest(BaseModel):
    student_id: str = Field(default="local-student", pattern=r"^[A-Za-z0-9._-]{1,80}$")
    document_id: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=240)
    text: str = Field(min_length=1, max_length=500000)

@app.post("/v2/documents/text")
def ingest_document_text(request: DocumentTextRequest) -> dict:
    return document_store.ingest(
        request.student_id, request.document_id, request.text, request.source
    )

@app.post("/v2/documents/pdf")
async def ingest_document_pdf(
    student_id: str,
    document_id: str,
    file: UploadFile = File(...),
) -> dict:
    validate_student_id(student_id)
    if not _STUDENT_ID.fullmatch(document_id):
        raise HTTPException(status_code=400, detail="Invalid document_id format.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail="Only PDF uploads are supported.")
    data = await file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds the 15 MB limit.")
    try:
        reader = PdfReader(BytesIO(data))
        total = 0
        for page_no, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                result = document_store.ingest(
                    student_id, document_id, text, file.filename or document_id, page_no
                )
                total += result["chunks"]
        return {"document_id": document_id, "pages": len(reader.pages), "chunks": total}
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to parse the PDF.") from exc

@app.get("/v2/documents/{student_id}")
def list_documents(student_id: str) -> dict:
    return {"items": document_store.list_documents(student_id)}

@app.get("/v2/students/{student_id}/snapshot")
def student_snapshot(student_id: str) -> dict:
    return student_memory.snapshot(student_id)

@app.get("/v2/students/{student_id}/graph")
def student_graph(student_id: str) -> dict:
    return student_memory.graph(student_id)

@app.get("/v2/students/{student_id}/mistakes")
def student_mistakes(student_id: str, limit: int = 20) -> dict:
    return {"items": student_memory.mistakes(student_id, max(1, min(limit, 100)))}

@app.get("/v2/students/{student_id}/practice")
def practice(student_id: str, count: int = 5) -> dict:
    recommendation = student_memory.next_problem(student_id)
    return {
        "recommendation": recommendation,
        "items": generate_session(
            recommendation["skill"], recommendation["difficulty"], count=count
        ),
    }

@app.get("/v2/students/{student_id}/next-problem")
def next_problem(student_id: str) -> dict:
    return student_memory.next_problem(student_id)

class SocraticRequest(BaseModel):
    student_id: str = "local-student"
    problem: str = Field(min_length=1, max_length=12000)
    student_attempt: str = ""
    skill: str = "general"
    hint_level: int = Field(default=0, ge=0, le=3)

@app.post("/v2/socratic")
def socratic(request: SocraticRequest) -> dict:
    step = next_socratic_step(
        problem=request.problem,
        student_attempt=request.student_attempt,
        skill=request.skill,
        hint_level=request.hint_level,
    )
    return step.__dict__

@app.post("/v2/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    thread_id = request.thread_id or str(uuid4())
    metrics = new_trace()
    try:
        state = make_initial_state(
            student_id=request.student_id,
            thread_id=thread_id,
            raw_text=request.problem,
        )
        with active_trace(metrics):
            result = chatbot.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}},
            )
        trace = metrics.finish("ok")
        trace_store.append({
            **trace,
            "student_id": request.student_id,
            "thread_id": thread_id,
            "execution_tier": (result.get("execution_plan") or {}).get("tier"),
        })
        return SolveResponse(
            thread_id=thread_id,
            execution_plan=result.get("execution_plan"),
            final_response=result.get("final_response"),
            status="ok" if result.get("final_response") else "needs_follow_up",
        )
    except Exception as exc:
        trace_store.append({**metrics.finish("error", str(exc)), "student_id": request.student_id, "thread_id": thread_id})
        raise HTTPException(status_code=500, detail="MathTutor request failed; inspect trace_id for diagnostics.") from exc
