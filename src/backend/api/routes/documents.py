"""Authenticated endpoints for durable, per-student PDF notes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.api.deps import get_current_student, get_redis
from backend.agents.nodes.tools.persistent_documents import (
    MAX_PDF_BYTES,
    delete_document,
    ingest_pdf,
    list_documents,
)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
def get_documents(
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
) -> dict:
    return {"documents": list_documents(redis, student_id)}


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
) -> dict:
    filename = (file.filename or "document.pdf").replace("\\", "/").split("/")[-1]
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail={
            "code": "unsupported_document_type",
            "message": "Only PDF documents are supported.",
            "retryable": False,
        })
    data = await file.read(MAX_PDF_BYTES + 1)
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(status_code=413, detail={
            "code": "document_too_large",
            "message": "PDF exceeds the 10 MiB upload limit.",
            "retryable": False,
        })
    try:
        document = ingest_pdf(redis, student_id, filename, data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={
            "code": "invalid_document",
            "message": str(exc),
            "retryable": False,
        }) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail={
            "code": "document_indexing_unavailable",
            "message": str(exc),
            "retryable": True,
        }) from exc
    return {"document": document}


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(
    document_id: str,
    student_id: str = Depends(get_current_student),
    redis=Depends(get_redis),
) -> None:
    if not delete_document(redis, student_id, document_id):
        # Do not reveal whether a document belonging to another student exists.
        raise HTTPException(status_code=404, detail={
            "code": "document_not_found",
            "message": "Document not found.",
            "retryable": False,
        })
