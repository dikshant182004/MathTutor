"""Expose only the model profiles explicitly allowed by the model registry."""
from fastapi import APIRouter, Depends

from backend.api.deps import get_current_student
from backend.llm.profiles import public_profiles

router = APIRouter(tags=["models"])


@router.get("/models")
def list_models(_student_id: str = Depends(get_current_student)) -> dict:
    return {"profiles": public_profiles()}
