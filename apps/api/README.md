# MathTutor V2 API

FastAPI adapter around the existing LangGraph runtime.

Run from the repository root with the project environment installed:

`uvicorn apps.api.main:app --reload --port 8000`

Endpoints:
- GET `/health`
- POST `/v2/plan`
- POST `/v2/solve`

The API intentionally reuses the existing graph/checkpointer rather than duplicating agent logic.