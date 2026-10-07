# MathTutor V2 Runbook

Backend: run FastAPI from the repository with PYTHONPATH=src on port 8000.
Frontend: run the Next.js app in apps/web on port 3000 with NEXT_PUBLIC_API_URL
pointing at the backend.

V2 APIs:
- GET /health
- POST /v2/plan
- POST /v2/solve
- POST /v2/socratic
- POST /v2/documents/text
- GET /v2/documents/{student_id}
- GET /v2/students/{student_id}/snapshot
- GET /v2/students/{student_id}/graph
- GET /v2/students/{student_id}/mistakes
- GET /v2/students/{student_id}/next-problem
- GET /v2/students/{student_id}/practice
- GET /v2/observability/traces

Redis is the recommended persistent backend. Student memory and documents are
student-scoped; local JSON fallback is used for development when Redis is absent.

The model gateway supports Groq, OpenRouter, OpenAI, Anthropic, Ollama and custom
OpenAI-compatible endpoints through environment configuration.

Before merge, run the complete Python test suite, build the web app, then manually
exercise solve, Socratic tutoring, document retrieval, mastery, Mistake Lab,
Practice, Knowledge Graph and real observability traces.

Do not merge V2 until the local end-to-end validation is complete.
