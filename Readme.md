# MathTutor — Agentic JEE Math Tutor

MathTutor is a full-stack tutoring application built around a LangGraph workflow, a FastAPI API, and a Next.js App Router frontend. The frontend lives in `src/frontend/`; the Python backend and tests live under `src/backend/` and `src/tests/`.

> **Status note:** this repository is in an active migration. The files listed below exist in the branch, but their presence alone does not mean every feature has passed an end-to-end or production validation. See [the cleanup and gap map](docs/CLEANUP_AUDIT.md) and [the verification log](docs/VERIFIED.md).

## Current layout

- `src/frontend/` — Next.js, React, TypeScript UI.
- `src/backend/api/` — FastAPI application, authentication, threads, chat/SSE, HITL resume, documents, and persistence.
- `src/backend/agents/` — LangGraph graph/state, intake and routing, solving, verification, safety, memory, tools, and runtime telemetry.
- `src/backend/llm/` — allowlisted model profiles, registry, and routing/fallback logic.
- `src/tests/` — unit and integration tests.
- `src/backend/api/migrations/` — SQL migrations.
- `.github/workflows/tests.yml` — backend test job and frontend lint/build job.

## Prerequisites

- Python 3.14 for the currently configured CI workflow (confirm dependency compatibility in your target environment).
- Node.js 20 or newer.
- Docker Compose for local Postgres and Redis.
- Google OAuth Web client ID.
- Provider keys required for the enabled features, stored only in backend environment variables.

## Local development (Windows PowerShell)

### 1. Start infrastructure

From the repository root:

```powershell
docker compose up -d postgres redis
```

### 2. Start the API

Install the API dependencies in your chosen virtual environment, then configure the required environment variables. Do not use real secrets in committed files.

```powershell
python -m pip install -r requirements-api.txt
$env:PYTHONPATH = "src"
$env:DATABASE_URL = "postgresql://mathtutor:mathtutor@localhost:5432/mathtutor"
$env:REDIS_URL = "redis://localhost:6379"
$env:JWT_SECRET = "replace-with-a-random-secret-at-least-32-characters"
$env:GOOGLE_CLIENT_ID = "your-google-client-id.apps.googleusercontent.com"
$env:ALLOWED_ORIGINS = "http://localhost:3000"
python -m uvicorn backend.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

You must also set the provider credentials needed by the configured model and document-retrieval paths. See `.env.example` for variable names; it contains placeholders, not working credentials.

### 3. Start the frontend

In a second terminal, from the repository root:

```powershell
Set-Location src/frontend
Copy-Item .env.example .env.local
# Set NEXT_PUBLIC_GOOGLE_CLIENT_ID in .env.local.
# Set NEXT_PUBLIC_API_BASE_URL if the API is not at http://127.0.0.1:8000.
npm install
npm run dev
```

Open http://localhost:3000. The frontend's full setup instructions are in [src/frontend/README.md](src/frontend/README.md).

## Validation

Backend tests:

```powershell
python -m pip install -r requirements-test.txt
$env:DATABASE_URL = "postgresql://mathtutor:mathtutor@localhost:5432/mathtutor"
python -m pytest
```

Frontend checks, from `src/frontend/`:

```powershell
npm install
npm run lint
npm run build
```

CI currently runs the Python tests and the frontend lint/build checks. The frontend package does not yet include a committed npm lockfile, so CI uses `npm install`; add and commit a lockfile before switching CI to `npm ci`.

## API surface currently registered

The FastAPI app registers routers for Google authentication, model profiles, threads/history, chat streaming, HITL resume, and student PDF documents. Do not assume unregistered analytics, feedback, preferences, export/deletion, or memory-graph endpoints exist; see the audit map.

## Important migration notes

- Streamlit has been removed as the UI. The root Python dependency file no longer installs Streamlit, and the launch scripts now target the FastAPI backend.
- The legacy retrieval and memory modules are still present in the repository. They are not to be deleted solely because they look old: trace imports, graph reachability, tests, and runtime use first.
- The Next.js frontend is at `src/frontend/`; CI and setup docs use that path.
- Local Compose credentials are development defaults only. Replace them for any shared or deployed environment.
- The API's `/health` route currently reports process health; it should not be treated as proof that all external dependencies and provider calls are healthy.

## Further documentation

- [Frontend setup](src/frontend/README.md)
- [Phase F/G implementation notes](docs/PHASE_F_G.md)
- [Dependency decisions](docs/DEPENDENCIES.md)
- [Verification log](docs/VERIFIED.md)
- [Cleanup and gap map](docs/CLEANUP_AUDIT.md)
