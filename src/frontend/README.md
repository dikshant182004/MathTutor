# MathTutor web frontend

The Next.js App Router frontend is located at `src/frontend/`. It communicates with the FastAPI backend and never connects directly to Redis, Postgres, or LLM providers.

## Requirements

- Node.js 20 or newer
- The MathTutor API and local Postgres/Redis services running
- A Google OAuth Web client ID configured for both frontend and API

## Local setup (Windows PowerShell)

Run these commands from the repository root.

Start Postgres and Redis:

```powershell
docker compose up -d postgres redis
```

In the first PowerShell terminal, configure the API. Use your own Google OAuth client ID and a randomly generated JWT secret of at least 32 characters. Keep provider API keys on the backend only.

```powershell
$env:DATABASE_URL = "postgresql://mathtutor:mathtutor@localhost:5432/mathtutor"
$env:REDIS_URL = "redis://localhost:6379"
$env:JWT_SECRET = "replace-this-with-a-random-secret-at-least-32-characters"
$env:GOOGLE_CLIENT_ID = "your-google-client-id.apps.googleusercontent.com"
$env:ALLOWED_ORIGINS = "http://localhost:3000"
$env:COHERE_API_KEY = "your-cohere-api-key"
$env:PYTHONPATH = "src"
python -m uvicorn backend.api.main:app --app-dir src --host 127.0.0.1 --port 8000
```

In a second PowerShell terminal, run the frontend from its actual directory:

```powershell
Set-Location src/frontend
Copy-Item .env.example .env.local
# Edit .env.local and set NEXT_PUBLIC_GOOGLE_CLIENT_ID to the same OAuth Web client ID.
npm install
npm run dev
```

Open http://localhost:3000. In Google Cloud Console, add `http://localhost:3000` to the OAuth client's **Authorized JavaScript origins**. The API's `GOOGLE_CLIENT_ID` must match the frontend client ID.

## Current frontend flows

- Google Identity Services sign-in and API JWT exchange
- Student-scoped conversation list and new-thread creation
- Streaming assistant tokens and solution events over Server-Sent Events
- Human-in-the-loop interrupt display and resume with an idempotency key
- PDF notes upload, listing, and deletion
- API errors surfaced to the user without exposing server tracebacks

The browser keeps the API access token in tab-scoped `sessionStorage`, not `localStorage`. API keys for Groq, Cohere, and other providers belong only on the backend.

## Production

Set `NEXT_PUBLIC_API_BASE_URL` to the deployed API origin and configure the API's `ALLOWED_ORIGINS` to include the deployed frontend origin. Set `NEXT_PUBLIC_GOOGLE_CLIENT_ID` at build time. Use HTTPS for both services and a production-grade secret manager for server credentials.

## Build checks

From `src/frontend/`:

```powershell
npm install
npm run lint
npm run build
```

The repository currently has no npm lockfile; CI uses `npm install` until a lockfile is committed.
