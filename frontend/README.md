# MathTutor web frontend

The Next.js frontend lives in this directory and talks to the existing FastAPI API. It does not access Redis, Postgres, or LLM providers directly.

## Requirements

- Node.js 20 or newer
- MathTutor API running locally
- Google OAuth Web client ID configured for the frontend and API

## Local setup (Windows PowerShell)

From the repository root, start the data services if they are not already running:

```powershell
docker compose up -d postgres redis
```

Configure the API environment in the same PowerShell session. Use a random `JWT_SECRET` with at least 32 characters and keep all provider credentials out of frontend variables:

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

In a second PowerShell terminal:

```powershell
cd frontend
Copy-Item .env.example .env.local
# Edit .env.local and set NEXT_PUBLIC_GOOGLE_CLIENT_ID to the same OAuth Web client ID.
npm install
npm run dev
```

Open http://localhost:3000. In Google Cloud Console, add `http://localhost:3000` to the OAuth client's **Authorized JavaScript origins**. The API's `GOOGLE_CLIENT_ID` must match the frontend client ID.

## Included flows

- Google Identity Services sign-in and API JWT exchange
- Student-scoped conversation list and new-thread creation
- Streaming assistant tokens and solution events over Server-Sent Events
- HITL interrupt display and resume with an idempotency key
- PDF notes upload, listing, and deletion
- API errors surfaced to the user without exposing server tracebacks

The browser keeps the API access token in `sessionStorage`, not localStorage. API keys for Groq, Cohere, and other providers belong only on the backend.

## Production

Set `NEXT_PUBLIC_API_BASE_URL` to the deployed API origin and configure the API's `ALLOWED_ORIGINS` to include the deployed frontend origin. Set `NEXT_PUBLIC_GOOGLE_CLIENT_ID` at build time. Use HTTPS for both services and a production-grade secret manager for server credentials.
