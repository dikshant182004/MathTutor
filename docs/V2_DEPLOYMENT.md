# MathTutor V2 deployment

## Local
```powershell
$env:GROQ_API_KEY="..."
docker compose -f docker-compose.v2.yml up --build
```

API: `http://localhost:8000`
Health: `GET /health`
Web: run `npm install && npm run dev` in `apps/web`.

## Production shape
- Next.js frontend: deploy separately with `NEXT_PUBLIC_API_URL` pointing at the API.
- FastAPI/LangGraph API: run as a long-lived Python service.
- Redis: durable managed Redis for LangGraph checkpoints, student memory and document metadata.
- Secrets: inject provider keys through the deployment platform; never commit them.
- Set `V2_API_KEY` for service-level API protection when appropriate.
- Set `CORS_ORIGINS` to exact frontend origins, not `*`.

The Python agent is intentionally not coupled to a specific hosting vendor. Cloudflare can front the API and host the web surface while the Python service remains on a Python-compatible runtime.
