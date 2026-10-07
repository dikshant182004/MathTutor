# MathTutor V2 deployment

## Local
Set GROQ_API_KEY and run:
docker compose -f docker-compose.v2.yml up --build

API: http://localhost:8000
Health: GET /health
Web: run npm install && npm run dev in apps/web.

## Production shape
- Deploy the Next.js frontend separately with NEXT_PUBLIC_API_URL pointing at the API.
- Run the FastAPI/LangGraph API on a long-lived Python-compatible service.
- Use durable managed Redis for LangGraph checkpoints, student memory and document metadata.
- Inject provider keys through the deployment secret manager.
- Set V2_API_KEY for trusted service/admin protection when appropriate.
- Set CORS_ORIGINS to exact frontend origins.

The Python agent is intentionally vendor-neutral. Cloudflare can front the web/API surface while the Python service remains on a Python-compatible runtime.
