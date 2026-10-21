# Verification log

Last checked: 2026-10-12 against official provider documentation.

## Vercel FastAPI entrypoint, Python runtime and request lifecycle

- FastAPI detection: export a FastAPI instance named app at a supported entrypoint such as app.py, index.py, server.py, main.py, wsgi.py, or asgi.py at the project root or under src/, app/, or api/. A custom module can be declared with tool.vercel.entrypoint in pyproject.toml. Source: https://vercel.com/docs/frameworks/backend/fastapi and https://vercel.com/kb/guide/ship-a-fastapi-app-on-vercel
- Supported Python versions: 3.12 (default), 3.13, and 3.14. The project instructions choose Python 3.12 as its baseline. Source: https://vercel.com/docs/functions/runtimes/python
- Function request/response body limit: 4.5 MB; oversized payloads return 413. Source: https://vercel.com/docs/functions/limitations and https://vercel.com/docs/errors/function_payload_too_large
- Continuing work after a response: Vercel documents waitUntil for Node.js and Edge, and Next.js after() for Next.js handlers. The Python runtime page does not document an equivalent post-response continuation API. Per the implementation instruction, assume background work must not continue after the response; required work must be awaited before completing the response. Sources: https://vercel.com/docs/functions/functions-api-reference and https://vercel.com/docs/functions/runtimes/python

## Redis Cloud free tier and eviction

- The Redis Cloud free plan provides one free database with 30 MB of space. Source: https://redis.io/docs/latest/operate/rc/databases/create-database/create-free-database/
- The data eviction policy is configurable in the database settings for Essentials plans; volatile-lru is an available policy and is the documented default. It evicts only keys that have an expiry. Sources: https://redis.io/docs/latest/operate/rc/databases/configuration/data-eviction-policies/ and https://redis.io/docs/latest/operate/rc/databases/view-edit-database/
- The implementation must not assume that no-TTL semantic, procedural, or document keys survive under all eviction configurations unless volatile-lru is selected. Confirm the actual console setting when provisioning the database.

## Neon free tier

- As of Neon's October 2, 2026 announcement, the Free plan provides 100 projects, 1 GB of Postgres storage per project, 100 CU-hours per project per month, autoscaling up to 2 CU, 10 branches per project, and 5 GB of public network transfer per project per month. Source: https://neon.com/blog/neon-free-plan-1-gb-per-project
- Recheck the live plan documentation before provisioning because free-tier limits can change. Supabase was not selected or independently audited in this phase.

## Groq model capabilities and rate limits

- Groq documents openai/gpt-oss-20b and openai/gpt-oss-120b as supporting strict JSON-schema structured output, tool calling, and reasoning_effort values low, medium, and high. Source: https://console.groq.com/docs/structured-outputs and https://console.groq.com/docs/api-reference
- Groq's published base free-plan table lists 30 RPM, 1,000 RPD, 8K TPM, and 200K TPD for GPT-OSS 20B and 120B; the safeguard model is listed with lower request/token limits. Actual organization limits must be checked in the account console. Source: https://console.groq.com/docs/rate-limits

## Redis LangGraph checkpoint TTL

- LangGraph Redis checkpoint savers and stores support a TTL configuration with default_ttl in minutes and refresh_on_read. Source: https://github.com/redis-developer/langgraph-redis
- This confirms the API exists; the exact installed version is pinned and import-tested by CI, while the 24-hour checkpoint TTL is configured in the relevant implementation phase.


## MathTutor API authentication, HITL resume and checkpoint lifetime

- The API requires short-lived HS256 JWTs, validates thread ownership before reading graph state, and applies Redis-backed per-IP and per-student rate limits.
- HITL resume requires an `Idempotency-Key`, the current `interrupt_id`, a decision compatible with the interrupt type, and a response payload. A Redis NX lock serializes resume execution per thread; completed idempotent requests replay their cached SSE events.
- The LangGraph Redis checkpointer is configured with `ttl={"default_ttl": 120, "refresh_on_read": True}`; the API's checkpoint metadata lifetime is aligned to 2 hours. The Redis saver refreshes checkpoint expiry on reads. This configuration is covered by a unit test; actual expiry behavior against a provisioned Redis service still requires a deployment smoke test.
