#!/usr/bin/env bash
# Container/runtime entrypoint for the FastAPI backend.
# Start Postgres and Redis separately; this script does not start infrastructure.

set -euo pipefail

: "${DATABASE_URL:?DATABASE_URL must be configured}"
: "${REDIS_URL:?REDIS_URL must be configured}"
: "${JWT_SECRET:?JWT_SECRET must be configured}"
: "${GOOGLE_CLIENT_ID:?GOOGLE_CLIENT_ID must be configured}"
: "${ALLOWED_ORIGINS:?ALLOWED_ORIGINS must be configured}"

export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

if [ "${RUN_TESTS:-0}" = "1" ]; then
    echo "[entrypoint] RUN_TESTS=1; running pytest"
    exec pytest
fi

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
echo "[entrypoint] Starting MathTutor API on ${HOST}:${PORT}"
exec python -m uvicorn backend.api.main:app --app-dir src --host "$HOST" --port "$PORT"
