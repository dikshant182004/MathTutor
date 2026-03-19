#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  entrypoint.sh — JEE Math Tutor
#
#  What this script does (in order):
#    1. Loads .env into the shell environment
#    2. Waits until Redis is reachable (up to 30 s)
#    3. Starts the Manim MCP server in the background (optional, skipped if
#       SKIP_MANIM=1 is set or the venv doesn't have manim installed)
#    4. Starts the Streamlit app in the foreground
#
#  Usage:
#    chmod +x entrypoint.sh
#    ./entrypoint.sh
#
#  Skip Manim:
#    SKIP_MANIM=1 ./entrypoint.sh
#
#  Custom port:
#    STREAMLIT_PORT=8502 ./entrypoint.sh
#
#  Requirements:
#    • docker compose up -d redis   (or Redis already running)
#    • pip install -r requirements.txt already done in your venv
#    • cd to the repo root before running (so paths are correct)
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${CYAN}[entrypoint]${NC} $*"; }
ok()   { echo -e "${GREEN}[entrypoint] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[entrypoint] ⚠${NC} $*"; }
die()  { echo -e "${RED}[entrypoint] ✗${NC} $*"; exit 1; }

# ── Step 0: banner ────────────────────────────────────────────────────────────
echo -e "${BOLD}"
echo "  ╔══════════════════════════════════════╗"
echo "  ║       JEE Math Tutor — Startup       ║"
echo "  ╚══════════════════════════════════════╝"
echo -e "${NC}"

# ── Step 1: load .env ─────────────────────────────────────────────────────────
ENV_FILE="${ENV_FILE:-.env}"
if [ -f "$ENV_FILE" ]; then
    log "Loading environment from $ENV_FILE"
    # Export every non-comment, non-blank line
    set -o allexport
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +o allexport
    ok "Environment loaded"
else
    warn ".env not found at $ENV_FILE — relying on existing environment variables"
fi

# ── Step 2: verify required variables ─────────────────────────────────────────
REQUIRED_VARS=(
    GROQ_API_KEY
    COHERE_API_KEY
    TAVILY_API_KEY
    REDIS_URL
    GOOGLE_CLIENT_ID
    GOOGLE_CLIENT_SECRET
    OAUTH_REDIRECT_URI
)

MISSING=0
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var:-}" ]; then
        warn "Required variable not set: $var"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    die "$MISSING required variable(s) missing. Check your .env file."
fi
ok "All required environment variables present"

# Check at least one Vision credential is set (warn only — OCR is optional)
if [ -z "${GOOGLE_CREDENTIALS_JSON:-}" ] && [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    warn "Neither GOOGLE_CREDENTIALS_JSON nor GOOGLE_APPLICATION_CREDENTIALS is set."
    warn "Image (OCR) input will not work."
fi

# ── Step 3: wait for Redis ────────────────────────────────────────────────────
log "Waiting for Redis at ${REDIS_URL} ..."

# Extract host:port from REDIS_URL for redis-cli
# Handles: redis://:pass@host:port  and  rediss://user:pass@host:port
REDIS_HOST=$(echo "$REDIS_URL" | sed -E 's|rediss?://[^@]*@([^:]+):.*|\1|')
REDIS_PORT=$(echo "$REDIS_URL" | sed -E 's|rediss?://[^@]*@[^:]+:([0-9]+).*|\1|')
REDIS_PASS=$(echo "$REDIS_URL" | sed -E 's|rediss?://[^:]*:([^@]*)@.*|\1|')

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"

WAIT=0
MAX_WAIT=30
until redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ${REDIS_PASS:+-a "$REDIS_PASS"} ping \
        2>/dev/null | grep -q PONG; do
    if [ $WAIT -ge $MAX_WAIT ]; then
        die "Redis did not respond after ${MAX_WAIT}s. Is it running? Try: docker compose up -d redis"
    fi
    echo -n "."
    sleep 1
    WAIT=$((WAIT + 1))
done
echo ""
ok "Redis is up (${REDIS_HOST}:${REDIS_PORT})"

# ── Step 4: create required directories ──────────────────────────────────────
mkdir -p uploads manim_outputs logs
ok "Directories ready (uploads/, manim_outputs/, logs/)"

# ── Step 5: start Manim MCP server (background) ───────────────────────────────
MANIM_PID=""

if [ "${SKIP_MANIM:-0}" = "1" ]; then
    warn "SKIP_MANIM=1 — skipping Manim MCP server"
else
    MANIM_SCRIPT="src/backend/agents/nodes/tools/mcp/manim_mcp_server.py"

    if [ ! -f "$MANIM_SCRIPT" ]; then
        warn "Manim MCP script not found at $MANIM_SCRIPT — skipping"
    elif ! python -c "import manim" 2>/dev/null; then
        warn "manim not installed in current Python env — skipping Manim MCP"
        warn "Install with: pip install manim fastmcp"
    elif ! python -c "import fastmcp" 2>/dev/null; then
        warn "fastmcp not installed — skipping Manim MCP"
        warn "Install with: pip install fastmcp"
    else
        MANIM_PORT="${MANIM_SERVER_PORT:-8765}"
        log "Starting Manim MCP server on port $MANIM_PORT ..."
        MANIM_OUTPUT_DIR="${MANIM_OUTPUT_DIR:-./manim_outputs}" \
        MANIM_SERVER_PORT="$MANIM_PORT" \
        MANIM_MCP_SERVER_URL="${MANIM_MCP_SERVER_URL:-http://localhost:${MANIM_PORT}/mcp}" \
            python "$MANIM_SCRIPT" >> logs/manim_mcp.log 2>&1 &
        MANIM_PID=$!

        # Give it 3 seconds to start
        sleep 3
        if kill -0 "$MANIM_PID" 2>/dev/null; then
            ok "Manim MCP server started (PID $MANIM_PID, port $MANIM_PORT)"
            ok "Logs: logs/manim_mcp.log"
        else
            warn "Manim MCP server exited immediately — check logs/manim_mcp.log"
            MANIM_PID=""
        fi
    fi
fi

# ── Step 6: trap for clean shutdown ───────────────────────────────────────────
cleanup() {
    echo ""
    log "Shutting down..."
    if [ -n "$MANIM_PID" ] && kill -0 "$MANIM_PID" 2>/dev/null; then
        log "Stopping Manim MCP server (PID $MANIM_PID)..."
        kill "$MANIM_PID"
        wait "$MANIM_PID" 2>/dev/null || true
        ok "Manim MCP server stopped"
    fi
    ok "Goodbye."
}
trap cleanup EXIT INT TERM

# ── Step 7: start Streamlit ───────────────────────────────────────────────────
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
APP_PATH="${APP_PATH:-src/frontend/app.py}"

if [ ! -f "$APP_PATH" ]; then
    die "Streamlit app not found at $APP_PATH. Are you running from the repo root?"
fi

# Verify secrets.toml exists (Streamlit will start without it but OAuth will fail)
SECRETS_FILE=".streamlit/secrets.toml"
if [ ! -f "$SECRETS_FILE" ]; then
    warn "No .streamlit/secrets.toml found."
    warn "Copy and fill in: cp .streamlit/secrets.toml.example .streamlit/secrets.toml"
    warn "Continuing anyway — st.secrets will fall back to environment variables."
fi

log "Starting Streamlit on http://localhost:${STREAMLIT_PORT} ..."
echo ""
echo -e "${BOLD}  App URL:  ${GREEN}http://localhost:${STREAMLIT_PORT}${NC}"
if [ -n "$MANIM_PID" ]; then
    echo -e "${BOLD}  Manim MCP: ${GREEN}http://localhost:${MANIM_SERVER_PORT:-8765}/mcp${NC}"
fi
echo -e "${BOLD}  Redis UI:  ${GREEN}http://localhost:8001${NC}  (if RedisInsight is running)"
echo ""

exec streamlit run "$APP_PATH" \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false \
    --theme.base dark