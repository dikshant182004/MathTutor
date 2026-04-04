#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
#  entrypoint.sh — JEE Math Tutor
#
#  What this script does (in order):
#    1. Loads .env into the shell environment
#    2. Waits until Redis is reachable (up to 30 s)
#    3. Starts the Streamlit app in the foreground
#
#  Usage:
#    chmod +x entrypoint.sh
#    ./entrypoint.sh
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

    # Export each KEY=value line safely
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines and comments
        [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue

        # Export the variable
        if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
            key="${line%%=*}"
            value="${line#*=}"
            export "$key=$value"
            # Show masked value for logging
            if [ ${#value} -gt 12 ]; then
                masked="${value:0:6}...${value: -4}"
            else
                masked="$value"
            fi
            log "  Set: $key=${masked}"
        fi
    done < "$ENV_FILE"

    ok "Environment variables loaded"
else
    warn ".env file not found at $ENV_FILE"
fi

# ── Step 2: verify required variables ─────────────────────────────────────────
REQUIRED_VARS=(
    REDIS_URL
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

# ── Optional: run tests only and exit ─────────────────────────────────────────
if [ "${RUN_TESTS:-0}" = "1" ]; then
    log "RUN_TESTS=1 detected — running pytest and exiting"
    exec pytest
fi

# ── Step 3: wait for Redis ────────────────────────────────────────────────────
log "Waiting for Redis at ${REDIS_URL} ..."

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

# ── Step 3.5: Check RedisInsight UI (Optional & Non-blocking) ─────────────────
log "Checking RedisInsight UI (optional)..."

INSIGHT_AVAILABLE=false

# Check both common ports
if curl -s -f http://localhost:5540 >/dev/null 2>&1; then
    INSIGHT_AVAILABLE=true
    ok "RedisInsight UI detected on port 5540 (recommended)"
elif curl -s -f http://localhost:8001 >/dev/null 2>&1; then
    INSIGHT_AVAILABLE=true
    ok "RedisInsight UI detected on port 8001 (built-in)"
else
    warn "RedisInsight UI is not running (this is optional)"
    echo -e "${YELLOW}   Tip: Start it with → docker compose up -d redisinsight${NC}"
fi

if [ "$INSIGHT_AVAILABLE" = true ]; then
    echo -e "${GREEN}   → Open RedisInsight at: http://localhost:5540${NC}"
fi

# ── Step 4: create required directories ──────────────────────────────────────
mkdir -p uploads logs
ok "Directories ready (uploads/, logs/)"

# ── Step 5: trap for clean shutdown ───────────────────────────────────────────
cleanup() {
    echo ""
    log "Shutting down..."
    ok "Goodbye."
}
trap cleanup EXIT INT TERM

# ── Step 6: start Streamlit ───────────────────────────────────────────────────
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

export PYTHONPATH="$(pwd):$(pwd)/src:${PYTHONPATH:-}"

log "Starting Streamlit on http://localhost:${STREAMLIT_PORT} ..."

echo ""
echo -e "${BOLD}  App URL:           ${GREEN}http://localhost:${STREAMLIT_PORT}${NC}"
echo -e "${BOLD}  Redis server:      ${GREEN}${REDIS_URL}${NC}"
echo -e "${BOLD}  RedisInsight UI:   ${GREEN}http://localhost:5540${NC} (if running)"
echo ""

exec streamlit run "$APP_PATH" \
    --server.port "$STREAMLIT_PORT" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.fileWatcherType none \
    --browser.gatherUsageStats false \
    --theme.base dark