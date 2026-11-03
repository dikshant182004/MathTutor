# Start the MathTutor FastAPI backend from the repository root.
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = Join-Path $PSScriptRoot "src"

if (-not $env:DATABASE_URL) {
    $env:DATABASE_URL = "postgresql://mathtutor:mathtutor@localhost:5432/mathtutor"
}
if (-not $env:REDIS_URL) {
    $env:REDIS_URL = "redis://localhost:6379"
}

Write-Host "Starting MathTutor API at http://127.0.0.1:8000"
Write-Host "Start the Next.js frontend separately from src/frontend/ (see src/frontend/README.md)."
python -m uvicorn backend.api.main:app --app-dir (Join-Path $PSScriptRoot "src") --host 127.0.0.1 --port 8000
