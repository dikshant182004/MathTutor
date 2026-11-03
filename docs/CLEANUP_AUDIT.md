# Cleanup and implementation gap map

Audit scope: the tracked tree on branch `Agent` after the frontend move to `src/frontend/`. This is a static repository review, not a successful local test run or a claim that every runtime path was exercised. Items are classified as **fix now**, **verify before removing**, or **missing / incomplete** to avoid deleting code that may still be reachable.

## 1. Fixed in this migration update

| Area | Previous mismatch | Change made |
|---|---|---|
| GitHub Actions | Frontend job ran in `frontend/` | `.github/workflows/tests.yml` now runs from `src/frontend/`. |
| Node setup | npm cache configured without a lockfile | Removed the cache configuration that depended on a lockfile. CI uses `npm install` until a lockfile is committed. |
| Frontend setup guide | Instructions used `cd frontend` and `frontend/.env.example` | Updated to `src/frontend/`. |
| Phase F/G notes | Documentation described the app under `frontend/` | Updated the frontend path references to `src/frontend/`. |
| Root README | Described Streamlit auth and old UI as current, and called Next.js a later phase | Replaced with a migration-aware overview and current paths. |
| `run.ps1` | Launched deleted `src/frontend/app.py` via Streamlit | Now starts the FastAPI backend and points to the Next.js setup guide. |
| `entrypoint.sh` | Started Streamlit and checked Streamlit secrets | Now starts the FastAPI backend; requires deployment environment variables. |
| `requirements.txt` | Installed Streamlit | Removed the retired Streamlit dependency. |
| `.env.example` | Contained token-shaped placeholder strings and a Streamlit OAuth callback URL | Replaced token-shaped values with blanks/placeholders and removed the obsolete callback setting. |
| `.gitignore` | Ignored build output under the old root `frontend/` path and retained Streamlit-specific rules | Updated Next.js ignore paths to `src/frontend/` and removed the obsolete Streamlit section. |
| `.devcontainer/devcontainer.json` | Installed Streamlit and opened the deleted `src/frontend/app.py` on port 8501 | Updated the devcontainer to Python/Node tooling, install current dependencies, launch Next.js, and forward ports 3000/8000. |

These are committed directly to `Agent`. They still need a fresh CI run to validate the combined result.

## 2. Immediate validation tasks

| Priority | File / area | What to verify |
|---|---|---|
| P0 | `.github/workflows/tests.yml` | Confirm Python tests pass, then frontend lint and production build pass with the new working directory. |
| P0 | `src/frontend/package.json` | There is no tracked npm lockfile in the current tree. Generate and commit `src/frontend/package-lock.json`; after that, switch CI from `npm install` to `npm ci` and configure npm caching against the lockfile. |
| P0 | `src/frontend/app/page.tsx` + API routers | Run browser/API integration tests for sign-in, thread creation/history, streaming, HITL resume, and PDF upload/list/delete. A successful Next.js build alone does not prove these work against the API. |
| P1 | `run.ps1`, `entrypoint.sh`, `src/backend/api/main.py` | Test startup with real Postgres and Redis. The API startup checks Postgres; Redis availability, graph checkpointing, and provider access must also be tested through the relevant operations. |
| P1 | `.env.example` and `src/tests/unit/test_env_example_keys.py` | Ensure every documented variable matches current code usage. The test currently requires `GOOGLE_CLIENT_SECRET`, but the API's Google ID-token verification appears to use `GOOGLE_CLIENT_ID`; confirm whether the secret is truly needed before removing the key or changing the test. |
| P1 | Python runtime/dependencies | CI uses Python 3.14. Run dependency installation and `pip check` under the intended deployment runtime, not only unit tests. |

## 3. Missing or incomplete application capabilities

These are gaps against the planned full application, not automatic cleanup candidates.

| Area | Current evidence | Recommended action |
|---|---|---|
| API endpoint coverage | `src/backend/api/main.py` registers auth, models, threads, chat, HITL, and documents routers. No analytics, memory-graph read/write, feedback, profile/preferences, or export/account-deletion router is registered. | Compare each endpoint to the requirements document; implement only required flows and add authorization/integration tests. Do not build endpoints just because a frontend mock expects them. |
| Document retrieval limits | `docs/PHASE_F_G.md` describes 10 MiB uploads and up to 25 documents per student; this differs from the earlier target limits in the upgrade plan. | Reconcile implementation against the source instruction PDF. Decide the authoritative size, page, document, and chunk limits, then encode them in validation and tests. |
| Retrieval strategy | The documented student document store uses Redis hashes and application-side cosine scoring. It combines with legacy per-thread FAISS/BM25 retrieval when available. | Confirm whether the intended architecture requires vector-index top-k plus sparse retrieval and reciprocal-rank fusion. If so, implement and evaluate it before removing either path. |
| Citation quality | Document metadata stores page/chunk information, but end-to-end page-aware citation behavior has not been established by this static review. | Add tests proving answers cite the correct uploaded document/page and do not leak citations or chunks across students. |
| Memory graph UI/API | The current registered API routers do not expose a memory-graph endpoint. | Trace the actual required graph visualization design. Add a real backend endpoint and real persisted data flow only if required; do not add mock graph nodes. |
| Observability / analytics | A persistence migration includes conversation history and run analytics, but no analytics router is registered. | Decide which metrics are needed in the product and expose only supported, student-scoped data. |
| Health checks | `/health` returns `{"status":"ok"}` without checking dependency readiness. | Keep it as liveness or add a separate readiness check that validates Postgres/Redis appropriately without exposing secrets. |
| Deployment parity | The repository has Compose for Postgres/Redis and an API entrypoint, but no tracked frontend lockfile or complete deployment manifest for the full web/API split. | Document and test frontend/API deployment independently, including CORS, HTTPS, secrets, persistence, migrations, and rollback. |

## 4. Potentially redundant or legacy code — do not delete until traced

| Candidate | Why it needs review | Safe decision process |
|---|---|---|
| `requirements.txt` vs `requirements-api.txt` vs `requirements-test.txt` | Multiple dependency lists can drift. `requirements.txt` still contains older RAG/OCR and broad runtime dependencies, while the API file is narrower. | Inventory imports and runtime entry points; establish one canonical API dependency set plus a test/dev overlay. Keep packages still imported by the active graph. |
| `src/notebooks/rag_tool.ipynb` | Notebook may be exploratory or document an older retrieval pipeline. | Search for production imports/calls and retain if it is a needed evaluation/reference artifact; otherwise move to an explicitly archival location or remove after review. |
| `src/backend/agents/nodes/verify_math.py` | Tiny compatibility shim re-exports `verify_math` from `backend.agents.verification`. | Search all imports and external entry points; delete only if no supported imports depend on it. |
| `src/backend/agents/utils/___init__.py` | Empty file has three underscores and is not the normal package initializer name. | Check imports and packaging behavior. If accidental, remove it; it does not replace `__init__.py`. |
| Legacy node modules such as `input.py`, `parser.py`, `router.py`, `explainer.py`, `guardrail.py`, `safety.py`, and `compose.py` | Newer prefilter/intake/routing code coexists with older node implementations. The graph and compatibility imports determine which are live. | Build a module import/call graph from `graph.py`, tests, and API entry points; compare to the required agent topology. Remove only unreachable duplicate paths after coverage tests. |
| `src/backend/agents/nodes/tools/tools.py` and `persistent_documents.py` | Legacy RAG and newer persistent student-document retrieval coexist. | Identify whether both are intentionally composed. Keep the intended retrieval path, measure quality/latency, and delete duplicate indexing or retrieval only after migration tests pass. |
| OCR / audio / vision dependencies | Root dependencies include Google Cloud Vision and Groq audio-related packages; the current Next.js page and registered API surface may not expose all legacy input modes. | Check actual API routes and graph call paths. Remove only capabilities explicitly out of scope and only after their tests/docs are removed or updated. |
| `docs/JEE_Math_Tutor_Architecture.pdf` and older README content | Static docs may still describe Streamlit, old node counts, or previous retrieval implementation. | Regenerate the architecture PDF from the accepted design and update claims to distinguish planned, implemented, and verified behavior. |
| `examples/images/` | The tracked image may be a sample artifact or stale screenshot. | Check references in docs/tests; retain if it is used, otherwise remove or relocate as a clearly labeled sample. |

## 5. Security and correctness checks before cleanup

- Keep all provider credentials and service secrets out of source control. Rotate any real credential that was ever committed; replacing example values does not rotate a leaked secret.
- Verify student ownership for every thread, document, history, analytics, and memory operation.
- Verify Redis checkpoint persistence and HITL resume after API restart, including idempotency and duplicate submissions.
- Verify that the API does not fall back to in-memory persistence in a production path.
- Test malformed/large PDFs, empty or scanned PDFs, upload quotas, partial-ingestion cleanup, deletion, and cross-student retrieval isolation.
- Test the actual frontend against the API rather than relying on build-only validation.
- Avoid removing memory/RAG/security modules based on filename or apparent age alone; use imports, graph reachability, and tests as evidence.

## 6. Recommended cleanup order

1. Get the current CI green after the directory move.
2. Commit an npm lockfile and use reproducible installs.
3. Add integration tests for the frontend-to-API user journey.
4. Reconcile endpoint and retrieval requirements with the source instruction PDF.
5. Generate a call/import graph for agent nodes, tools, memory, and retrieval.
6. Remove only proven-unreachable duplicate code and unused dependencies, one coherent change at a time.
7. Update architecture PDF, dependency documentation, environment-variable tests, and deployment docs after behavior is settled.

No potentially destructive legacy-code deletions were made as part of this audit.
