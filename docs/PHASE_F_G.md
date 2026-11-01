# Phase F and Phase G: persistent notes and web workspace

## Phase F — persistent student documents

### Implemented

- Authenticated `GET /documents`, `POST /documents`, and `DELETE /documents/{document_id}` endpoints.
- PDF upload limit of 10 MiB; files must be PDFs and contain extractable text.
- PDF extraction and chunking with 800-character chunks and 150-character overlap.
- Cohere `embed-english-v3.0` embeddings stored with chunk text and metadata in Redis.
- Redis keys are namespaced by the authenticated 16-character student ID. Both retrieval and deletion check ownership; the API never accepts a student ID from the request body.
- Persistent retrieval is added to the existing `rag_tool` and combined with the legacy per-thread FAISS/BM25 retrieval when available.
- The document store itself uses ordinary Redis hashes and application-side cosine scoring, so it does not require a RediSearch vector index. The LangGraph Redis checkpointer does require RedisJSON and RediSearch; Compose therefore uses the lightweight `redis:8-alpine` image, which includes both modules.
- Unit tests cover cross-student isolation, ownership-safe deletion, metadata listing, ingestion, and relevant retrieval.

### Data layout

- `student-doc:{student_id}:index` — sorted set of that student's document IDs.
- `student-doc:{student_id}:{document_id}:meta` — filename, page/chunk counts, timestamp, hash, and owner.
- `student-doc:{student_id}:{document_id}:chunk:{index}` — chunk text, normalized embedding, document ID, and owner.

Deleting a document removes its metadata, chunk keys, and index entry. The Redis volume must be retained for persistence.

### Configuration and limitations

- Set `COHERE_API_KEY` in the API process environment.
- Scanned/image-only PDFs are rejected with a clear validation error; OCR ingestion is not implemented by this phase.
- Retrieval uses application-side cosine ranking over the student's indexed chunks. This avoids requiring Redis Stack but is a linear scan, so a dedicated vector index may be appropriate at larger scale.
- A failed embedding operation returns a retryable service error; a partially written document is cleaned up before returning.
- Provider API failures and live Redis behavior still require the integration test run with real services.

## Phase G — Next.js frontend

### Implemented

- Next.js App Router application in `frontend/`, with TypeScript and a responsive layout.
- Google Identity Services sign-in; the ID token is exchanged through the existing `POST /auth/google` endpoint.
- API JWT kept in tab-scoped `sessionStorage`.
- Conversation creation, history list, message history, streamed SSE tokens/solution events, and visible API errors.
- Human-in-the-loop interrupt prompts and resume requests with an `Idempotency-Key`.
- PDF notes upload/list/delete through the Phase F endpoints.
- CI job that installs frontend dependencies and builds the Next.js application.

### Local development

Follow `frontend/README.md`. Set `NEXT_PUBLIC_API_BASE_URL` and `NEXT_PUBLIC_GOOGLE_CLIENT_ID` in `frontend/.env.local`. Set the matching server-side `GOOGLE_CLIENT_ID`, `ALLOWED_ORIGINS=http://localhost:3000`, `DATABASE_URL`, `REDIS_URL`, `JWT_SECRET`, and `COHERE_API_KEY` for the API process.

## Acceptance checks before release

1. Install API requirements in the target Python environment and run `python -m pip check`.
2. Run `pytest` with Redis and Postgres available.
3. Run `npm install` and `npm run build` from `frontend/`.
4. With two distinct test identities, upload notes for one identity and prove they cannot be listed, retrieved, or deleted by the other.
5. Upload a text PDF, restart the API process without deleting Redis volumes, and verify the document remains searchable.
6. Complete Google sign-in, create a thread, stream a response, trigger/resume an interrupt, reload history, and delete a document through the browser.
7. Confirm the deployed API's CORS allowlist contains only the deployed frontend origin(s).

A green frontend build does not by itself establish end-to-end readiness. Phase F/G should be marked fully accepted only after these checks pass against the actual API and persistent services.
