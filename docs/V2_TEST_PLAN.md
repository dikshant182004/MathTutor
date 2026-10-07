# V2 test plan

## Local bootstrap
1. Checkout V2.
2. Install the existing Python dependencies and test dependencies.
3. Copy .env.example to .env.
4. Configure required provider and service secrets.
5. Start Redis if using persistent checkpoints.
6. Run pytest.
7. Start the existing Streamlit app and run representative text/image/audio flows.
8. Start apps/web separately and verify the new workspace.

## Golden functional cases
- arithmetic: simple direct response
- algebra: equation solving
- calculus: derivative/integral
- probability: tool-assisted reasoning
- uploaded PDF: retrieval + citation
- repeated mistake: mastery decreases and mistake is retained
- successful practice: mastery increases
- Socratic mode: asks a guiding question before revealing a solution
- verifier: catches an invalid final answer
- tool failure: workflow continues or escalates rather than fabricating output
- ambiguous OCR/ASR: HITL path
- unsupported/safety-sensitive request: existing safety path

## Regression gates
- no deprecated Groq model identifiers
- no loss of student isolation
- deterministic verifier never marks an explicitly undefined answer as correct
- direct questions do not call LTM retrieval
- RAG remains available when a session actually has an indexed document
- existing V1 tests remain green
- frontend builds successfully
- no secrets committed
