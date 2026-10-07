# MathTutor V2 — Migration Roadmap

## Goal

Transform MathTutor from a multi-agent math chatbot into a production-grade adaptive mathematics learning agent.

The V2 branch is developed incrementally. Each commit must leave the branch internally coherent and must be reviewed for regressions before the next architectural change.

## Non-negotiable product invariants

1. Mathematical correctness takes priority over response speed.
2. No fabricated calculations, citations, retrieval evidence, or memory.
3. Student data and memory remain isolated per user.
4. Tool failures degrade gracefully instead of crashing the tutoring session.
5. Expensive LLM calls are invoked only when the task requires them.
6. Every major agent component has deterministic/unit coverage and evaluation coverage.
7. Production traces must make latency, token usage, tool calls, retrieval, verification, memory and failures observable.
8. V2 must remain usable with low-cost hosted models; provider/model choice must be configuration-driven.
9. Existing V1 behavior is preserved until an equivalent or better V2 path is verified.

## Migration phases

### Phase 0 — Baseline and safety
- Capture repository/runtime/dependency baseline.
- Add V2 architecture documentation and smoke/regression checks.
- Establish evaluation dataset format and golden examples.
- Do not change user-facing behavior yet.

### Phase 1 — Runtime and model modernization
- Introduce provider/model configuration and model routing.
- Migrate deprecated Groq models.
- Add structured outputs where supported.
- Add timeouts, retries, fallback policy and per-node telemetry.
- Remove hidden/global mutable execution state.

### Phase 2 — Agent orchestration
- Replace unconditional sequential LLM calls with adaptive routing.
- Separate planning, execution, verification and teaching.
- Add deterministic math verification before LLM verification.
- Add bounded repair loops and failure recovery.

### Phase 3 — Retrieval and knowledge
- Move RAG from thread-local process memory to persistent document-oriented storage.
- Add metadata-aware/hierarchical retrieval, reranking and citation provenance.
- Add retrieval-specific regression evaluations.

### Phase 4 — Student memory and mastery
- Evolve episodic/semantic/procedural memory into a student model.
- Track skills, misconceptions, mastery, hint dependence and learning progress.
- Add memory isolation, relevance and contamination tests.

### Phase 5 — Evaluation platform
- Component evals: parser, router, RAG, tools, solver, verifier, tutor, memory and safety.
- Workflow evals: representative LangGraph trajectories and failure recovery.
- End-to-end evals: correctness, groundedness, pedagogy, personalization, safety, reliability, cost and latency.
- Regression gates for pull requests and periodic production evaluation.

### Phase 6 — Student experience
- Replace the monolithic Streamlit product UI with a dedicated learning workspace.
- Keep a developer/evaluation UI during migration.
- Add interactive math workspace, graphing, Socratic mode, mistake lab, adaptive practice, mastery dashboard and exam mode.

## Commit discipline

For each architectural change:
1. Inspect current implementation and its callers.
2. Make one coherent change.
3. Update tests/docs/configuration required by that change.
4. Inspect the resulting diff.
5. Run the strongest available static/unit/smoke checks.
6. Check the commit status/workflow where available.
7. Only then continue to the next phase.

## Initial model strategy

The first production configuration will use a fast model for routing/extraction/simple tutoring and a stronger reasoning model for complex solving/verification. Model IDs remain configuration-driven so Groq/OpenRouter/other providers can be added without rewriting agent nodes.

Current Groq production candidates are GPT-OSS 20B for low-latency/cost-sensitive work and GPT-OSS 120B for complex mathematical reasoning. This is a starting configuration, not a permanent provider lock-in.

## Definition of done for V2

V2 is merge-ready only when:
- application starts cleanly from documented setup;
- core math flows pass regression tests;
- RAG/tool/memory/safety failure modes have explicit tests;
- evaluation datasets and runner are reproducible;
- end-to-end quality is demonstrably no worse than V1 and improves targeted metrics;
- latency/cost are measured from real traces rather than mocked numbers;
- no deprecated production model IDs remain;
- UI, API, agent runtime, persistence and observability are documented;
- a clean checkout can reproduce the documented development/evaluation workflow.
