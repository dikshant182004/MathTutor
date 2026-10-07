# MathTutor V2 implementation

## Runtime
- Provider/model selection is configuration-driven through ModelGateway.
- Fast role defaults to Groq openai/gpt-oss-20b.
- Reasoning role defaults to Groq openai/gpt-oss-120b.
- Existing BaseAgent llm / reserve_llm interfaces remain compatible.
- A second Groq key is optional.

## Adaptive execution
The graph now inserts an execution-planning node after parsing. Simple/direct questions can skip LTM retrieval. Problems with student notes can retain retrieval, while complex problems are classified for deeper verification.

## Correctness
Verification now has a deterministic SymPy gate before the LLM verifier. Definite symbolic failures are returned immediately; inconclusive cases continue to the existing LLM verifier.

## Learning model
V2 introduces skill mastery state, adaptive difficulty, recent mistake tracking, weakest-skill selection, Socratic hint progression, retrieval/citation primitives, and a persistence abstraction.

The existing production memory system remains intact during migration.

## Evaluation
 evals/runner.py provides a deterministic evaluation entry point and evals/datasets/golden.jsonl contains the initial golden set. The intended V2 evaluation suite should expand this with parser, routing, retrieval, tool-use, solver, verifier, tutoring, memory, safety, trajectory, latency/cost, transfer, and retention datasets.

## Web
apps/web is the new student workspace built with Next.js 16.4 and React 19.3. It provides the V2 UX shell for workspace, Socratic tutoring, mastery, practice, mistake lab, notebook, and agent activity.

The existing Streamlit application remains available while the new API contract is validated.

## Verification status
The branch includes CI configuration for the V2 branch, Python compilation of V2 modules, and unit tests. Full end-to-end validation still requires the repository to be run with its external credentials/services (Groq, Redis, Cohere, Tavily, Google Vision).