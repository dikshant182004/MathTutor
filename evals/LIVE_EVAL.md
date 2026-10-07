# Live LLM evaluation

Run sampled public or private JSONL datasets against the real V2 LangGraph workflow.
This intentionally lives outside PR CI because it consumes provider quota.

Supported row fields:
- GSM8K-style: question + answer
- MATH-style/custom: problem/prompt + answer/target

Example:
python -m evals.live evals/datasets/gsm8k-test.jsonl --limit 25

Metrics recorded are exact final-answer accuracy, per-case latency and failures.
Use separate sampled runs for solver, verifier, Socratic, RAG and transfer/retention
experiments; do not treat one benchmark score as the complete tutor quality metric.
