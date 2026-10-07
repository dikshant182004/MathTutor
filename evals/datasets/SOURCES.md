# Public evaluation sources

V2 keeps a small checked-in golden set for fast regression and uses public datasets
for broader model evaluation. The datasets are not copied into the repository.

- GSM8K: https://huggingface.co/datasets/openai/gsm8k
  - MIT licensed; 8.5K grade-school math word problems.
  - Use the test split for solver correctness and reasoning regression.
- MATH: https://huggingface.co/datasets/baber/hendrycks_math
  - MIT-labelled dataset card; 12,500 competition mathematics problems.
  - Use selected test examples for difficult solver/verifier and transfer evaluation.

Recommended live-eval dimensions:
- exact/final-answer correctness
- solution validity and verifier catch rate
- grounded retrieval/citation correctness
- Socratic hint quality
- personalization and memory isolation
- transfer and retention after repeated practice

Do not run large public datasets in pull-request CI. Run sampled, version-pinned
benchmarks locally or on scheduled evaluation jobs and store only aggregate metrics.
