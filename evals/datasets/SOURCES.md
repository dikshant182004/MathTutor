# Public evaluation sources

- GSM8K: https://huggingface.co/datasets/openai/gsm8k (MIT). Use the test split for solver correctness.
- MATH: https://huggingface.co/datasets/baber/hendrycks_math (MIT-labelled card). Use sampled test problems for difficult solver and verifier evaluation.

Keep large public benchmarks out of pull-request CI. Run sampled, version-pinned
evaluations on demand or on scheduled jobs and store aggregate metrics only.
