# Dependency decisions — Phase A

Checked on 2026-10-12. Versions newer than 14 days were not selected, per the implementation instructions.

| Package | Pin | Reason |
|---|---:|---|
| langgraph-prebuilt | 1.1.0 | Required by LangGraph 1.2.12 (which declares langgraph-prebuilt>=1.1.0,<1.2.0); the 1.1.0 release is older than 14 days. |
| langgraph | 1.2.12 | Stable release published 2026-09-21; selected instead of 1.2.14 published 2026-10-06. |
| langchain-core | 1.6.5 | Stable release published 2026-09-24; selected instead of releases published 2026-09-29 onward. |
| langchain-groq | 1.1.3 | Latest stable release found; published 2026-06-10. Supports ChatGroq structured-output and reasoning-effort parameters. |
| pydantic | 2.13.5 | Stable release published 2026-08-28; selected instead of 2.14.x releases published in October. The indirect pydantic-core pin was removed so the resolver selects a compatible core version. |
| langgraph-checkpoint-redis | 0.5.2 | Stable release published 2026-08-20; supports the documented Redis checkpoint TTL configuration. |
| redisvl | 0.27.2 | Stable release published 2026-09-10; selected instead of 0.28.0 published 2026-10-09. |

Compatibility references:
- Groq supported models and reasoning effort: https://console.groq.com/docs/models and https://console.groq.com/docs/api-reference
- Groq Structured Outputs: https://console.groq.com/docs/structured-outputs
- ChatGroq parameters and structured-output methods: https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq and https://reference.langchain.com/python/langchain-groq/chat_models/ChatGroq/with_structured_output
- LangGraph release metadata: https://pypi.org/project/langgraph/1.2.12/
- LangChain Core release metadata: https://pypi.org/project/langchain-core/1.6.5/
- LangChain Groq release metadata: https://pypi.org/project/langchain-groq/1.1.3/
- Pydantic release metadata: https://pypi.org/project/pydantic/2.13.5/
- Redis checkpoint TTL API: https://github.com/redis-developer/langgraph-redis
- RedisVL release metadata: https://pypi.org/project/redisvl/0.27.2/

Notes:
- Groq documents JSON-schema structured outputs for openai/gpt-oss-20b and openai/gpt-oss-120b, including strict mode; both support reasoning_effort values low/medium/high and tool calling. openai/gpt-oss-safeguard-20b supports best-effort JSON schema, so the registry does not mark it as strict-schema compatible.
- ModelRouter sets ChatGroq max_retries=0 and owns one 500 ms backoff retry for HTTP 429/5xx, followed by at most one profile downgrade; non-transient errors are not retried. The downgrade is skipped when it would repeat the same model and effort.
- A successful dependency installation and test run is still required before declaring the phase accepted.
- The remaining full-stack dependency matrix will be documented when those stacks are introduced in their relevant phases.
