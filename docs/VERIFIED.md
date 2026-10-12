# Verification log

Last checked: 2026-10-12. This log distinguishes verified official documentation from checks that remain pending; no unverified assumption is presented as fact.

## Verified for Phase A

- Groq currently documents openai/gpt-oss-20b and openai/gpt-oss-120b as supporting strict JSON-schema structured output, tool calling, and reasoning_effort values low, medium, and high. Sources: https://console.groq.com/docs/structured-outputs and https://console.groq.com/docs/api-reference
- Groq's public rate-limit table lists base free-plan limits of 30 RPM, 1,000 RPD, 8K TPM, and 200K TPD for GPT-OSS 20B and 120B; actual organization limits must be checked in the account console. Source: https://console.groq.com/docs/rate-limits
- LangGraph Redis checkpointer supports TTL configuration with default_ttl in minutes and refresh_on_read. Source: https://github.com/redis-developer/langgraph-redis
- Redis Cloud documents volatile-lru as an available eviction policy; it evicts only keys with an expiry. Source: https://redis.io/docs/latest/operate/rc/databases/configuration/data-eviction-policies/

## Still pending; do not treat as verified

- Vercel's currently supported Python runtime versions, request-body limit, and whether execution may continue after a response is sent. Official documentation could not be retrieved reliably during this pass; verify against live Vercel docs before implementing deployment-dependent assumptions.
- Current Redis Cloud free-tier memory allowance and whether eviction policy can be changed on that plan. Policy documentation was available, but plan-specific availability was not confirmed.
- Current Neon/Supabase free-tier limits. The official pricing page could not be retrieved in a parseable form during this pass.

These pending checks are explicit blockers for deployment-dependent decisions in later phases. No code in Phase A relies on the unverified Vercel or database-plan assumptions.
