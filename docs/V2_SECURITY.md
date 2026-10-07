# V2 security checklist

Before production:
- Use authenticated student identities; do not trust arbitrary browser-supplied student_id values.
- Keep student namespaces isolated at the API/auth boundary.
- Use durable Redis with access controls/TLS and backups.
- Set exact CORS origins.
- Store provider keys only in the deployment secret manager.
- Treat retrieved documents and web results as untrusted content.
- Keep tool permissions explicit and reject unknown tool names/arguments.
- Do not persist raw secrets or unnecessary PII in traces or memory.
- Restrict observability endpoints to trusted operators.
- Add rate limiting at the edge before exposing /v2/solve publicly.
