# V2 validation checklist

Run these after pulling V2 into the IDE.

1. Install root Python dependencies and start the API.
2. Install apps/web dependencies and run the Next.js app.
3. Run the complete Python test suite.
4. Confirm the Next.js production build succeeds.
5. Open the workspace and verify Workspace, Practice, Mastery, Mistake Lab,
   Knowledge Graph, Graphing, Whiteboard and Exam Mode.
6. Submit a Socratic problem and a normal solve problem.
7. Submit one wrong and one correct solution; confirm mastery and mistakes change.
8. Upload a text document and a PDF, then ask a question grounded in that document.
9. Verify document citations retain source/page metadata.
10. Verify semantic/procedural/profile memory stays isolated between student IDs.
11. Inspect /v2/observability/traces and confirm latency, LLM calls, tool calls and
    provider token metadata are real request measurements, not mock values.
12. Repeat the same problem in a new thread and confirm student memory persists.
13. Wait or backdate a local test skill record and confirm retention-adjusted mastery
    affects the next-best problem.
14. Run a small live LLM evaluation sample with evals/live.py.
15. Do not merge until all of the above are green locally.
