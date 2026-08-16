# Reject thinking-only responses and introduce the AgentHubError base class

- **Date:** 2026-07-14
- **Type:** fix
- **Scope:** `errors`, `base_client`, `tests`
- **PR:** [#155](https://github.com/Prism-Shadow/agenthub/pull/155), [#157](https://github.com/Prism-Shadow/agenthub/pull/157)

## What changed

- A response that completes with thinking output only now raises `EmptyResponseError` as soon as the stream ends, because replaying a thinking-only assistant message on the next turn fails with a 400 error ([#157](https://github.com/Prism-Shadow/agenthub/pull/157)).
- `EmptyResponseError` and `ToolCallArgumentParseError` (raised for malformed streamed tool arguments, [#155](https://github.com/Prism-Shadow/agenthub/pull/155)) now inherit the new `AgentHubError` base class, so callers can catch all AgentHub-raised errors in one place.

*Backfilled when the changelog was reorganized by release version; see the git history of the release range for full context.*
