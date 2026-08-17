# The dev skill now specifies the client implementation standard

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[中文版](2026-08-16-client-implementation-standard.zh.md)

## What changed

Stage 3 of the `agenthub-dev` skill gained a client implementation standard, replacing the rule that a conversion be "bijective … reproduce the original exactly, including `fidelity` payloads":

- Replay is minimal: a field enters `fidelity` only when a probe shows the request fails or the model degrades without it, and never when a universal field already carries it. Provider-generated item ids stay out; `phase`, thinking signatures, and an upstream's required reasoning field name stay in.
- `tool_call_id` is mandatory — always the provider's call id, not the item id.
- Each protocol family has a reference client to follow: `gpt5_5/` for Responses-style protocols, `openai/` for Chat Completions.
- Inlining is the default; a helper is for a large self-contained block, never for a few lines.
- Streaming reads deltas only: a partial tool call opens on the item-added event, accumulates through argument deltas, and is emitted as a complete `tool_call` at the terminal event. Completion events are ignored rather than cross-checked, and provider error events are left to the unknown-event guard.

Stage 2 gained the probe that supplies this: after a capture, the assistant turn is re-sent with one candidate field removed at a time, and the removals the API tolerates are recorded next to the capture.
