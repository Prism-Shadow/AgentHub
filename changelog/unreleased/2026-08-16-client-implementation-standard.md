# The dev skill now specifies the client implementation standard

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[中文版](2026-08-16-client-implementation-standard.zh.md)

## What changed

- **Replay is minimal, not exhaustive.** The Stage 3 rule that a conversion be "bijective … reproduce the original exactly, including `fidelity` payloads" is gone. A replayed assistant turn now only has to be accepted by the API with the model's behavior intact, and `fidelity` is an exception channel: a field earns a place there only when a live probe shows the request fails or the model degrades without it. Provider-generated ids the API regenerates or ignores — a reasoning item id, an output-item id — stay out; `phase`, thinking signatures, and the exact reasoning field name a strict upstream requires stay in.
- **Stage 2 gained the probe that decides this.** After the capture, the assistant turn is re-sent repeatedly with one candidate field removed each time, and the removals the API tolerates are recorded next to the capture. What the API needs back is now measured, not inferred from what it sent.
- **`fidelity` may not duplicate a universal field** — the thinking text, a tool call's name or parsed arguments. The wire item is rebuilt from the universal fields instead.
- **`tool_call_id` is mandatory**: always capture the provider's call id and replay it. Where a format carries both an item id and a call id, the call id is the one that correlates a result to its call.
- **Follow the reference client; do not redesign.** `gpt5_5/` is the shape for Responses-style protocols, `openai/` for Chat Completions — same method order, control flow, and names, so a new client reads as a diff against its reference.
- **Readability decides where code goes.** A reader should follow one request or response straight through the file, so inlining is the default: SDK attributes are read directly, and field access, usage arithmetic, and error text stay where they are used. A helper is for a genuinely large, self-contained block that would otherwise bury the main flow — fetching and decoding an image, say — never for a few lines, and never as a layer beside the reference client's own private methods. Dict/attribute dual-access shims are never warranted, because the events are typed.
- **Stream on deltas only.** A partial tool call opens on the item-added event, accumulates through the argument deltas, and is emitted as a complete `tool_call` at the terminal event, as `gpt5_5` does — every client must emit a complete `tool_call`, not just partials. Completion events (`response.output_item.done` and equivalents) are listed with the ignored types and never cross-checked against the deltas, and provider error events (`response.failed`, `response.error`) are left to the unknown-event guard instead of being translated.

## Why

- Every rule here is a defect the [MiniMax M3 PR](2026-08-03-minimax-m3.md) shipped and review had to remove: fields stored in `fidelity` that the API never wanted back, a layer of module-level helpers around dict/SDK dual access, JSON validation of completed items, translated provider error events, and tool calls finalized from the completion event rather than the deltas. The skill was the cause, not the bystander — its "reproduce the original exactly" wording actively pushed toward storing everything, so the fix is a rewrite of that rule rather than an addition next to it.
- The minimal set cannot be reasoned out from a capture, because a capture shows what the API *sends*, not what it *needs back*. Probing is the only way to know, and it is cheap: the MiniMax probe took four requests and showed the reasoning item replays from its text alone, with no id at all.
- Naming a reference implementation per protocol family turns "write it in the house style" into something checkable. A reviewer diffs the new client against `gpt5_5/`; anything that is not a protocol difference is a finding.
