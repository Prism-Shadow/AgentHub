# The dev skill now specifies the client implementation standard

- **Date:** 2026-08-16
- **Type:** process
- **Scope:** `skills`
- **PR:** [#170](https://github.com/Prism-Shadow/agenthub/pull/170)

[中文版](2026-08-16-client-implementation-standard.zh.md)

## What changed

- **Replay is minimal, not exhaustive.** `fidelity` is an exception channel: a field earns a place there only when a live probe shows the request fails or the model degrades without it. Provider-generated ids the API regenerates or ignores — a reasoning item id, an output-item id — stay out; `phase`, thinking signatures, and the exact reasoning field name a strict upstream requires stay in. `fidelity` may never duplicate what a universal field already carries.
- **`tool_call_id` is mandatory**: always capture the provider's call id and replay it. Where a format carries both an item id and a call id, the call id is the one that correlates a result to its call.
- **Stage 2 gained the probe that decides this.** After the capture, the assistant turn is re-sent repeatedly with one candidate field removed each time, and the removals the API tolerates are recorded next to the capture.
- **A reference client per protocol family.** `gpt5_5/` for Responses-style protocols, `openai/` for Chat Completions — same method order, control flow, and names, so a new client reads as a diff against its reference.
- **Readability decides where code goes.** Inlining is the default; a helper is for a genuinely large, self-contained block that would otherwise bury the main flow, never for a few lines, and never as a layer beside the reference client's own private methods.
- **Stream on deltas only.** A partial tool call opens on the item-added event, accumulates through the argument deltas, and is emitted as a complete `tool_call` at the terminal event. Completion events are ignored rather than cross-checked, and provider error events (`response.failed`, `response.error`) are left to the unknown-event guard.

## Problem

The [MiniMax M3 client](2026-08-03-minimax-m3.md) shipped a first draft that review had to take apart: fields stored in `fidelity` that the API never wanted back, a layer of module-level helpers wrapping dict/SDK dual access, JSON validation of completed items, translated provider error events, and tool calls finalized from the completion event instead of the deltas. None of it was careless — each piece followed the skill as written.

That is what made it worth fixing at the source. The skill's Stage 3 required conversion to be "bijective … reproduce the original exactly, including `fidelity` payloads", which reads as an instruction to store every field, and it named no reference implementation, which left the shape of a new client to whoever wrote it. The skill was the cause, not the bystander.

## Decision

Stage 3 states the standard as rules an author can follow and a reviewer can check: what may enter `fidelity` (only what a probe proves is needed), which id is mandatory (`call_id`), which file a new client is shaped after (`gpt5_5/` or `openai/`), where code lives (inline by default, helpers only for large self-contained blocks), and what streaming reads (deltas, never completion events, never provider error events). Stage 2 owns the probe that supplies the evidence, because the minimal set is measured rather than reasoned out.

The "bijective … reproduce the original exactly" rule is deleted rather than qualified: leaving it in place next to the new rules would leave the contradiction that produced the defects.

## Alternatives considered

- **Add the rules beside the existing wording.** Rejected: the old rule actively pushes toward storing everything, so an author following the skill top to bottom would hit the contradiction and could satisfy either half.
- **Enforce the shape with a linter instead of prose.** Rejected: "no helper layer unless the block is large", "read like a diff against `gpt5_5`", and "only store what the API needs back" are judgments a linter cannot make. The mechanical part that does exist — the probe — is a workflow step, not a static check.
- **Leave it to code review.** Rejected: review caught these defects once, at the cost of a full rewrite of a shipped PR. The skill is what an agent reads while writing, so that is where the standard belongs; review remains the backstop.
- **Infer the minimal replay set from the capture rather than probing.** Rejected: a capture shows what the API *sends*, not what it *needs back*. Only removal probes distinguish the two.

## Verification

- Every rule traces to a concrete defect removed from [#167](https://github.com/Prism-Shadow/agenthub/pull/167), and to the shape of the reference client it now points at.
- The probe procedure is not hypothetical: four requests against MiniMax M3 established that a reasoning item replays from its text alone, with no id, and that the id-less replay still hits the prompt cache (512 cached tokens). The revised client passed the full model-scoped e2e in both languages.

## Risks

- The rules pin the shape to today's reference clients. If `gpt5_5/` or `openai/` is restructured, "reads as a diff against its reference" silently changes meaning, and this entry's `## Decision` has to be revisited with them.
- "Genuinely large, self-contained block" is a judgment, not a threshold. It is deliberately not a line count — a numeric rule would be gamed — but two authors can still disagree, and review settles it.
