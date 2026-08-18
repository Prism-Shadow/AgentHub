# Add the `fidelity` field and replay the exact reasoning field the upstream produced

- **Date:** 2026-07-20
- **Type:** fix
- **Scope:** `types`, `base_client`, `openai`, `glm5_1`, `kimi_k2_6`
- **PR:** [#159](https://github.com/Prism-Shadow/agenthub/pull/159)
- **Breaking:** yes — clients no longer read top-level `signature`/`phase` on content items; histories recorded by earlier versions must move them into `fidelity`

[中文版](2026-07-20-reasoning-field-fidelity.zh.md)

## The bug

OpenAI Chat Completions-compatible servers spell the streamed thinking field differently — vLLM & SiliconFlow use `reasoning_content` while OpenRouter uses `reasoning` — and when sending assistant history back, the `openai`, `glm5_1`, and `kimi_k2_6` clients always set **both** fields on the message. Strict upstreams reject the spelling they did not emit (e.g. a server that returned `reasoning_content` refuses a request containing `reasoning`), breaking multi-turn conversations.

## The `fidelity` field

Fixing this needs a place to record which wire field carried the thinking. Rather than overloading `signature`, content items now carry a single dedicated field:

- `fidelity` (`dict[str, Any]` / `Record<string, any>`, optional) — an arbitrary JSON-style object of wire-level data a client records to reproduce the original message on replay. Opaque to consumers: pass it back unchanged.

It replaces and absorbs the former item-level `signature` and `phase` fields:

| Client | Old | New |
| --- | --- | --- |
| `claude5` / `claude4_6` | `signature: <sig>` on thinking (also holds redacted-thinking data) | `fidelity: {"signature": <sig>}` |
| `gemini3` | `signature: <thought_signature>` on text / thinking / inline / tool_call items (key present even when `None`) | `fidelity: {"signature": <thought_signature>}`, omitted entirely when absent |
| `gpt5_5` | `signature: json.dumps({"id": ..., "encrypted_content": ...})` on thinking; `phase: <p>` on text | `fidelity: {"id": ..., "encrypted_content": ...}` (no more JSON-in-a-string); `fidelity: {"phase": <p>}` |
| `openai` / `glm5_1` / `kimi_k2_6` / `deepseek_v4` | nothing recorded; both reasoning spellings sent back | `fidelity: {"reasoning_field": "reasoning_content" \| "reasoning"}` per thinking delta |

## The reasoning-field fix

On receive, the OpenAI-compatible clients record the wire field name that carried each thinking delta. On send, the message conversion replays the thinking through exactly that field. Fallbacks keep the old maximum-compatibility behavior: thinking without a recorded `reasoning_field` (hand-written histories, foreign-protocol fidelity), mixed fields within one message, and the ambiguous case where one chunk carries both spellings (such deltas record no fidelity) all still send both fields.

## Concatenation rules

`concat_uni_events_to_uni_message` / `concatUniEventsToUniMessage` now key on `fidelity`:

- text: a phase change starts a new item (same-phase and phaseless deltas merge, per the GPT-5.5 `phase` guide); an incoming fidelity payload (e.g. a signature) merges into the open item's fidelity and finishes it.
- thinking: an incoming fidelity payload finishes the open item (Claude signature deltas, GPT-5.5 reasoning markers), and a run of deltas carrying **equal** fidelity concatenates into one item (the OpenAI-compatible per-delta `reasoning_field` tags).

## Compatibility

Histories recorded by earlier versions carry `signature` / `phase` at the top level of content items; clients no longer read those fields. To replay an old history, move each item's `signature`/`phase` into `fidelity` (`{"signature": ...}` for Claude/Gemini, `{"id": ..., "encrypted_content": ...}` parsed from the GPT-5.5 JSON string, `{"phase": ...}` for GPT-5.5 text).

## Tests

Offline fake-stream suites `src_py/tests/test_reasoning_fidelity.py` and `src_ts/tests/reasoning-fidelity.test.ts` cover both reasoning field spellings, the ambiguous both-fields case, and the no-fidelity fallback across the `openai`, `glm5_1`, and `kimi_k2_6` clients.
