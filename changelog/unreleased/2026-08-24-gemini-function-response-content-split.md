# Gemini function responses ride in their own contents

- **Date:** 2026-08-24
- **Type:** fix
- **Scope:** `gemini3_7`, `tests`

[中文版](2026-08-24-gemini-function-response-content-split.zh.md)

## What changed

The Gemini client used to convert one universal message into exactly one `Content`, so a user message that carried a tool result **and** text — an agent resending an interrupted turn's tool output together with the user's next prompt, or folding tool outputs into a summarization request — produced a single content mixing `functionResponse` and `text` parts.

Vertex AI rejects that mix with a misleading error: HTTP 400 `"Requests ending with a model turn are not supported."`, even though the request plainly ends with a user content. The official Gemini API endpoint accepts the same request, so the failure only surfaced on Vertex, where it also made every retry of the same conversation fail identically. Verified live against `gemini-3.7-flash` on both endpoints: the mixed content is the one discriminating factor — splitting it is accepted by both, everything else about the request was valid.

`transform_uni_message_to_model_input` now splits a message into consecutive same-role contents by part kind: each run of function responses becomes its own content, the surrounding parts keep theirs, and part order is preserved. A message without function responses — or with nothing else — still converts to a single content, so ordinary requests are byte-identical to before. Both language implementations changed the same way.

## Tests

The client E2E suites gained `should handle tool result mixed with text` (`test_tool_result_mixed_with_text` in Python): after a real tool call, the tool result goes back in the same universal message as a follow-up instruction, and the reply must prove both halves reached the model — the temperature from the tool result, and a marker word demanded by the text. Verified live across the available providers, including `gemini-3.7-flash` on Vertex AI, the endpoint that rejected the pre-fix shape.

The Python tracer suite's two live-model tests moved into `test_client.py` unchanged, restoring the invariant that every test needing a real API key lives in the client E2E suite and the rest of the suites run offline.
