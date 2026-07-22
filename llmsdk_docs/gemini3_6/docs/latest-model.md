# Latest Gemini models

> Source: https://ai.google.dev/gemini-api/docs/latest-model (snapshot 2026-07-22)

## Gemini 3.6 Flash

- **Model ID:** `gemini-3.6-flash` (stable)
- **Pricing:** $1.50 / 1M input tokens, $7.50 / 1M output tokens
- **Context window:** 1,048,576 input tokens; 65,536 output tokens
- **Capabilities:** thinking (default `medium` level), tool calling, structured outputs,
  multimodal input (text, image, video, audio, PDF), Computer Use, code generation,
  spatial reasoning
- Balances speed with intelligence for agentic and multimodal tasks.

## Gemini 3.5 Flash-Lite

- **Model ID:** `gemini-3.5-flash-lite` (stable)
- **Pricing:** $0.30 / 1M input tokens, $2.50 / 1M output tokens
- **Context window:** 1,048,576 input tokens; 65,536 output tokens
- **Capabilities:** thinking (default `minimal` level), tool calling, structured outputs,
  multimodal input, Computer Use, data extraction, structured JSON parsing
- The fastest, lowest-cost 3.5 model for high-throughput execution.

## API changes

> "Starting with Gemini 3.6 Flash and Gemini 3.5 Flash-Lite, the following API changes
> apply to these models and all future Gemini model releases."

1. **Sampling parameter deprecation.** `temperature`, `top_p`, and `top_k` are deprecated.
   The API currently ignores these parameters; in future model generations, supplying them
   returns an HTTP 400 error.
2. **Prefilled model turn validation.** API requests ending with a non-empty `model` role
   turn are disallowed and return an HTTP 400 error.
3. **Thinking configuration.** Use the `thinking_level` enum instead of `thinking_budget`.
   See [thinking.md](./thinking.md) for supported levels per model.

## Live capture notes (api_captures/gemini3_6/)

- Both models stream the same event shapes as the Gemini 3 generation: parts with optional
  `thought_signature` on function-call and final text parts, `usage_metadata` with
  `thoughts_token_count`, finish reason `STOP`.
- Thought summaries may be absent for simple requests; a signature can arrive on a
  non-thought part and must be replayed as received.
- A `temperature=0.5` probe against both models succeeded (silently ignored), matching the
  documented "ignored today, error later" behavior.
