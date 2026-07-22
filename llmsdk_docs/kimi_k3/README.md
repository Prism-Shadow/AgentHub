# Kimi K3 SDK Documentation

This directory contains documentation for Moonshot's Kimi K3 API, snapshotted from the
official platform documentation (https://platform.kimi.com/docs/).

## Quick Start

- See [quickstart.md](./quickstart.md) (OpenAI-compatible API, Python and cURL examples)

## Documentation

The `docs/` folder contains detailed guides on Kimi K3 features:

- [thinking-effort.md](./docs/thinking-effort.md) - The `reasoning_effort` parameter (`low`/`high`/`max`, default `max`; reasoning cannot be disabled)
- [tool-choice.md](./docs/tool-choice.md) - `tool_choice` values (`auto`/`none`/`required`/specific function)
- [tool-calling-best-practice.md](./docs/tool-calling-best-practice.md) - K3 tool calling best practices
- [tool-calls.md](./docs/tool-calls.md) - Complete tool calling walkthrough
- [vision.md](./docs/vision.md) - Image and video input (base64 or `ms://<file-id>`; public URLs are not supported)
- [context-caching.md](./docs/context-caching.md) - Automatic context caching (no extra request parameters)
- [streaming.md](./docs/streaming.md) - Streaming output (`reasoning_content` and `content` deltas)
- [chat-api.md](./docs/chat-api.md) - Chat Completion API reference
- [models-overview.md](./docs/models-overview.md) - Model overview
- [pricing.md](./docs/pricing.md) - K3 pricing and context window

## Key protocol differences vs Kimi K2.x

- Reasoning is configured with the top-level `reasoning_effort` parameter (`low`/`high`/`max`,
  default `max`) instead of the K2.x `extra_body.thinking` object, and cannot be disabled.
- `tool_choice` additionally supports `required`; forcing a specific function is incompatible
  with reasoning (which is always on).
- Context caching is fully automatic; no `prompt_cache_key` or other cache parameters are needed.
- Multi-turn conversations must replay the complete assistant message exactly as received,
  including `reasoning_content` and `tool_calls`.
- Sampling parameters (temperature and friends) remain fixed, as in K2.x.
