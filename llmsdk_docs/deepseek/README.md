# DeepSeek SDK Documentation

This directory contains documentation for using the DeepSeek API with OpenAI-compatible SDK patterns.

## Quick Start

- [quickstart.md](./quickstart.md) - First API call with curl, Python, and Node.js

## Documentation

The `docs/` folder contains focused guides for DeepSeek API features:

- [thinking-mode.md](./docs/thinking-mode.md) - Thinking mode, reasoning content, effort control, and tool-call context rules
- [multi-round-chat.md](./docs/multi-round-chat.md) - Stateless multi-turn conversation history
- [json-mode.md](./docs/json-mode.md) - Structured JSON output
- [tool-calls.md](./docs/tool-calls.md) - Tool calling, thinking-mode tool calls, and strict mode
- [kv-cache.md](./docs/kv-cache.md) - Default context caching, hit rules, and usage counters

## Notes

DeepSeek supports OpenAI-compatible and Anthropic-compatible API formats. The examples in this folder focus on the OpenAI-compatible path because it matches the existing SDK documentation style in this repository.
