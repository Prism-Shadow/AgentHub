# Kimi K3, Gemini 3.6 generation, supported-model registry, and UnsupportedParameterError

## What changed

- New `kimi_k3/` clients (Python and TypeScript) for Moonshot's `kimi-k3`.
- New `gemini3_6/` clients for the Gemini 3.6 protocol generation: `gemini-3.6-flash` and
  `gemini-3.5-flash-lite` (which belongs to this generation despite its 3.5 name).
- New supported-model registry: `agenthub.list_supported_models(currency="USD"|"CNY")` /
  `listSupportedModels(currency)` returns one entry per supported (model, platform) pair:
  the `(model, base_url, client)` triple (mapping directly onto the `AutoLLMClient`
  constructor) plus input/output modalities (Text/Image/Video/Audio/Embed), context window,
  and per-million-token list pricing stored in the vendor's official currency and converted
  at 7 CNY/USD on request. Covers official endpoints plus OpenRouter and SiliconFlow
  (including `z-ai/glm-5.2`, `zai-org/GLM-5.2`, `moonshotai/kimi-k3`, and the OpenRouter
  entries from the AgentHub app catalog: Claude/GPT/Gemini/MiniMax/Grok/Step/Hy3/MiMo/
  Nemotron via the generic OpenAI client, DeepSeek via its native client). OpenRouter
  prices, context windows, and modality flags were pulled from the live `/models` API;
  SiliconFlow prices from the vendors' official CNY price lists. Modalities record what is
  usable through the routed AgentHub client, not the raw upstream capability. Every newly
  added entry passed a live streaming smoke call, except the official Anthropic entries,
  which could not be smoke-tested locally (invalid local `ANTHROPIC_API_KEY`; their ids and
  prices were verified via the OpenRouter passthrough listings).
- New `UnsupportedParameterError` (subclass of `AgentHubError`, hence still a `ValueError` in
  Python) raised for unsupported `temperature`/`tool_choice`/`prompt_caching` values across
  all clients; messages are unchanged, so existing `except ValueError` / message matching
  keeps working.
- Docs snapshots: `llmsdk_docs/kimi_k3/` (raw pages from platform.kimi.com) and
  `llmsdk_docs/gemini3_6/`.

## Protocol differences found

- **Kimi K3 vs K2.6**: reasoning is configured with the top-level `reasoning_effort`
  parameter (`low`/`high`/`max`, default `max`) instead of `extra_body.thinking`, and cannot
  be disabled. `tool_choice` gains `required`; forcing a specific function is incompatible
  with the always-on reasoning. Context caching is fully automatic (no `prompt_cache_key`).
  Everything else on the wire matches K2.6 (`reasoning_content` deltas, standard incremental
  tool-call chunks, `completion_tokens_details.reasoning_tokens`); the K3 stream additionally
  embeds a non-standard usage object inside the finishing choice, which the existing
  top-level-usage accumulation already handles.
- **Gemini 3.6 generation vs Gemini 3**: identical wire format (thought signatures, usage
  fields, event shapes verified by capture). The generation deprecates `temperature`/
  `top_p`/`top_k` — the API silently ignores them today (verified live) and will return
  HTTP 400 in future generations — and disallows requests ending with a non-empty model
  turn. The `gemini3_6` client therefore rejects `temperature` instead of sending a no-op.

## Config mapping decisions (user-confirmed)

- K3 `thinking_level` → `reasoning_effort`: NONE→`low` (cannot disable; degrade, do not
  raise), LOW→`low`, MEDIUM→`high`, HIGH→`high`, XHIGH→`max`; unset → not sent (server
  default `max`).
- K3 drops `trace_id` → `prompt_cache_key` (caching is automatic).
- Gemini 3.6 generation rejects `temperature` via `UnsupportedParameterError`; old
  `gemini3/` client and its models are left untouched.

## Policy changes recorded in the dev skill

- When old and new generations differ, keep the old model's client and routing and add a
  new client folder; never remove old model support unless explicitly instructed.
- Every `ThinkingLevel` must stay usable on every client (map to the closest supported
  level, never raise). `temperature`/`tool_choice` may reject values, but only with
  `UnsupportedParameterError`.
