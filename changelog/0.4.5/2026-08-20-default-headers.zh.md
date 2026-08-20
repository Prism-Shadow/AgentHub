# Client 支持传入默认 header，以对接要求特定 header 的 endpoint

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `auto_client`, `claude5`, `gemini3_7`, `integration`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-default-headers.md)

## 变更内容

- `AutoLLMClient` 与全部 client 新增 `default_headers` / `defaultHeaders` 构造选项，其内容会随发往该
  endpoint 的每一个请求一同发送。有些网关会拒绝它不认识的客户端（`unauthorized_client_error`、缺少
  `HTTP-Referer` 或 `X-Title`、缺少 endpoint 要求的 `Anthropic-Beta`），声明它们要求的 header 即可接上。
- 该选项属于连接层面而非请求层面：只在构造时传入一次，不进入 `UniConfig`，因此不会逐请求变化，也不会
  抵达模型。
- Playground 在 Base URL 下方新增 **Extra Headers** 输入框，接受一个 JSON 对象，以 `default_headers`
  随面板配置发送，内容变化时会重建会话 client；不是 JSON 对象的文本只会把输入框标红，不会被发送。
- 两种语言的测试各起一个本地 HTTP 服务端，分别通过 OpenAI、Anthropic 与 Gemini 三种 client 列出模型，
  断言所声明的 header 确实到达，并断言未声明 header 的 client 不会凭空添加：
  `src_py/tests/test_default_headers.py`、`src_ts/tests/default-headers.test.ts`。

## 各 SDK 的承载方式

| SDK | Client | 承载于 |
| --- | --- | --- |
| OpenAI | `openai_chat`, `openai_responses`, `gpt5_6`, `minimax_m3`, `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding` | `default_headers` |
| Anthropic | `claude5`（直连与 Bedrock）、`ant_messages` | `default_headers` |
| Google GenAI | `gemini3_7` | `http_options.headers` |
