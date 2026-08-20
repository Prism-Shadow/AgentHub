# AutoLLMClient 可以列出 endpoint 提供的模型 id

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `base_client`, `auto_client`, `claude5`, `gemini3_7`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-list-models.md)

## 变更内容

- 两种语言的 `AutoLLMClient` 新增 `list_models()` / `listModels()`，返回所配置的 endpoint 提供的
  模型 id，顺序与 endpoint 返回的一致。OpenAI 兼容的 client（`openai_chat`、`openai_responses`、
  `gpt5_6`、`minimax_m3`、`deepseek_v4`、`glm5_3`、`kimi_k3`、`openai_embedding`）与 Anthropic
  Messages client（`claude5`、`ant_messages`）读取各自 SDK 的 `models.list()`；`gemini3_7` 读取
  Gemini 的模型列表。翻页由 SDK 负责，因此结果覆盖 endpoint 提供的全部页。
- `gemini3_7` 只取每个 name 的最后一段路径，因此 `models/gemini-3.7-flash` 与
  `publishers/google/models/gemini-3.7-flash` 都会变成 `gemini-3.7-flash`——也就是
  `AutoLLMClient` 的 `model` 参数所接受的写法。
- `claude5` 在 `bedrock://` 形式的 base URL 上抛出 `UnsupportedParameterError`：两种语言的
  Bedrock SDK client 都不带 models 资源。
- `LLMClient` 将该方法声明为抽象方法，因此每个 client 都实现了它。
- 两种语言的离线测试把一个假的 models endpoint 接到全部 client 上，并钉住 Gemini 的路径截取与
  Bedrock 的拒绝行为：`src_py/tests/test_list_models.py`、`src_ts/tests/list-models.test.ts`。

## 各协议对应的请求

| 协议 | Client | 请求 |
| --- | --- | --- |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding` | `GET {base}/models` |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `GET {base}/models` |
| Anthropic Messages | `claude5`, `ant_messages` | `GET {base}/v1/models` |
| Gemini generateContent | `gemini3_7` | `GET {base}/v1beta/models` |
