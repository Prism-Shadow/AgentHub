# AutoLLMClient 可以列出 endpoint 提供的模型 id

- **Date:** 2026-08-20
- **Type:** feature
- **Scope:** `auto_client`, `claude5`, `gemini3_7`, `integration`, `tests`
- **PR:** [#181](https://github.com/Prism-Shadow/agenthub/pull/181)

[English](2026-08-20-list-models.md)

## 变更内容

- 两种语言的 `AutoLLMClient` 新增 `list_models()` / `listModels()`，返回所配置的 endpoint 提供的
  模型 id，顺序与 endpoint 返回的一致。OpenAI 兼容的 client（`openai_chat`、`openai_responses`、
  `gpt5_6`、`minimax_m3`、`deepseek_v4`、`glm5_3`、`kimi_k3`、`openai_embedding`）与 Anthropic
  Messages client（`claude5`、`ant_messages`）读取各自 SDK 的 `models.list()`；`gemini3_7` 读取
  Gemini 的模型列表。翻页由 SDK 负责，因此结果覆盖 endpoint 提供的全部页。
- 列表会按所路由的 client 过滤。协议 client（`openai-chat`、`openai-responses`、`ant-messages`、
  `openai-embedding`）是被显式指定的，代表 endpoint 提供的一切，因此原样返回；而由模型 id 推导出的
  client 只保留能推导回它自己的 id，于是面对一个汇聚多家供应商的网关，列表会收窄到该 client 自己的
  模型。
- `AutoLLMClient._client_class_for_model` 把路由判断从 `_create_client_for_model` 中拆出，后者改为
  先查类再构造。判断条件只存在于一处，同时回答两个问题：某个模型 id 该由哪个 client 服务，以及某个
  列出的 id 是否属于手上这个 client。
- `gemini3_7` 只取每个 name 的最后一段路径，因此 `models/gemini-3.7-flash` 与
  `publishers/google/models/gemini-3.7-flash` 都会变成 `gemini-3.7-flash`——也就是
  `AutoLLMClient` 的 `model` 参数所接受的写法。
- 两种语言的 `errors` 新增 `UnsupportedOperationError`，用于报告所路由的 client 根本不具备的能力
  （而不是某个参数值被拒），`claude5` 在 `bedrock://` 形式的 base URL 上抛出它，因为 Bedrock SDK
  client 不带 models 资源。
- Playground 把结果列进自己的模型下拉框：Model 标签行上的 **List models** 控件调用
  `POST /api/models`，把返回的每个 id 添加为一个选项，并标记该次列举所使用的 client type。请求被拒
  时，服务方的报错显示在该字段下方。
- 两种语言的离线测试把一个假的 models endpoint 接到全部 client 上，并钉住过滤规则、Gemini 的路径截取
  与 Bedrock 的拒绝行为：`src_py/tests/test_list_models.py`、`src_ts/tests/list-models.test.ts`。

## 各协议对应的请求

| 协议 | Client | 请求 |
| --- | --- | --- |
| OpenAI Chat Completions | `openai_chat`, `deepseek_v4`, `glm5_3`, `kimi_k3`, `openai_embedding` | `GET {base}/models` |
| OpenAI Responses | `gpt5_6`, `openai_responses`, `minimax_m3` | `GET {base}/models` |
| Anthropic Messages | `claude5`, `ant_messages` | `GET {base}/v1/models` |
| Gemini generateContent | `gemini3_7` | `GET {base}/v1beta/models` |
