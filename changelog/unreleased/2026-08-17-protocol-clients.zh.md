# 通用 OpenAI Responses 与 Anthropic Messages 协议 client

- **Date:** 2026-08-17
- **Type:** feature
- **Scope:** `openai_responses`, `ant_messages`, `openai_chat`, `auto_client`, `tests`
- **PR:** [#171](https://github.com/Prism-Shadow/agenthub/pull/171)
- **Breaking:** yes — `openai` client 文件夹重命名为 `openai_chat`，类 `OpenaiClient` 重命名为 `OpenaiChatClient`；旧路径的深层导入失效。

[English](2026-08-17-protocol-clients.md)

## 变更内容

- 新增 `openai_responses`（`OpenaiResponsesClient`，client type `openai-responses`）：通用的
  OpenAI Responses 兼容 client，覆盖 OpenAI、OpenRouter（`https://openrouter.ai/api/v1`）、
  DeepSeek（`https://api.deepseek.com`）、Z.AI（`https://api.z.ai/api/v1`）与 MiniMax
  （`https://api.minimax.io/v1`）。
- 新增 `ant_messages`（`AntMessagesClient`，client type `ant-messages`）：通用的 Anthropic
  Messages 兼容 client，覆盖 Anthropic、OpenRouter（`https://openrouter.ai/api`）、DeepSeek
  （`https://api.deepseek.com/anthropic`）、Z.AI（`https://api.z.ai/api/anthropic`）与
  MiniMax（`https://api.minimax.io/anthropic`）。
- 通用 Chat Completions client 从 `openai`/`OpenaiClient` 迁移为
  `openai_chat`/`OpenaiChatClient`，client type 为 `openai-chat`；裸 `openai` client type
  继续作为别名路由到它，registry 条目改为 `openai-chat`。
- 两个新 client 均透传 `temperature`，`prompt_caching` 默认使用服务商的自动缓存
  （`ENABLE`；其他取值抛 `UnsupportedParameterError`），并按
  [fast mode 支持](2026-08-17-fast-mode.zh.md) 映射 `fast_mode`。
- e2e 覆盖：`deepseek-v4-flash`、`glm-5.2`（GLM Coding Plan base URL）与 `MiniMax-M3`
  通过三个协议 client 全量测试；`openai/gpt-5.6-luna` 在 `RUN_SLOW_TEST` 下经 OpenRouter
  通过三个协议测试。测试的 `Model` 增加 `base_url` 覆盖字段与
  `deepseek`/`zai`/`minimax` provider，parametrize id 追加 client type。

## 协议实现

- `openai_responses` 同时从 `response.reasoning_text.delta`（DeepSeek/Z.AI/MiniMax/OpenRouter
  方言）与 `response.reasoning_summary_text.delta`（OpenAI 方言）流式读取思考内容，并在
  `response.output_item.done` 上收尾 reasoning item，记录 fidelity：`channel`（item 携带
  summary 时为 `summary`）以及存在时的 `encrypted_content`/`signature`/`format`。回放的
  reasoning item 始终携带 `summary` 键（OpenAI API 缺少它会拒绝请求），并按记录的 channel
  把思考文本重建到 `summary` 或 reasoning-text `content`；reasoning item 的 id 不回放。
  assistant 消息的 `phase` 按 `gpt5_6` client 的方式记录并回传，消息内缓冲的文本在每个
  顶层 item 之前先落盘以保持 wire 顺序。
- `ant_messages` 将凭证同时通过 `x-api-key` 与 `Authorization: Bearer` 两个请求头发送
  （`AsyncAnthropic(api_key=key, auth_token=key)`），所有覆盖的服务端都接受。
  `ThinkingLevel.NONE` 显式映射为 `thinking: {"type": "disabled"}`（Z.AI 默认开思考）；
  其他级别映射为 `adaptive` 加 `output_config.effort`，`thinking_summary` 时追加
  `display: "summarized"`。thinking 块仅在记录过 `signature` 时回放它。最终 usage 合并
  `message_start` 与 `message_delta`，delta 中的输入/缓存计数存在时优先（网关在 start
  阶段报零），服务端报告 `output_tokens_details.thinking_tokens` 时填入 `thoughts_tokens`。
