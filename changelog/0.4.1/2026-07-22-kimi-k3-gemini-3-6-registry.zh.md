# Kimi K3、Gemini 3.6 代际、受支持模型注册表与 UnsupportedParameterError

- **Date:** 2026-07-22
- **Type:** feature
- **Scope:** `kimi_k3`, `gemini3_6`, `glm5_2`, `registry`, `errors`
- **PR:** [#163](https://github.com/Prism-Shadow/agenthub/pull/163)

[English](2026-07-22-kimi-k3-gemini-3-6-registry.md)

## 变更内容

- 新增 `kimi_k3/` 客户端（Python 与 TypeScript），支持 Moonshot 的 `kimi-k3`。
- 新增 `gemini3_6/` 客户端，面向 Gemini 3.6 协议代际：`gemini-3.6-flash` 与
  `gemini-3.5-flash-lite`（后者名称虽为 3.5，但属于该代际）。
- 新增 `glm5_2/` 客户端，支持 Z.AI 的 GLM-5.2（`glm5_1/` 保持不变）：在
  `extra_body.thinking` 配置之外新增顶层 `reasoning_effort` 参数。映射为 1:1，因为官方
  schema 已文档化服务端的兼容性映射（`low`/`medium` 映射为 `high`，`xhigh` 映射为
  `max`）：NONE 保持 `thinking={"type": "disabled"}` 且不发送 effort；LOW/MEDIUM/HIGH/XHIGH
  发送 `reasoning_effort` 的 `low`/`medium`/`high`/`xhigh`，并附带
  `thinking={"type": "enabled", "clear_thinking": false}`（保留思考内容；客户端原样回放
  `reasoning_content`）。`tool_choice` 仍仅支持 auto，temperature 直接透传，与 GLM-5.1
  一致。已针对官方端点的实况抓取（含一次禁用思考的探测）以及 OpenRouter 与 SiliconFlow
  完成验证；网关的 `glm-5.2` 模型 id 现在会自动路由到新客户端。
- Gemini 图像端点迁移：`gemini-3.1-flash-image-preview` 已弃用，并在注册表、测试与 skill
  引用中统一替换为 `gemini-3.1-flash-image`（`gemini-3-pro-image-preview` 同样改为
  `gemini-3-pro-image`）；图像生成已在新端点上重新完成实况验证。
- 新增受支持模型注册表：`agenthub.list_supported_models(currency="USD"|"CNY")` /
  `listSupportedModels(currency)` 为每个受支持的（模型，平台）组合返回一个条目：包含
  `(model, base_url, client)` 三元组（可直接对应 `AutoLLMClient` 构造函数），以及输入/输出
  模态（Text/Image/Video/Audio/Embed）、上下文窗口，和按每百万 token 计的标价——以厂商官方
  币种存储，并在请求时按 7 CNY/USD 换算。覆盖官方端点以及 OpenRouter 与 SiliconFlow
  （包括 `z-ai/glm-5.2`、`zai-org/GLM-5.2`、`moonshotai/kimi-k3`，以及来自 AgentHub 应用
  目录的 OpenRouter 条目：Claude/GPT/Gemini/MiniMax/Grok/Step/Hy3/MiMo/Nemotron 走通用
  OpenAI 客户端，DeepSeek 走其原生客户端）。OpenRouter 的价格、上下文窗口与模态标志取自
  实况 `/models` API；SiliconFlow 的价格取自各厂商官方人民币价目表。模态记录的是经由路由后
  的 AgentHub 客户端实际可用的能力，而非上游的原始能力，并逐条内联写出。定价键与 AgentHub
  的用量分桶保持一致（`cached_tokens`、`prompt_tokens`、`thoughts_tokens`、
  `response_tokens`；thoughts 与 response 均采用厂商的输出价格），以美元存储，请求 CNY 时
  乘以 7；以人民币计价的官方标价通过 `cny()` 初始化器声明，在写入时按 7 CNY/USD 换算。
  Gemini 的缓存命中价格取自官方定价页面（gemini-3.6-flash $0.15，gemini-3.5-flash-lite
  $0.03）。`qwen/qwen3-embedding-4b` 继续列出（OpenRouter 仅含 chat 的 `/models` API 中没有
  embedding 模型，但其模型页面仍然有效）。每个新增条目都通过了一次实况流式冒烟调用，官方
  Anthropic 条目除外——它们无法在本地进行冒烟测试（本地 `ANTHROPIC_API_KEY` 无效；其 id 与
  价格通过 OpenRouter 的透传列表核验）。
- 新增 `UnsupportedParameterError`（`AgentHubError` 的子类，因此在 Python 中仍是
  `ValueError`），在所有客户端中针对不受支持的 `temperature`/`tool_choice`/`prompt_caching`
  取值抛出；错误消息保持不变，因此现有的 `except ValueError` 与消息匹配仍然有效。
- 文档快照：`llmsdk_docs/kimi_k3/`（来自 platform.kimi.com 的原始页面）与
  `llmsdk_docs/gemini3_6/`。

## 发现的协议差异

- **Kimi K3 与 K2.6**：推理改用顶层 `reasoning_effort` 参数配置（`low`/`high`/`max`，默认
  `max`），不再使用 `extra_body.thinking`，且无法关闭。`tool_choice` 新增 `required`；强制
  指定某个函数与始终开启的推理不兼容。上下文缓存完全自动（没有 `prompt_cache_key`）。协议
  层面的其余部分与 K2.6 一致（`reasoning_content` 增量、标准的增量工具调用分片、
  `completion_tokens_details.reasoning_tokens`）；K3 的流式响应还会在结束的 choice 内嵌入
  一个非标准的 usage 对象，现有的顶层 usage 累加逻辑已能处理。
- **Gemini 3.6 代际与 Gemini 3**：协议格式完全相同（thought signature、usage 字段、事件结构
  均由实况抓取验证）。该代际弃用了 `temperature`/`top_p`/`top_k`——目前 API 会静默忽略它们
  （已实况验证），并将在未来代际中返回 HTTP 400——并且不允许请求以非空的 model 回合结尾。
  因此 `gemini3_6` 客户端直接拒绝 `temperature`，而不是发送一个不起作用的参数。

## 配置映射决策（用户已确认）

- K3 的 `thinking_level` → `reasoning_effort`：NONE→`low`（无法关闭；降级处理，不抛错）、
  LOW→`low`、MEDIUM→`high`、HIGH→`high`、XHIGH→`max`；未设置时不发送（服务端默认
  `max`）。
- K3 取消了 `trace_id` → `prompt_cache_key` 的映射（缓存是自动的）。
- Gemini 3.6 代际通过 `UnsupportedParameterError` 拒绝 `temperature`；旧的 `gemini3/`
  客户端及其模型保持不变。

## 测试矩阵策略

- `AVAILABLE_MODELS` 现在在每个供应商分组中只保留各模型家族的最新版本：gemini-3.6-flash
  取代 gemini-3.5-flash 与 3.5-flash-lite（含 Vertex），glm-5.2 取代 glm-5.1（官方、
  OpenRouter、SiliconFlow），kimi-k3 取代 kimi-k2.6（官方、OpenRouter；SiliconFlow 仍以
  Kimi-K2.6 作为其可用的最新 Kimi）。旧客户端仍受支持并保留路由，但不再进行 e2e 测试。

## 记录在开发 skill 中的策略变更

- 当新旧代际存在差异时，保留旧模型的客户端与路由，并新增一个客户端目录；除非有明确指示，
  绝不移除对旧模型的支持。
- 每个 `ThinkingLevel` 在每个客户端上都必须保持可用（映射到最接近的受支持思考档位，绝不
  抛错）。`temperature`/`tool_choice` 可以拒绝某些取值，但只能抛出
  `UnsupportedParameterError`。
