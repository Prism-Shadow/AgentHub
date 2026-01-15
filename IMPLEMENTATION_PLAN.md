# 统一多模型 SDK 实施计划（JS / Python + API）

## 1. 目标与范围

### 1.1 项目目标
- 提供一个**实时可靠**的统一 SDK，覆盖主流大模型（GPT、Claude、Gemini、GLM 等）。
- 支持 **JavaScript** 与 **Python** 两个 SDK 版本，并提供统一的 **API 服务**能力。
- 提供一致的接口、稳定的流式输出、可靠的错误处理与可观测性。

### 1.2 范围与非目标
**范围内：**
- 统一接口（对话、流式输出、工具调用/函数调用基础能力）。
- 多模型适配器（OpenAI / Anthropic / Google / Zhipu）。
- SDK 级别重试、超时、速率限制、熔断与回退。
- API 服务层（HTTP + SSE/WebSocket）作为可选扩展。

**范围外（当前阶段不做）：**
- 训练、微调平台与模型托管。
- 多租户计费系统。
- 非主流模型的深度定制适配（可在后续扩展）。

## 2. 系统架构与模块划分

### 2.1 模块概览
1. **Core SDK**
   - 统一接口定义、请求/响应标准化、错误模型。
2. **Provider Adapters**
   - OpenAI / Anthropic / Gemini / GLM 的协议适配层。
3. **Transport Layer**
   - HTTP/SSE/WebSocket 适配；支持流式与非流式。
4. **Reliability Layer**
   - 重试（指数退避 + 抖动）、超时、熔断、速率限制、回退策略。
5. **API Service（可选）**
   - 统一 REST + SSE/WebSocket 服务，支持多模型路由与访问控制。
6. **Observability**
   - 日志、指标、链路追踪（OpenTelemetry 可选）。

### 2.2 交互流程
1. SDK 发起请求 → Core SDK 规范化参数
2. Provider Router 选择目标模型适配器
3. Adapter 将请求转换为供应商协议
4. Transport 发送请求，支持流式回传
5. Reliability Layer 处理失败与重试
6. 输出统一响应至调用方

## 3. API 设计（统一接口）

### 3.1 Python SDK 示例
```python
from agent_adapter import Client

client = Client(provider="openai", api_key="...", base_url=None)
response = client.chat(
    model="gpt-4o",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in response:
    print(chunk.delta)
```

### 3.2 JavaScript SDK 示例
```ts
import { Client } from "@agent-adapter/sdk";

const client = new Client({ provider: "anthropic", apiKey: "..." });
const stream = await client.chat({
  model: "claude-3.5-sonnet",
  messages: [{ role: "user", content: "Hello" }],
  stream: true,
});
for await (const chunk of stream) {
  console.log(chunk.delta);
}
```

### 3.3 统一请求/响应模型
- `Message`: `{ role: "system" | "user" | "assistant", content: string }`
- `ChatRequest`:
  - `model`, `messages`, `temperature`, `max_tokens`, `stream`, `tools`
- `ChatResponse`:
  - `id`, `model`, `created`, `content`, `usage`, `finish_reason`
- `StreamChunk`:
  - `delta`, `index`, `finish_reason`

### 3.4 错误模型
- `AdapterError`（供应商错误包装）
- `RetryableError` / `TimeoutError` / `RateLimitError`
- `ValidationError`（客户端参数错误）

## 4. 数据模型与配置

### 4.1 配置方式
- 环境变量（API key / base URL）
- 显式传参（SDK 初始化或请求时）
- 配置文件（YAML/JSON，作为可选扩展）

### 4.2 数据结构
- `ProviderConfig`: `{ provider, api_key, base_url, timeout }`
- `RetryPolicy`: `{ max_retries, base_delay_ms, max_delay_ms, jitter }`
- `CircuitBreaker`: `{ failure_threshold, reset_timeout_ms }`

## 5. 可靠性与实时性设计

### 5.1 实时流式能力
- Python: `async generator` 或 `iterator`
- JS: `AsyncIterable` / `ReadableStream`
- 统一心跳与终止信号

### 5.2 可靠性机制
- 自动重试：幂等请求自动重试，非幂等请求提示调用方
- 速率限制：客户端侧限流 + 服务端 429 处理
- 熔断与回退：当供应商不可用时自动切换备用模型
- 超时与取消：支持请求级超时与中断

## 6. 实施计划（分阶段）

### Phase 0（1 周）准备与对齐
- 明确统一 API 与数据模型
- 定义 provider 适配器接口
- 确定 SDK 与 API 服务的目录结构

### Phase 1（2~3 周）核心 SDK 与 OpenAI Adapter
- 实现 Core SDK（Python/JS）
- OpenAI Adapter + 流式能力
- 统一错误模型与重试策略

### Phase 2（2~3 周）多模型适配与可靠性增强
- Anthropic / Gemini / GLM 适配
- 熔断、回退与多模型路由
- 统一的 metrics/logging 钩子

### Phase 3（2 周）API 服务层
- 统一 REST + SSE/WebSocket 服务
- API Key 管理与基础鉴权
- 运行时配置与模型路由策略

### Phase 4（持续）优化与生态
- 文档、示例、性能优化
- 社区插件机制与新模型接入

## 7. 测试策略

### 7.1 单元测试
- 适配器参数映射
- 错误分类与重试逻辑

### 7.2 集成测试
- Mock Provider 流式响应
- API 服务端到端（SDK → API → Provider）

### 7.3 性能测试
- 并发流式响应延迟
- 重试/熔断稳定性验证

## 8. 部署与发布

### 8.1 SDK 发布
- Python: PyPI，版本语义化
- JS: NPM，支持 ESM + CJS 构建

### 8.2 API 服务部署
- Docker 镜像 + Helm chart（可选）
- 环境变量注入 + 密钥管理

## 9. 风险与应对

| 风险 | 影响 | 缓解方案 |
| --- | --- | --- |
| 供应商 API 变更 | 适配器失效 | Adapter 接口版本化 + CI Contract Tests |
| 流式传输不稳定 | 实时性下降 | 心跳 + 断线重连 + 降级到非流式 |
| 各模型能力差异 | 接口不一致 | 统一最小能力集 + capability flags |

## 10. 验收标准
- [ ] Python 与 JS SDK 支持至少 4 个主流模型
- [ ] 流式输出在 3 个模型上稳定可用
- [ ] 统一错误模型与重试策略通过测试
- [ ] API 服务可完成基本路由与鉴权
- [ ] 文档与示例完整
