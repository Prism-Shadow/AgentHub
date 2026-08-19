# 思考档位新增 max 一档

- **Date:** 2026-08-19
- **Type:** feature
- **Scope:** `types`, `claude5`, `deepseek_v4`, `gemini3_7`, `tests`

[English](2026-08-19-thinking-level-max.md)

## 变更内容

- `ThinkingLevel` 在 `XHIGH` 之上新增 `MAX`：两种语言的档位现为 `NONE`、`LOW`、`MEDIUM`、
  `HIGH`、`XHIGH`、`MAX`。各 client 将其映射到自家服务方所接受的最深档位，遇到没有对应档位的
  服务方则静默降级，因此不会有 client 因为思考档位而抛错。
- `deepseek_v4` 按 DeepSeek 当前的取值重排了映射：该服务接受 `low`/`high`/`max`，并在服务端把
  `medium` 与 `xhigh` 映射为 `high`。因此 `LOW` 现在发送 `low`（此前发送 `high`），`XHIGH`
  现在发送 `high`（此前发送 `max`），`MAX` 发送 `max`。
- `gemini3_7` 改为在「没有从 chunk 中读出任何内容」时（无内容项、无用量、无结束原因）把事件标记为
  `unused`，而不再依据原始 chunk 上缺少 candidates 与 usage 来推断这是网关心跳。不返回用量的流在
  两种写法下都不受影响。
- playground 的思考档位选择器新增 `Max` 选项，`XHigh` 的描述也相应改为扩展档而非最高档。
- `gemini3_7` 记录了 `thinking_summary` 在哪些模型上真的会返回思考摘要：2026-08-19 实测，
  `gemini-2.5-flash` 与 `gemini-3.1-pro-preview` 会返回 thought part，而 `gemini-3.5-flash`、
  `gemini-3.6-flash`、`gemini-3.7-flash` 都不返回。请求本就带上了 `include_thoughts`，
  client 无需改动。

## 思考档位映射

| AgentHub | OpenAI Responses | Claude 4.6+ | Gemini | DeepSeek V4 | GLM-5.3 | GLM-5.2 | Kimi K3 | MiniMax M3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `NONE` | `none` | 不发送 thinking | `minimal` | 关闭思考 | `low` | 关闭思考 | `low` | `none` |
| `LOW` | `low` | `low` | `low` | `low` | `low` | `low` | `low` | `low` |
| `MEDIUM` | `medium` | `medium` | `medium` | `high` | `high` | `medium` | `high` | `medium` |
| `HIGH` | `high` | `high` | `high` | `high` | `high` | `high` | `high` | `high` |
| `XHIGH` | `xhigh` | `xhigh`（4.6 上为 `high`） | `high` | `high` | `max` | `xhigh` | `max` | `high` |
| `MAX` | `max` | `max` | `high` | `max` | `max` | `max` | `max` | `high` |

Gemini 的档位随后还会钳制到目标模型支持的集合；GLM-5.3 与 Kimi K3 无法关闭思考，因此 `NONE` 落在
它们最轻的档位上。

## 参考文档

- `llmsdk_docs/deepseek_v4/docs/thinking-mode.md` 依据
  https://api-docs.deepseek.com/guides/thinking_mode 重新同步：`reasoning_effort` 接受
  `low`/`high`/`max`，兼容映射把 `medium` 与 `xhigh` 归入 `high`。
