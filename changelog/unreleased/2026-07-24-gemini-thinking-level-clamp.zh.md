# Gemini 3 客户端将思考档位钳制到各模型支持的范围

- **Date:** 2026-07-24
- **Type:** fix
- **Scope:** `gemini3`, `tests`
- **PR:** [#166](https://github.com/Prism-Shadow/agenthub/pull/166)

[English](2026-07-24-gemini-thinking-level-clamp.md)

## 变更内容

- `gemini3/` 客户端（Python 与 TypeScript）不再转发目标模型会拒绝的思考档位。
  `_convert_thinking_level` 现在会把映射得到的 Gemini 档位钳制到该模型支持的最接近档位，
  并列时向上取，而不是让 API 报出
  `400 Thinking level MINIMAL is not supported for this model`。这使 `gemini3` 符合记录在
  `UnsupportedParameterError` 上的设计规则（"Thinking levels never raise this by
  design"）；此前它是唯一违反该规则的客户端。
- 各模型的支持矩阵（来自刷新后的官方 thinking 文档，并于 2026-07-24 针对 Gemini API 完成
  实况验证）：
  - `gemini-3.1-pro*`：`low`、`medium`、`high`——因此 NONE 现在降级为 `low`（此前它映射为
    `minimal`，总是报错）。
  - `gemini-3-pro*`：`low`、`high`（依据文档；官方端点已下线该模型——为 Vertex 保留）——
    NONE 降级为 `low`，MEDIUM 向上取为 `high`。
  - `*-image` 模型（`gemini-3.1-flash-image`、`gemini-3-pro-image`，以及它们的
    `-preview`/`-lite` 变体）：仅支持 `minimal`、`high`——LOW 降级为 `minimal`，MEDIUM
    向上取为 `high`。`-image` 判断先执行，因此 `gemini-3-pro-image` 命中的是图像模型的档位
    集合，而不是 `gemini-3-pro` 的集合。
  - 其他任意 `*-pro*` 模型（一个通用分支，以免未来的 pro 代际静默回退到四档默认值）：
    `low`、`medium`、`high`。
  - `gemini-2.5*`：厂商表格声称支持 `low`/`medium`/`high`，但实况 API 对 2.5 系列拒绝
    **所有** `thinking_level` 取值（"Thinking level is not supported for this model"）——
    实况抓取的效力高于文档，因此该参数被完全丢弃，模型保持其默认的动态思考。只能通过显式
    的 `client_type="gemini-3"` 覆盖或直接构造 `Gemini3Client` 才能走到这里；基于名称的
    路由绝不会把 2.5 模型送到此处。
  - Flash 文本模型保留完整的 `minimal`/`low`/`medium`/`high` 集合，未作改动（NONE 仍映射为
    `minimal`）。
- 降级是有意静默的：不钳制的替代方案就是一个硬性的 400，而全仓库的规则是思考档位绝不抛错。
  有两处钳制是*向上*取的（`gemini-3-pro` 与图像模型上的 MEDIUM → `high`），而 3.1-pro 上的
  NONE → `low` 会为调用方本想避免的思考付费——这是可接受的代价，之所以记录在此，是因为
  客户端没有日志基础设施可以在运行时暴露它。
- `gemini3_6/` 客户端未受影响：其路由的两个模型（`gemini-3.6-flash`、
  `gemini-3.5-flash-lite`）都接受全部四个档位。

## 实况抓取发现

- `gemini-3.1-pro-preview` + `minimal` → 400 "Thinking level MINIMAL is not supported
  for this model. Please retry with other thinking level."；`low` 与 `medium` 成功。
- `gemini-3.1-flash-image` + `low`/`medium` → 同样的 400，印证了图像模型仅支持
  minimal/high 的矩阵。
- `gemini-2.5-pro`/`gemini-2.5-flash`/`gemini-2.5-flash-lite` + 任意档位（包括官方列出的
  `low`）→ 400 "Thinking level is not supported for this model."，与厂商表格中的 2.5 行
  相矛盾。
- `gemini-3-pro-preview` 在官方端点上已不再提供服务（对每个请求都返回 "no longer
  available"），因此其 `low`/`high` 矩阵来自文档表格。
- `gemini-3.1-flash-lite`（文档表格中没有）接受 `minimal`。
- 原始探测记录：`api_captures/gemini3/thinking_levels.probe.jsonl`。

## 文档

- `llmsdk_docs/gemini3/docs/thinking.md`：用实况页面上的逐模型表格替换了过时的三列
  支持/不支持表格（新增 gemini-3.6/3.5 行、`gemini-3-pro-preview` 的 low/high 行，以及
  2.5 行——上文的实况抓取表明 API 实际上并不遵循这些行）。

## 测试

- 离线回归测试在两种语言中固化了钳制矩阵（18 个互相对应的用例，外加两条配置层断言，确认
  `transform_uni_config_to_model_config` 实际输出的正是被钳制/丢弃后的取值）：
  `src_py/tests/test_thinking_level_mapping.py`、
  `src_ts/tests/thinking-level-mapping.test.ts`。

## 同一 PR 中的顺带修复

- 为适配 ruff 0.16 重新格式化了 `src_py/README.md` 中的代码块：Makefile 使用未固定版本的
  `uvx ruff` 进行 lint，而 0.16.0 开始对 markdown 代码块做格式检查，导致每个分支上的
  `make lint`（以及 CI）失败。仅涉及格式。
- 在两种语言中强化了不稳定的 system prompt e2e 提示词（"kitten … meow"）：旧提示词只是
  暗示了该词，模型会围绕它进行角色扮演（`deepseek-v4-flash`、`gemini-3.6-flash:vertex`
  在连续多次 CI 运行中出现抖动）。
