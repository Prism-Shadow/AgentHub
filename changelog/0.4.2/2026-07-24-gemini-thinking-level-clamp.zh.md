# Gemini 3 客户端将思考档位钳制到各模型实际支持的范围

- **Date:** 2026-07-24
- **Type:** fix
- **Scope:** `gemini3`, `tests`
- **PR:** [#166](https://github.com/Prism-Shadow/agenthub/pull/166)

[English](2026-07-24-gemini-thinking-level-clamp.md)

## 变更内容

- `gemini3/` 客户端（Python 与 TypeScript）不再把目标模型会拒绝的思考档位发出去。
  `_convert_thinking_level` 会把映射后的 Gemini 档位钳制到该模型支持的最接近档位，
  平局时向上取，而不是任由 API 以
  `400 Thinking level MINIMAL is not supported for this model` 失败。
- 各模型支持矩阵：
  - `gemini-3.1-pro*`：`low`、`medium`、`high` —— NONE 降级为 `low`（此前映射为
    `minimal`，必然报错）。
  - `gemini-3-pro*`：`low`、`high` —— NONE 降级为 `low`，MEDIUM 向上取到 `high`。
  - `*-image` 模型（`gemini-3.1-flash-image`、`gemini-3-pro-image` 及其
    `-preview`/`-lite` 变体）：只有 `minimal`、`high` —— LOW 降级为 `minimal`，
    MEDIUM 向上取到 `high`。`-image` 判断在前，因此 `gemini-3-pro-image` 命中的是
    image 集合，而不是 `gemini-3-pro` 的集合。
  - 其他任何 `*-pro*` 模型走一个通用分支，使未来的 pro 代际不会悄悄退回四档默认值：
    `low`、`medium`、`high`。
  - `gemini-2.5*`：该参数被整个丢弃，模型保持其默认的动态思考。线上 API 会拒绝 2.5
    系列的每一个 `thinking_level` 取值（"Thinking level is not supported for this
    model"），与厂商文档表格不符。
  - Flash 文本模型保留完整的 `minimal`/`low`/`medium`/`high` 集合，未受影响
    （NONE 仍映射为 `minimal`）。
- 降级是静默的：`gemini-3-pro` 与 image 模型上 MEDIUM 会向上取到 `high`，3.1-pro 上
  NONE 会映射为 `low`，因此要求不思考的调用方仍会为一部分思考付费。
- `llmsdk_docs/gemini3/docs/thinking.md`：把过时的"支持/不支持"三列表格替换为线上页面
  的各模型表格，新增 gemini-3.6/3.5 行、`gemini-3-pro-preview` 的 low/high 行，以及
  2.5 系列各行。
- 离线回归测试在两种语言下钉住了这套钳制矩阵 ——
  `src_py/tests/test_thinking_level_mapping.py`、
  `src_ts/tests/thinking-level-mapping.test.ts`。

## 同一 PR 中的顺带修复

- `src_py/README.md` 的代码块按 ruff 0.16 重新格式化：该版本开始对 markdown 代码块做
  格式检查，导致每个分支上的 `make lint` 都失败。仅涉及格式。
- 在两种语言下强化了那个不稳定的 system-prompt e2e 提示词（"kitten … meow"）；旧提示词
  只是在暗示那个词，模型会绕着它演。
