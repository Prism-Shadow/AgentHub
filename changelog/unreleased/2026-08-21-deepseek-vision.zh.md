# deepseek-v4-flash-vision-exp 可读图片，提示词与工具返回都支持

- **Date:** 2026-08-21
- **Type:** feature
- **Scope:** `deepseek_v4`, `registry`, `llmsdk_docs`, `tests`

[English](2026-08-21-deepseek-vision.md)

## 变更内容

- 注册表新增官方 API 上的 `deepseek-v4-flash-vision-exp`，输入模态为 `Text, Image`，上下文窗口
  1M，价格与 `deepseek-v4-flash` 相同（谷值每百万 token 输入 ¥1.5、输出 ¥4.5，命中缓存 ¥0.05）。
- `DeepSeekV4Client` 把 `image_url` 条目发成 `input_image` 内容块，把 `tool_result` 里的图片发成
  `function_call_output` 内的 `input_image`，于是工具可以把自己产出的图片交给模型。HTTP(S) 链接与
  base64 data URL 都可用。
- 不带视觉能力的模型遇到图片条目会直接报
  `DeepSeek <model> does not support image inputs.` 而不是照发：Chat Completions 会返回 `400`，
  Responses API 则会静默替换成占位文本。
- 两份 E2E 模型清单里，官方 DeepSeek 条目由 `deepseek-v4-flash` 换成
  `deepseek-v4-flash-vision-exp` 并打开图像理解；三个协议模式条目仍用 `deepseek-v4-flash`。
- `llmsdk_docs/deepseek_v4/` 新增视觉与 Responses API 两份官方文档快照，quickstart 补上第三个模型。
