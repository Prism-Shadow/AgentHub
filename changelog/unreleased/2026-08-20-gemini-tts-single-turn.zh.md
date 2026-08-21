# Gemini TTS 请求只带一轮文本，Playground 播放单条音频

- **Date:** 2026-08-20
- **Type:** fix
- **Scope:** `gemini3_7`, `base_client`, `integration`, `tests`

[English](2026-08-20-gemini-tts-single-turn.md)

## 变更内容

- `Gemini3_7Client` 面向 TTS 模型只发送最新一条消息，其余对话不再进入请求，于是有状态会话在第一次
  合成之后仍然可用，不会把模型拒收的历史音频重放回去。
- `Gemini3_7Client` 构造 TTS 请求时只取语音设置与 `max_tokens`，system prompt、thinking level 与
  thinking summary、tools、tool choice、image config 一律不带，因此从别的模型沿用下来的配置不会再
  把一次合成请求变成 400。
- `concat_uni_events_to_uni_message` / `concatUniEventsToUniMessage` 会合并相邻且 mime type 同为
  `audio/` 的 `inline_data` 条目，一次语音回复在历史与 tracer 里是一条，而不是每个流式分片一条。
- Playground 先用一行 `🔊 Receiving audio...` 累计已收到的时长，把音频分片收集起来，流结束后渲染
  一个播放器；中途打断也会保留已经收到的音频。分片先按字节拼接再写 WAV 头，采样率与声道数取自
  mime type 参数（`audio/l16; rate=24000; channels=1`）。

## TTS 请求限制

以下几种请求，`gemini-3.1-flash-tts-preview` 均返回 `400 INVALID_ARGUMENT`（2026-08-20 实测）：

| 请求包含 | 报错信息 |
| --- | --- |
| 多于一轮的对话 | `Multiturn chat is not enabled for this model` |
| 音频 part | `Audio input modality is not enabled for this model` |
| `system_instruction` | `Developer instruction is not enabled for this model` |
| `thinking_config.thinking_level` | `Thinking level is not supported for this model.` |
| `thinking_config.include_thoughts` | `Thinking is not enabled for this model` |
| `tools` | `Function calling is not enabled for this model` |

`max_output_tokens`、`response_modalities` 与 `speech_config` 可以接受。
