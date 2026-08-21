# Gemini TTS requests carry a single text turn, and the playground plays one audio clip

- **Date:** 2026-08-20
- **Type:** fix
- **Scope:** `gemini3_7`, `base_client`, `integration`, `tests`
- **PR:** [#184](https://github.com/Prism-Shadow/agenthub/pull/184)

[中文版](2026-08-20-gemini-tts-single-turn.zh.md)

## What changed

- `Gemini3_7Client` sends only the newest message to a TTS model and leaves the rest of the
  conversation out of the request, so a stateful session keeps working after its first spoken
  answer instead of replaying recorded audio the model refuses.
- `Gemini3_7Client` builds a TTS request from the speech settings and `max_tokens` alone; a system
  prompt, a thinking level or summary, tools, a tool choice, and an image config are left out, so
  configuration carried over from another model no longer turns a synthesis request into a 400.
- `concat_uni_events_to_uni_message` / `concatUniEventsToUniMessage` merge consecutive
  `inline_data` items that share an `audio/` mime type, so a spoken response is one item in the
  history and one entry in the tracer rather than one per streamed chunk.
- The playground collects the audio chunks of a response behind a `🔊 Receiving audio...` line that
  counts up the received duration, then renders one player when the stream ends; an interrupted
  stream keeps the audio that had already arrived. Chunks are joined as bytes before the WAV header
  is written, and the sample rate and channel count are read from the mime type parameters
  (`audio/l16; rate=24000; channels=1`).
- The finished clip plays once by itself, and a browser that blocks autoplay leaves the player
  ready to press. A stream stopped from the Stop button does not play. The token footer is appended
  to the message card instead of re-parsed into it, which would have restarted the clip.

## TTS request limits

`gemini-3.1-flash-tts-preview` answered `400 INVALID_ARGUMENT` to each of these (verified live
2026-08-20):

| Request carries | Message |
| --- | --- |
| more than one turn | `Multiturn chat is not enabled for this model` |
| an audio part | `Audio input modality is not enabled for this model` |
| `system_instruction` | `Developer instruction is not enabled for this model` |
| `thinking_config.thinking_level` | `Thinking level is not supported for this model.` |
| `thinking_config.include_thoughts` | `Thinking is not enabled for this model` |
| `tools` | `Function calling is not enabled for this model` |

`max_output_tokens`, `response_modalities`, and `speech_config` were accepted.
