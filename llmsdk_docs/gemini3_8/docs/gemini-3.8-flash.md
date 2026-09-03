# Gemini 3.8 Flash

> Source: https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash (snapshot 2026-09-03)

- **Model code:** `gemini-3.8-flash` (stable; latest update September 2026)
- **Input token limit:** 1,048,576
- **Output token limit:** 65,536
- **Input modalities:** Text, Image, Video, Audio, and PDF
- **Output modality:** Text
- **Thinking:** Supported (default `medium`; supports `low`, `medium`, `high` — the model
  does not support the `minimal` thinking level and returns an error for it)
- **Function calling:** Supported
- **Structured outputs:** Supported
- **Search grounding:** Supported
- **Caching:** Supported
- **Code execution:** Supported
- **Computer use:** Supported (Preview)
- **Batch API:** Supported
- **Image generation / audio generation / Live API:** Not supported
- **Sampling parameters:** `temperature`/`top_p`/`top_k` are deprecated and ignored
  (HTTP 400 in future generations) — see
  [../../gemini3_6/docs/latest-model.md](../../gemini3_6/docs/latest-model.md)
