# Gemini 3.7 Flash

> Source: https://ai.google.dev/gemini-api/docs/models/gemini-3.7-flash (snapshot 2026-08-13)

- **Model code:** `gemini-3.7-flash` (stable; latest update August 2026)
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
- **Image generation / audio generation / Live API:** Not supported
- **Sampling parameters:** `temperature`/`top_p`/`top_k` are deprecated and ignored
  (HTTP 400 in future generations) — see
  [../../gemini3_6/docs/latest-model.md](../../gemini3_6/docs/latest-model.md)
