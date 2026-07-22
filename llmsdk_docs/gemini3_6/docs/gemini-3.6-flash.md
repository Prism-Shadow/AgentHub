# Gemini 3.6 Flash

> Source: https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash (snapshot 2026-07-22)

- **Model code:** `gemini-3.6-flash` (stable)
- **Input token limit:** 1,048,576
- **Output token limit:** 65,536
- **Input modalities:** Text, Image, Video, Audio, and PDF
- **Output modality:** Text
- **Thinking:** Supported (default `medium`; supports `minimal`, `low`, `medium`, `high`)
- **Function calling:** Supported
- **Structured outputs:** Supported
- **Search grounding:** Supported
- **Sampling parameters:** `temperature`/`top_p`/`top_k` are deprecated and ignored
  (HTTP 400 in future generations) — see [latest-model.md](./latest-model.md)
