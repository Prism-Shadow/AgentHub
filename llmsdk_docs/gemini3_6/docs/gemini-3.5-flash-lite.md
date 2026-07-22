# Gemini 3.5 Flash-Lite

> Source: https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite (snapshot 2026-07-22)

- **Model code:** `gemini-3.5-flash-lite` (stable)
- **Input token limit:** 1,048,576
- **Output token limit:** 65,536
- **Input modalities:** Text, Image, Video, Audio, and PDF
- **Output modality:** Text
- **Thinking:** Supported (default `minimal`; supports `minimal`, `low`, `medium`, `high`)
- **Function calling:** Supported
- **Structured outputs:** Supported
- **Caching:** Supported
- **Batch API / Flex inference / Priority inference:** Supported
- **Not supported:** Live API, image generation, audio generation, computer use
- **Sampling parameters:** `temperature`/`top_p`/`top_k` are deprecated and ignored
  (HTTP 400 in future generations) — this model belongs to the Gemini 3.6 protocol
  generation despite its 3.5 name; see [latest-model.md](./latest-model.md)
