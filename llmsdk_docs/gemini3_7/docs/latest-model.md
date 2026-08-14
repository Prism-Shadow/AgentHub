# Latest Gemini models

> Source: https://ai.google.dev/gemini-api/docs/latest-model and
> https://ai.google.dev/gemini-api/docs/pricing (snapshot 2026-08-13)

## Gemini 3.7 Flash

- **Model ID:** `gemini-3.7-flash` (stable)
- **Context window:** 1,048,576 input tokens; 65,536 output tokens
- **Capabilities:** thinking (default `medium` level; `minimal` is not supported), tool
  calling, structured outputs, multimodal input (text, image, video, audio, PDF),
  Computer Use, code execution, spatial reasoning
- Google's latest and most capable Flash model, built for complex coding, agentic
  workflows, and reliable multi-step execution; it powers the updated Antigravity agent.

## Pricing

Standard list price (per 1M tokens), with a launch discount through December 31, 2026:

| Bucket                | Through 2026-12-31 | From 2027-01-01 (list) |
| --------------------- | ------------------ | ---------------------- |
| Input                 | $0.75              | $1.50                  |
| Output (incl. thinking) | $3.75            | $7.50                  |
| Context caching read  | $0.075             | $0.15                  |
| Context caching storage | $0.50 /1M·hour   | $1.00 /1M·hour         |

`gemini-3.6-flash` shares this exact price table, including the launch discount.

## API changes

The Gemini 3.6 generation's parameter contract applies unchanged (see
[../../gemini3_6/docs/latest-model.md](../../gemini3_6/docs/latest-model.md)):
`temperature`/`top_p`/`top_k` stay deprecated and ignored, `candidate_count` is
unsupported, model turn prefill returns HTTP 400, and thinking is configured with the
`thinking_level` enum. The only 3.7 change is the removal of the `minimal` thinking
level on `gemini-3.7-flash`.
