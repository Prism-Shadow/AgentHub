# The live E2E suites read their test image from a reachable host

- **Date:** 2026-08-21
- **Type:** process
- **Scope:** `tests`
- **PR:** [#188](https://github.com/Prism-Shadow/agenthub/pull/188)

[中文版](2026-08-21-e2e-image-host.zh.md)

## What changed

- `IMAGE` in `src_py/tests/test_client.py` and `src_ts/tests/client.test.ts` points at
  `https://sghimages.shobserver.com/img/catch/2022/01/22/c1ae0300-9402-4128-a7e6-1244d3874167.jpg`,
  a photograph of the same narcissus subject, so `IMAGE_KEYWORDS` is unchanged. DeepSeek's
  server answered `400 Failed to download image` for the previous host on two CI runs.
