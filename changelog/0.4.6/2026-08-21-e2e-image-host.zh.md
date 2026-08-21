# 实况 E2E 套件改用可正常拉取的图片地址

- **Date:** 2026-08-21
- **Type:** process
- **Scope:** `tests`
- **PR:** [#188](https://github.com/Prism-Shadow/agenthub/pull/188)

[English](2026-08-21-e2e-image-host.md)

## 变更内容

- `src_py/tests/test_client.py` 与 `src_ts/tests/client.test.ts` 里的 `IMAGE` 改为
  `https://sghimages.shobserver.com/img/catch/2022/01/22/c1ae0300-9402-4128-a7e6-1244d3874167.jpg`，
  拍的仍是水仙，因此 `IMAGE_KEYWORDS` 不变。原地址曾让 DeepSeek 服务端在两次 CI 里返回
  `400 Failed to download image`。
