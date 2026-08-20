# 实况 E2E 测试对限流响应进行重试

- **Date:** 2026-08-19
- **Type:** process
- **Scope:** `tests`, `ci`
- **PR:** [#180](https://github.com/Prism-Shadow/agenthub/pull/180)

[English](2026-08-19-e2e-rate-limit-retry.md)

## 变更内容

- Python E2E 任务会重跑被限流的测试：`dev` extra 新增 `pytest-rerunfailures`，`.github/workflows/pytest.yml` 增加 `--reruns 5 --reruns-delay 30 --only-rerun RateLimitError --only-rerun 429`。`--only-rerun` 过滤条件确保其余失败（包括断言失败）仍在首次尝试即判定失败。
- TypeScript E2E 套件在共享的 `modelTest` 包装器内部重试，因此各测试主体无需改动。`withRateLimitRetry` 会在遇到 HTTP 429（或错误信息中出现限流字样）时最多重试 5 次、每次间隔 30 秒（与 pytest 参数保持一致），其他错误则立即抛出。每个测试的 jest 超时时间相应延长以覆盖该重试预算。
