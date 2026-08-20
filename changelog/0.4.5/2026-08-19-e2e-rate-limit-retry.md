# Live E2E tests retry rate-limited responses

- **Date:** 2026-08-19
- **Type:** process
- **Scope:** `tests`, `ci`
- **PR:** [#180](https://github.com/Prism-Shadow/agenthub/pull/180)

[中文版](2026-08-19-e2e-rate-limit-retry.zh.md)

## What changed

- The Python E2E job reruns rate-limited tests: `pytest-rerunfailures` was added to the
  `dev` extra, and `.github/workflows/pytest.yml` gained
  `--reruns 5 --reruns-delay 30 --only-rerun RateLimitError --only-rerun 429`. The
  `--only-rerun` filters keep every other failure, including assertion failures, failing
  on its first attempt.
- The TypeScript E2E suite retries inside the shared `modelTest` wrapper, so each test
  body runs unchanged. `withRateLimitRetry` re-runs a body that raised HTTP 429 (or an
  error whose message names a rate limit) up to 5 times, 30s apart, matching the pytest
  flags, and rethrows every other error immediately. Each test's jest deadline is
  extended by that retry budget.
