# The e2e suites run one client per worker

- **Date:** 2026-08-19
- **Type:** process
- **Scope:** `tests`, `workflows`, `pyproject`, `jest.config`
- **PR:** [#174](https://github.com/Prism-Shadow/agenthub/pull/174)

[中文版](2026-08-19-parallel-e2e-suites.zh.md)

## What changed

- `src_py/tests/conftest.py` marks every test carrying a `model` parameter with an
  `xdist_group` keyed on that model's id, and the Python suite runs under
  `pytest -n 16 --dist loadgroup` (`src_py/Makefile` and `.github/workflows/pytest.yml`).
  Distinct models run on separate workers; the tests of one model stay serial on the
  worker that owns its group, because providers rate-limit per client. `pytest-xdist>=3.6.0`
  joins the `dev` extra in `src_py/pyproject.toml`.
- `src_ts/tests/client.test.ts` declares each check through a `modelTest` helper that opens
  a describe block per check holding one `test.concurrent` per model. jest-circus runs
  describe blocks one after another and the tests inside one block concurrently, so the
  models of a check run in parallel while a single model never has two checks in flight.
  The helper takes its timeout ahead of the body.
- `src_ts/jest.config.js` raises `maxConcurrency` to 64 so the concurrency cap clears the
  model count instead of the default 5.

## Test names

The TypeScript e2e names regrouped from model-then-check to check-then-model:

| Before | After |
| --- | --- |
| `Client tests for <model> > <check>` | `Client tests > <check> > <model>` |

`npm run test -- -t "<model>"` still selects one model's tests, and
`uv run pytest -k "<model>"` is unaffected.
