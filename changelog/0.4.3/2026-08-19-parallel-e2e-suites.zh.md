# e2e 测试套件按 client 分配到各自的 worker

- **Date:** 2026-08-19
- **Type:** process
- **Scope:** `tests`, `workflows`, `pyproject`, `jest.config`
- **PR:** [#174](https://github.com/Prism-Shadow/agenthub/pull/174)

[English](2026-08-19-parallel-e2e-suites.md)

## 变更内容

- `src_py/tests/conftest.py` 为每个带 `model` 参数的测试打上以该模型 id 为键的 `xdist_group`
  标记，Python 套件改为在 `pytest -n 16 --dist loadgroup` 下运行（`src_py/Makefile` 与
  `.github/workflows/pytest.yml`）。不同模型分散到不同 worker；同一模型的测试则串行留在拥有
  该分组的 worker 上，因为服务方按 client 限流。`pytest-xdist>=3.6.0` 加入
  `src_py/pyproject.toml` 的 `dev` 附加依赖。
- `src_ts/tests/client.test.ts` 通过 `modelTest` 辅助函数声明每一项检查：每项检查是一个
  describe 块，其中每个模型对应一个 `test.concurrent`。jest-circus 逐个执行 describe 块、
  并发执行块内的测试，因此一项检查的各模型并行运行，而同一模型不会同时进行两项检查。该辅助函数
  把超时参数放在测试体之前。
- `src_ts/tests/client.test.ts` 在 `beforeAll` 钩子中预热 gaxios 延迟导入的 fetch，并把 `gaxios`
  加入 `devDependencies`。Jest 对整个 runtime 只维护一个 `isInsideTestCode` 标志，且在任意并发
  兄弟测试结束时就会清除它；gaxios（Vertex 服务账号鉴权所用的传输层）在首次请求时执行的动态
  `import()` 因此会让正在进行的测试以
  `ReferenceError: You are trying to import a file outside of the scope of the test code` 失败。
- `src_ts/jest.config.js` 把 `maxConcurrency` 提高到 64，使并发上限超过模型数量，而不是默认的 5。

## 测试名称

TypeScript e2e 的名称从「模型在前、检查在后」改为「检查在前、模型在后」：

| 变更前 | 变更后 |
| --- | --- |
| `Client tests for <model> > <check>` | `Client tests > <check> > <model>` |

`npm run test -- -t "<model>"` 仍可筛选出单个模型的测试，`uv run pytest -k "<model>"` 不受影响。
