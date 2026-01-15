# Agent Adapter

This repository contains a minimal Python package and a minimal TypeScript package for the Agent Adapter hello world examples.

## Python package (agent-adapter)

Python sources live in `src_py/`.

Install dependencies and run the example from `src_py/`:

```bash
cd src_py
make
uv run python -m agent_adapter.hello_world
```

Run linting/tests:

```bash
make lint
make test
```

## TypeScript package (agent-adapter)

TypeScript sources live in `src_ts/`. From the `src_ts` directory:

```bash
make
make lint
make build
make test
npm run start
```
