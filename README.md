# Agent Adapter

This repository contains a minimal Python package and a minimal TypeScript package for the Agent Adapter hello world examples.

## Python package (agent-adapter-py)

Python sources live in `src_py/`.

Install dependencies and run the example:

```bash
uv pip install -e .
uv run python -m agent_adapter.hello_world
```

Run linting/tests:

```bash
make quality
make test
```

## TypeScript package (agent-adapter-ts)

TypeScript sources live in `src_ts/`. From the `src_ts` directory:

```bash
npm install
npm run lint
npm run build
npm run start
```
