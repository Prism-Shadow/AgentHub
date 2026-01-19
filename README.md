# AgentHub

AgentHub is the only SDK you need to connect to state-of-the-art LLMs (GPT-5/Claude 4.5/Gemini 3).

## Supported Models

| Model Name | Supports Thinking Model | Supports Tool Calling | Supports Image Understanding |
|------------|------------------------|----------------------|------------------------------|
| Gemini 3 | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| Claude 4.5 | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| GLM-4.7 | :white_check_mark: | :white_check_mark: | :white_check_mark: |
| GPT-5.2 | :white_check_mark: | :white_check_mark: | :white_check_mark: |

## Python package

Python sources live in `src_py/`.

Install from PyPI:

```bash
uv add agenthub-python
# or
uv pip install agenthub-python
```

Build from source:

```bash
cd src_py
make
```

See [src_py/README.md](src_py/README.md) for comprehensive usage examples and API documentation.

## TypeScript package

TypeScript sources live in `src_ts/`.

```bash
cd src_ts
make
npm run start
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
