# Coding Guidelines

You are a senior software engineer working on the AgentHub project.

## Project Overview

AgentHub is the only SDK you need to connect to state-of-the-art LLMs (GPT-5/Claude 4.5/Gemini 3).

## Repository Structure

- `src_py/` - Python implementation
  - `agenthub/` - Main Python package
  - `pyproject.toml` - Python project configuration
  - `Makefile` - Python build and test commands
  - `tests/` - Python test files

- `src_ts/` - TypeScript implementation
  - `src/` - TypeScript source files
  - `package.json` - Node.js package configuration
  - `tsconfig.json` - TypeScript compiler configuration
  - `Makefile` - TypeScript build and test commands

- `docs/` - **Reference documentation for AI model SDKs**
  - See this directory for detailed development guidelines and code conventions

## Coding Standards

### General Code Quality

- **Avoid trivial comments**: Do not add comments that simply restate what the code obviously does. Comments should explain *why* something is done, not *what* is being done when it's already clear from the code itself.
  - ❌ Bad: `# Add temperature` before `config['temperature'] = 0.7`
  - ❌ Bad: `# Loop through items` before `for item in items:`
  - ✅ Good: `# Claude requires max_tokens to be specified` before `config['max_tokens'] = 1000`
  - ✅ Good: Comments explaining complex algorithms, non-obvious business logic, or workarounds for known issues

### Python

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Maintain Python 3.11+ compatibility
- Run `make lint` and `make test` from `src_py/` before committing

### TypeScript

- Use ESLint for code quality
- Follow TypeScript strict mode conventions
- Run `make lint` and `make test` from `src_ts/` before committing

## Implementation Rules

When adding support for new AI models in `auto_client.py`, follow these rules:

1. **DO NOT** use generic matching like `if "claude" in model.lower()` as this is too broad and may match unintended model names.
2. Put the implementation of the new model in a separate folder with the model identifier as the folder name, such as `claude4_5/` for Claude 4.5 series models.
3. **DO NOT** create new files or directories in examples and tests when adding a new model, use pytest parameters or environment variables instead.
4. **Always** consult the [../llmsdk_docs/README.md](../llmsdk_docs/README.md) for AI model sdk usage details.
5. When making changes, ensure consistency across both implementations when applicable.
6. When using JSON serialization, ensure that CJK strings are serialized correctly by using `ensure_ascii=False`.

When writing documentation, follow these rules:

1. Always provide clear and concise documentation.
2. **DO NOT** include unnecessary details in the documentation.
3. Ensure that the documentation is accurate and up-to-date.
4. Remember to update the documentation whenever changes are made to the code.

## GitHub Workflow Secrets for Testing

When writing tests that require calling AI models, the following secrets are available in GitHub workflows:

- `ANTHROPIC_API_KEY` - API key for Anthropic Claude SDK
- `GEMINI_API_KEY` - API key for Google Gemini SDK
- `OPENAI_API_KEY` - API key for OpenAI SDK
- `GLM_API_KEY` - API key for Z.AI GLM SDK
- `QWEN3_API_KEY` - API key for Qwen3 SDK
- `QWEN3_BASE_URL` - Base URL for Qwen3 SDK

To use these secrets in your workflow files, reference them in the `env:` section:

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  GLM_API_KEY: ${{ secrets.GLM_API_KEY }}
  QWEN3_API_KEY: ${{ secrets.QWEN3_API_KEY }}
  QWEN3_BASE_URL: ${{ secrets.QWEN3_BASE_URL }}
```

These secrets can be used in your test code to authenticate with the respective AI model providers. Make sure to handle these credentials securely and never log or expose them in test output.
