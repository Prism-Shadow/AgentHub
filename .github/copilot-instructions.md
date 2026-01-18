# Coding Guidelines

You are a senior software engineer working on the AgentHub project.

## Project Overview

AgentHub is the only SDK you need to connect to state-of-the-art LLMs (GPT-5/Claude 4.5/Gemini 3).

## Repository Structure

- `src_py/` - Python implementation
  - `agent_hub/` - Main Python package
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

### Python

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Maintain Python 3.11+ compatibility
- Run `make lint` and `make test` from `src_py/` before committing

### TypeScript

- Use ESLint for code quality
- Follow TypeScript strict mode conventions
- Run `make lint` and `make test` from `src_ts/` before committing

## Development Workflow

1. Both Python and TypeScript code live in separate directories with their own build systems
2. Each directory has its own `Makefile` for common tasks
3. Refer to `docs/` directory for detailed development guidelines and architecture decisions
4. Always check CONTRIBUTING.md for the contribution process

## Implementation Rules

When adding support for new AI models in `auto_client.py`, follow these rules:

1. **DO NOT** use generic matching like `if "claude" in model.lower()` as this is too broad and may match unintended model names.
2. Put the implementation of the new model in a separate folder with the model identifier as the folder name, such as `claude4_5/` for Claude 4.5 series models.
3. **DO NOT** create new files or directories in examples and tests when adding a new model, use pytest parameters or environment variables instead.
4. **Always** consult the [../docs/README.md](../docs/README.md) for AI model sdk usage details.
5. When making changes, ensure consistency across both implementations when applicable.

## GitHub Workflow Secrets for Testing

When writing tests that require calling AI models, the following secrets are available in GitHub workflows:

- `ANTHROPIC_API_KEY` - API key for Anthropic SDK (Claude models)
- `GEMINI_API_KEY` - API key for Google Gemini SDK
- `OPENAI_API_KEY` - API key for OpenAI SDK
- `OPENAI_BASE_URL` - Base URL for OpenAI API (optional, for custom endpoints)

To use these secrets in your workflow files, reference them in the `env:` section:

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
  OPENAI_BASE_URL: ${{ secrets.OPENAI_BASE_URL }}
```

These secrets can be used in your test code to authenticate with the respective AI model providers. Make sure to handle these credentials securely and never log or expose them in test output.
