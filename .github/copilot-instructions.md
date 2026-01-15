# Coding Guidelines

## Project Overview

AgentAdapter is a dual-language project containing both Python and TypeScript implementations of a minimal hello world example package.

## Repository Structure

- `src_py/` - Python package source code and configuration
  - `agent_adapter/` - Main Python package
  - `pyproject.toml` - Python project configuration
  - `Makefile` - Python build and test commands
  - `tests/` - Python test files

- `src_ts/` - TypeScript package source code and configuration
  - `src/` - TypeScript source files
  - `package.json` - Node.js package configuration
  - `tsconfig.json` - TypeScript compiler configuration
  - `Makefile` - TypeScript build and test commands

- `docs/` - **Reference documentation for coding agents**
  - See this directory for detailed architecture, development guidelines, and code conventions

## Coding Standards

### Python

- Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- Use `ruff` for linting and formatting
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

## Important Notes

- This is a monorepo containing both Python and TypeScript packages
- Each language has its own dependencies and build process
- When making changes, ensure consistency across both implementations when applicable
- Consult the `docs/` directory for comprehensive reference materials before making architectural changes

## GitHub Workflow Secrets for Testing

When writing tests that require calling AI models, the following secrets are available in GitHub workflows:

- `ANTHROPIC_API_KEY` - API key for Anthropic SDK (Claude models)
- `GEMINI_API_KEY` - API key for Google Gemini SDK
- `OPENAI_API_KEY` - API key for OpenAI SDK
- `OPENAI_BASE_URL` - Base URL for OpenAI API (optional, for custom endpoints)

These secrets can be used in your test code to authenticate with the respective AI model providers. Make sure to handle these credentials securely and never log or expose them in test output.
