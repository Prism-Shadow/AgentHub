# Development Guide

## Getting Started

### Prerequisites

**For Python development:**
- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

**For TypeScript development:**
- Node.js 18 or higher
- npm 9 or higher

### Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/Prism-Shadow/AgentAdapter.git
cd AgentAdapter
```

2. Choose your development path:

**Python Development:**
```bash
cd src_py
uv sync --dev
```

**TypeScript Development:**
```bash
cd src_ts
npm install
```

## Development Workflows

### Python Workflow

From the `src_py/` directory:

**Install dependencies:**
```bash
make
# or manually: uv sync --dev
```

**Run the application:**
```bash
uv run python -m agent_adapter.hello_world
```

**Run tests:**
```bash
make test
# or manually: uv run pytest
```

**Run linting:**
```bash
make lint
# or manually: uv run ruff check .
```

**Format code:**
```bash
uv run ruff format .
```

### TypeScript Workflow

From the `src_ts/` directory:

**Install dependencies:**
```bash
make
# or manually: npm install
```

**Build the project:**
```bash
make build
# or manually: npm run build
```

**Run the application:**
```bash
make test  # This runs the start script
# or manually: npm run start
```

**Run linting:**
```bash
make lint
# or manually: npm run lint
```

## Testing Strategy

### Python Testing

- Framework: `pytest`
- Test location: `src_py/tests/`
- Naming convention: `test_*.py`
- Run command: `make test` from `src_py/`

**Writing tests:**
```python
def test_function_name():
    # Arrange
    expected = "expected value"
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected
```

### TypeScript Testing

- Currently: Placeholder script
- Future: Can be extended with Jest, Mocha, or Vitest
- Test location: To be created in `src_ts/tests/` or `src_ts/src/__tests__/`

## Build Process

### Python Build

The Python package uses `uv_build` as the build backend:
- Configuration in `pyproject.toml`
- Module root is the directory containing `agent_adapter/`
- Build artifacts are managed by uv

### TypeScript Build

The TypeScript package compiles to JavaScript:
- Configuration in `tsconfig.json`
- Output directory: `dist/`
- Entry point after build: `dist/index.js`

## Code Quality Tools

### Python

**Ruff** handles both linting and formatting:
- Target version: Python 3.11
- Line length: 119 characters
- Configuration in `[tool.ruff]` section of `pyproject.toml`

### TypeScript

**ESLint** for code quality:
- Configuration in `eslint.config.cjs`
- Uses TypeScript ESLint plugin
- Parser: `@typescript-eslint/parser`

## Continuous Integration

Before committing:

1. **Python changes:**
   ```bash
   cd src_py
   make lint && make test
   ```

2. **TypeScript changes:**
   ```bash
   cd src_ts
   make lint && make build
   ```

## Troubleshooting

### Python Issues

**Module not found errors:**
```bash
cd src_py
uv sync --dev
```

**Linting failures:**
```bash
uv run ruff check . --fix
uv run ruff format .
```

### TypeScript Issues

**Build failures:**
```bash
cd src_ts
rm -rf node_modules dist
npm install
npm run build
```

**Type errors:**
- Check `tsconfig.json` settings
- Ensure all dependencies have type definitions

## Making Changes

1. Create a new branch from `main`
2. Make changes in the appropriate directory (`src_py/` or `src_ts/`)
3. Run linting and tests
4. Commit with descriptive messages
5. Push and create a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.
