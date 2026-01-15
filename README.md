# Python Template

A template for Python projects.

## 1. Install uv project manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Follow https://docs.astral.sh/uv/ to install uv project manager on other operating systems.

## 2. Change the project name

Replace all occurrences of `custom_proj` with your new project name.

```
custom_proj/
    ├── __init__.py
    ├── hello_world.py
    └── openrouter.py

# change to

new_name/
    ├── __init__.py
    ├── hello_world.py
    └── openrouter.py
```

And pyproject.toml:

```toml
name = "custom-proj"

# change to

name = "new-name"
```

## 3. Run the example

```bash
make
uv run python examples/invoke.py
```

## 4. Run the tests

```bash
make test
```

## 5. Push to GitHub

See [CONTRIBUTING.md](CONTRIBUTING.md) for more details.
