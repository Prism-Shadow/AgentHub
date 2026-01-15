.PHONY: default install lint build test

default: install

install:
	uv pip install -e .

lint:
	uvx ruff check
	uvx ruff format --check

build:
	uvx build

test:
	uv pip install -e .[dev]
	uv run pytest -vvv tests
