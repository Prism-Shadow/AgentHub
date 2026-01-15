.PHONY: default install lint build test

check_dirs := .

default: install

install:
	uv pip install -e .

lint:
	uv run ruff check $(check_dirs)
	uv run ruff format --check $(check_dirs)

build:
	uv build

test:
	uv pip install -e .[dev]
	uv run pytest -vvv tests
