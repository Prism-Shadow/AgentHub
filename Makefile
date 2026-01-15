.PHONY: default build commit quality style test

check_dirs := .

default:
	uv pip install -e .

build:
	uv build

commit:
	uv run pre-commit install
	uv run pre-commit run --all-files

quality:
	uv run ruff check $(check_dirs)
	uv run ruff format --check $(check_dirs)

style:
	uv run ruff check $(check_dirs) --fix
	uv run ruff format $(check_dirs)

test:
	uv pip install -e .[tests]
	uv run pytest -vvv tests
