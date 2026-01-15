.PHONY: default build commit quality style test

check_dirs := .

default:
	cd src_py && uv pip install -e .

build:
	cd src_py && uv build

commit:
	cd src_py && uv run pre-commit install --config ../.pre-commit-config.yaml
	cd src_py && uv run pre-commit run --all-files --config ../.pre-commit-config.yaml

quality:
	cd src_py && uv run ruff check $(check_dirs)
	cd src_py && uv run ruff format --check $(check_dirs)

style:
	cd src_py && uv run ruff check $(check_dirs) --fix
	cd src_py && uv run ruff format $(check_dirs)

test:
	cd src_py && uv pip install -e .[dev]
	cd src_py && uv run pytest -vvv tests
