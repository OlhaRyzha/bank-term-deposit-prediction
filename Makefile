sync:
	uv sync

lab:
	uv run jupyter lab

format:
	uv run ruff check . --fix
	uv run ruff format .

lint:
	uv run ruff check .
	uv run ruff format . --check

typecheck:
	uv run mypy

test:
	uv run pytest

check: lint typecheck test

hooks:
	uv run pre-commit run --all-files
	uv run pre-commit run --all-files --hook-stage pre-push
