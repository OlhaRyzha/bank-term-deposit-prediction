setup:
	uv sync
	$(MAKE) notebook-diff
	uv run pre-commit install --hook-type pre-commit --hook-type pre-push

notebook-diff:
	uv run nbdime config-git --enable
	git config --local diff.jupyternotebook.command "uv run git-nbdiffdriver diff --ignore-outputs --ignore-metadata --ignore-id"

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
