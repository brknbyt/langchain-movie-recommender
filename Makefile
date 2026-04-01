
run:
	uv run python -m cli.main

debug:
	uv run python -m cli.main --debug

lint:
	uv run ruff format . && uv run ruff check . && uv run ty check