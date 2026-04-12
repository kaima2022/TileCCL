.PHONY: install install-dev lint lint-fix format typecheck clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install tncc in editable mode
	pip install -e .

install-dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

lint:  ## Run ruff linter
	ruff check tncc/ examples/

lint-fix:  ## Run ruff linter with auto-fix
	ruff check --fix tncc/ examples/

format:  ## Format code with ruff
	ruff format tncc/ examples/

typecheck:  ## Run mypy type checker
	mypy tncc/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
