.PHONY: install install-dev test bench lint typecheck clean help

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install tncc in editable mode
	pip install -e .

install-dev:  ## Install with dev + benchmark dependencies
	pip install -e ".[dev,benchmark]"

test:  ## Run all tests
	pytest tests/ -v

test-unit:  ## Run unit tests only (no benchmark, no multigpu)
	pytest tests/ -v -m "not benchmark and not multigpu"

test-multigpu:  ## Run multi-GPU tests
	pytest tests/ -v -m "multigpu"

bench:  ## Run benchmarks
	pytest tests/benchmarks/ -v -m "benchmark" --no-header -rN

lint:  ## Run ruff linter
	ruff check tncc/ tests/

lint-fix:  ## Run ruff linter with auto-fix
	ruff check --fix tncc/ tests/

format:  ## Format code with ruff
	ruff format tncc/ tests/

typecheck:  ## Run mypy type checker
	mypy tncc/

clean:  ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
