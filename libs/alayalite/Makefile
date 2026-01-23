.PHONY: lint lint_package lint_tests format test tests test_watch integration_tests help

# Define source directories
PYTHON_FILES = langchain_alayalite tests

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  lint            Run all linting checks"
	@echo "  lint_package    Lint the package source code"
	@echo "  lint_tests      Lint the test files"
	@echo "  format          Auto-format code with ruff"
	@echo "  test            Run unit tests"
	@echo "  tests           Alias for test"
	@echo "  integration_tests  Run integration tests"

lint: lint_package lint_tests

lint_package:
	uv run ruff check langchain_alayalite
	uv run ruff format langchain_alayalite --check
	uv run mypy langchain_alayalite

lint_tests:
	uv run ruff check tests
	uv run ruff format tests --check

format:
	uv run ruff check $(PYTHON_FILES) --fix
	uv run ruff format $(PYTHON_FILES)

test:
	uv run pytest tests/unit_tests

tests: test

integration_tests:
	uv run pytest tests/integration_tests
