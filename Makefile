PYTHON := python3

.PHONY: install lint test clean help docs-serve docs-build

install:
	$(PYTHON) -m pip install -e ".[dev,test]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest --cov=src tests/

help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make lint        - Run linters"
	@echo "  make test        - Run tests"
	@echo "  make demo        - Run the demo presets script"
	@echo "  make docs-serve  - Serve documentation locally (http://127.0.0.1:8000)"
	@echo "  make docs-build  - Build static documentation site to site/"
	@echo "  make clean       - Remove build artifacts, python cache, and demo outputs"

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build

demo:
	$(PYTHON) -m demos

clean:
	@echo "Cleaning up..."
	# Python cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	# Project specific outputs
	rm -rf runs/
	rm -rf output_demos/
	rm -rf dist build
	rm -rf .coverage
	@echo "Clean complete."
