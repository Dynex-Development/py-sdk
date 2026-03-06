.PHONY: help install sync test test-unit test-integration \
        format check lint pre-commit-check build clean lock

UV      := $(shell command -v uv 2>/dev/null || echo $(HOME)/.local/bin/uv)
VENV    = .venv
PYTEST  = $(UV) run pytest
BLACK   = $(UV) run black
ISORT   = $(UV) run isort
FLAKE8  = $(UV) run flake8
PRECOMMIT = $(UV) run pre-commit

# ─── Help ────────────────────────────────────────────────────────────────────

help:
	@echo "Dynex SDK — available make targets"
	@echo ""
	@echo "Setup:"
	@echo "  make install           Install SDK + dev dependencies (first-time setup)"
	@echo "  make sync              Re-sync dependencies from uv.lock"
	@echo "  make lock              Regenerate uv.lock"
	@echo ""
	@echo "Testing:"
	@echo "  make test              Run all tests (unit + integration)"
	@echo "  make test-unit         Run unit tests only"
	@echo "  make test-integration  Run integration tests (requires .env credentials)"
	@echo ""
	@echo "Code quality:"
	@echo "  make format            Format code with black + isort"
	@echo "  make check             Check formatting without changes"
	@echo "  make lint              Run flake8 linter"
	@echo "  make pre-commit-check  Run all pre-commit hooks"
	@echo ""
	@echo "Build:"
	@echo "  make build             Build wheel and sdist"
	@echo "  make clean             Remove build artifacts and venv"

# ─── Setup ───────────────────────────────────────────────────────────────────

install:
	@command -v $(UV) >/dev/null 2>&1 || { \
	  echo "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh"; \
	  exit 1; }
	$(UV) sync --group dev
	@echo ""
	@echo "Done. Run commands with:  uv run <cmd>"
	@echo "Or activate the venv:     source $(VENV)/bin/activate"
	@echo ""
	@echo "For integration tests copy .env.example to .env and add credentials."

sync:
	$(UV) sync --group dev

lock:
	$(UV) lock

# ─── Tests ───────────────────────────────────────────────────────────────────

test:
	@echo "========================================="
	@echo "Running unit tests..."
	@echo "========================================="
	$(PYTEST) tests/unit/ -v
	@echo ""
	@echo "========================================="
	@echo "Running integration tests..."
	@echo "========================================="
	RUN_INTEGRATION_TESTS=true $(PYTEST) tests/integration/ -v

test-unit:
	$(PYTEST) tests/unit/ -v

test-integration:
	@if [ ! -f ".env" ]; then echo "Warning: .env file not found. Copy .env.example to .env"; fi
	RUN_INTEGRATION_TESTS=true $(PYTEST) tests/integration/ -v

# ─── Code quality ────────────────────────────────────────────────────────────

format:
	@echo "Formatting with black..."
	@$(BLACK) . --quiet
	@echo "Sorting imports with isort..."
	@$(ISORT) . --quiet
	@echo "[DONE] Code formatted"

check:
	@echo "Checking formatting with black..."
	@$(BLACK) --check --diff --quiet .
	@echo "Checking import sorting with isort..."
	@$(ISORT) --check-only --diff --quiet .
	@echo "[PASS] Formatting OK"

lint:
	@echo "Running flake8..."
	@$(FLAKE8) --max-line-length=120 --extend-ignore=E203,W503 dynex/ tests/ \
	  && echo "[PASS] No linting errors"

pre-commit-check:
	$(PRECOMMIT) run --all-files

# ─── Build ───────────────────────────────────────────────────────────────────

build:
	$(UV) build

# ─── Clean ───────────────────────────────────────────────────────────────────

clean:
	rm -rf $(VENV) dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ coverage.xml
	rm -rf tmp/
