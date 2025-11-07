# ============================================================
# Makefile for Phrasely – GPU-default local dev, CPU-safe CI
# ============================================================

.PHONY: install install-cpu test lint format clean release changelog ci help strip-notebooks lint-fix

# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

help:
	@echo ""
	@echo "🧭 Phrasely Makefile Commands"
	@echo "──────────────────────────────────────────────"
	@echo " install         → Install GPU + dev dependencies"
	@echo " install-cpu     → Install CPU-only (no RAPIDS/cuML)"
	@echo " test            → Run unit tests with pytest"
	@echo " lint            → Run flake8 + mypy checks"
	@echo " format          → Format code with black + isort"
	@echo " lint-fix        → Auto-fix lint issues (format only)"
	@echo " clean           → Remove build/test caches"
	@echo " ci              → Local CI simulation (CPU-only)"
	@echo " release         → Create a tagged version and update CHANGELOG"
	@echo " changelog       → Preview next CHANGELOG entry (no commit)"
	@echo " help            → Show this help message"
	@echo ""

# ------------------------------------------------------------
# Basic Commands
# ------------------------------------------------------------

install:
	@echo "📦 Installing Phrasely with GPU + dev dependencies..."
	pip install -e .[gpu,dev]

install-cpu:
	@echo "📦 Installing Phrasely in CPU-only mode..."
	pip install -e .[dev]
	echo "USE_GPU=0" > .env

test:
	@echo "🧪 Running pytest..."
	pytest -v --disable-warnings

# ------------------------------------------------------------
# Linting and Formatting
# ------------------------------------------------------------

lint:
	@echo "🔍 Running flake8 + mypy..."
	flake8 --config .flake8 src/phrasely tests
	mypy --config-file mypy.ini src/phrasely

format:
	@echo "🪄 Formatting with Black + isort..."
	isort src tests
	black src tests

# A helper target for “fix problems then lint”
lint-fix: format
	@echo "✨ Re-formatting done. Now run: make lint"

strip-notebooks:
	find notebooks -name '*.ipynb' -exec nbstripout {} \;

clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build dist .pytest_cache .mypy_cache *.egg-info

# ------------------------------------------------------------
# CI Simulation (CPU mode)
# ------------------------------------------------------------

ci: clean install-cpu
	@echo "🚀 Running CI-style tests (CPU-only)..."
	pytest -v --disable-warnings
	flake8 --config .flake8 src/phrasely tests
	mypy --config-file mypy.ini src/phrasely

# ------------------------------------------------------------
# Release Automation
# Usage:
#   make release version=0.2.0 message="Add PhraseDatasetLoader"
# ------------------------------------------------------------

release:
	@if [ -z "$(version)" ]; then \
		echo "❌ Please specify a version: make release version=0.2.0 message='...'" && exit 1; \
	fi
	@if [ -z "$(message)" ]; then \
		echo "❌ Please specify a message: make release version=$(version) message='...'" && exit 1; \
	fi
	@echo "🏷️  Creating release $(version): $(message)"
	@echo "\n## [v$(version)] – $$(date +%Y-%m-%d)\n### Added\n- $(message)\n" | cat - CHANGELOG.md > CHANGELOG.tmp && mv CHANGELOG.tmp CHANGELOG.md
	@git add CHANGELOG.md
	@git commit -m "Update CHANGELOG for v$(version)"
	@git tag -a v$(version) -m "$(message)"
	@git push origin main
	@git push origin v$(version)
	@echo "\n✅ Release v$(version) created and pushed successfully."

# ------------------------------------------------------------
# Preview next changelog entry (no commit, no tag)
# Usage:
#   make changelog version=0.2.0 message="Add PhraseDatasetLoader"
# ------------------------------------------------------------

changelog:
	@if [ -z "$(version)" ]; then \
		echo "❌ Please specify a version: make changelog version=0.2.0 message='...'" && exit 1; \
	fi
	@if [ -z "$(message)" ]; then \
		echo "❌ Please specify a message: make changelog version=$(version) message='...'" && exit 1; \
	fi
	@echo "📝 Preview changelog entry for v$(version):"
	@echo "------------------------------------------"
	@echo "## [v$(version)] – $$(date +%Y-%m-%d)"
	@echo "### Added"
	@echo "- $(message)"
	@echo "------------------------------------------"
