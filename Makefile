# Makefile for Phrasely (GPU-default)

.PHONY: install test lint format clean

install:
	@pip install -e .[gpu,dev]

test:
	@pytest -v --disable-warnings

lint:
	@flake8 src/phrasely tests
	@mypy src/phrasely

format:
	@black src tests

clean:
	@rm -rf build dist .pytest_cache *.egg-info

# ==============================
# Release Automation
# Usage:
#   make release version=0.2.0 message="Add PhraseDatasetLoader"
# ==============================

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

# ==============================
# Preview next changelog entry (no commit, no tag)
# Usage:
#   make changelog version=0.2.0 message="Add PhraseDatasetLoader"
# ==============================

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
