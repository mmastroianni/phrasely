# Makefile for Phrasely (GPU-default)


.PHONY: install test lint format clean


install:
pip install -e .[gpu,dev]


test:
pytest -v --disable-warnings


lint:
flake8 src/phrasely tests
mypy src/phrasely


format:
black src tests


clean:
rm -rf build dist .pytest_cache *.egg-info
