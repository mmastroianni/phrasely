# tests/conftest.py
import logging
import pytest

@pytest.fixture(autouse=True, scope="session")
def configure_test_logging():
    """
    Configure consistent log formatting for all tests.
    Runs automatically once per test session.
    """
    log_format = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
    date_format = "%H:%M:%S"
    logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # silence noisy libs
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
