import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def catch_time(label: str):
    """Context manager that logs both start and completion of timed code blocks."""
    logger.info(f"▶️  {label}...")
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{label} completed in {elapsed:.3f}s.")
