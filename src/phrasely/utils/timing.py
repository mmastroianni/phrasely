import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def catch_time(label: str):
    """Context manager to measure and log elapsed time for a code block."""
    start = time.perf_counter()
    try:
        yield
    except Exception as e:
        logger.error(f"[{label}] failed after {time.perf_counter() - start:.3f}s: {e}")
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info(f"[{label}] completed in {elapsed:.3f}s.")
