import logging
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def catch_time(task_name: str = "operation"):
    start = time.time()
    try:
        yield
        duration = time.time() - start
        logger.info(f"{task_name} completed in {duration:.2f}s")
    except Exception as e:
        duration = time.time() - start
        logger.error(
            f"{task_name} failed after {duration:.2f}s: {e}"
        )  # ✅ explicit message
        raise
