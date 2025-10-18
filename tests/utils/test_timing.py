import logging
import time

import pytest

from phrasely.utils.timing import catch_time


def test_catch_time_logs_info(caplog):
    """Ensure catch_time logs start and completion messages."""
    caplog.set_level(logging.INFO)
    with catch_time("mock task"):
        time.sleep(0.01)

    messages = [r.message for r in caplog.records]
    assert any("mock task" in msg for msg in messages)
    assert any("completed" in msg for msg in messages)


def test_catch_time_handles_exceptions(caplog):
    """Ensure catch_time logs errors if the wrapped block raises."""
    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError):
        with catch_time("failing task"):
            raise ValueError("intentional failure")

    messages = [r.message for r in caplog.records]
    assert any("failing task" in msg for msg in messages)
    assert any("failed" in msg for msg in messages)
