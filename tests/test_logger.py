import logging

from phrasely.utils.logger import setup_logger


def test_logger_basic(caplog):
    """Ensure logger outputs messages and respects log levels."""
    logger = setup_logger("phrasely_test", level=logging.DEBUG)

    with caplog.at_level(logging.DEBUG):
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")

    logs = [rec.message for rec in caplog.records]
    assert "debug message" in logs
    assert "info message" in logs
    assert "warning message" in logs


def test_logger_idempotent():
    """Ensure setup_logger doesn't create duplicate handlers."""
    logger1 = setup_logger("phrasely_test")
    handler_count_1 = len(logger1.handlers)

    logger2 = setup_logger("phrasely_test")
    handler_count_2 = len(logger2.handlers)

    # The handler count should remain stable after repeated setup
    assert handler_count_1 == handler_count_2
