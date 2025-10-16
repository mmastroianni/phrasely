import logging


def test_basic_caplog_text(caplog):
    """Example 1: Capture all log output as text."""
    logger = logging.getLogger("phrasely.demo")
    with caplog.at_level(logging.INFO):
        logger.debug("This is hidden (below INFO).")
        logger.info("Info message here.")
        logger.warning("Something went wrong.")

    # caplog.text is a single string with all log lines
    assert "Info message here." in caplog.text
    assert "Something went wrong." in caplog.text
    assert "This is hidden" not in caplog.text  # DEBUG not shown


def test_caplog_records_and_messages(caplog):
    """Example 2: Access structured LogRecord objects and messages."""
    logger = logging.getLogger("phrasely.demo.structured")
    with caplog.at_level(logging.WARNING):
        logger.warning("Low disk space.")
        logger.error("Disk failure!")

    # caplog.records gives you actual LogRecord objects
    assert len(caplog.records) == 2
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.records[1].levelname == "ERROR"

    # caplog.messages is just a list of message strings
    assert "Low disk space." in caplog.messages
    assert "Disk failure!" in caplog.messages


def test_dynamic_log_level_change(caplog):
    """Example 3: Temporarily change log level inside a block."""
    logger = logging.getLogger("phrasely.dynamic")

    # Only capture warnings by default
    with caplog.at_level(logging.WARNING):
        logger.warning("This warning is captured.")
        logger.info("This info should be ignored.")

    assert "This warning is captured." in caplog.text
    assert "This info should be ignored." not in caplog.text

    # Now change to capture INFO too
    caplog.clear()  # reset previous logs
    with caplog.at_level(logging.INFO):
        logger.info("Now INFO messages appear.")
    assert "Now INFO messages appear." in caplog.text


def test_combining_with_asserts(caplog):
    """Example 4: Simulate a function that logs and returns data."""

    def example_fn(x):
        logger = logging.getLogger("phrasely.example_fn")
        if x < 0:
            logger.error("Negative input not allowed.")
            return None
        logger.info("Processing complete.")
        return x * 2

    with caplog.at_level(logging.INFO):
        result = example_fn(-5)
        assert result is None
        assert "Negative input not allowed." in caplog.text

        caplog.clear()
        result = example_fn(10)
        assert result == 20
        assert "Processing complete." in caplog.text
