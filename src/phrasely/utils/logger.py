import logging
import sys


class ColorFormatter(logging.Formatter):
    """Colorized log output for console readability."""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[41m",  # Red background
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logger(name: str = "phrasely", level: int = logging.INFO):
    """Return a configured logger with both console and file output."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid reinitializing if already set

    logger.setLevel(level)

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColorFormatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger
