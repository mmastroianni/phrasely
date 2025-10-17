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


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Set up a clean, single-stream logger (no color, no duplication)."""
    # Remove any pre-existing handlers (e.g., Jupyter root handler)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    # Configure base logging once
    logging.basicConfig(
        level=level,
        format="%(message)s",  # clean, no color or level prefixes
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Return a module-specific logger
    logger = logging.getLogger(name)
    logger.propagate = False  # prevent double emission through root
    logger.setLevel(level)
    return logger
