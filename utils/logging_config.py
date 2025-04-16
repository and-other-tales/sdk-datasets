import logging
import logging.handlers
import sys
from pathlib import Path
from config.settings import (
    LOG_DIR,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE,
    MAX_LOG_SIZE,
    LOG_BACKUP_COUNT,
)


class RealTimeLogHandler(logging.Handler):
    """Custom log handler to send log messages to the TUI in real-time."""

    def __init__(self, tui_callback):
        super().__init__()
        self.tui_callback = tui_callback

    def emit(self, record):
        log_entry = self.format(record)
        self.tui_callback(log_entry)


def setup_logging(tui_callback=None):
    """Configure logging for the application."""
    # Ensure log directory exists
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Create root logger
    root_logger = logging.getLogger()
    # Set to DEBUG to capture all logs
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    root_logger.handlers = []

    # Create formatter with more detail for debugging
    formatter = logging.Formatter(LOG_FORMAT)
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # Create console handler with DEBUG level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(detailed_formatter)
    console_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)

    # Create file handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(detailed_formatter)
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)

    # Create real-time log handler if callback is provided
    if tui_callback:
        real_time_handler = RealTimeLogHandler(tui_callback)
        real_time_handler.setFormatter(formatter)
        real_time_handler.setLevel(logging.DEBUG)  # Set to DEBUG for maximum detail
        root_logger.addHandler(real_time_handler)

    # Set lower level for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("github").setLevel(logging.DEBUG)  # Set to DEBUG
    logging.getLogger("huggingface_hub").setLevel(logging.DEBUG)  # Set to DEBUG

    logging.info("Logging configured with DEBUG level for real-time updates")
