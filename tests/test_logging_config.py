import logging
import os
from unittest import TestCase, mock
from utils.logging_config import setup_logging, RealTimeLogHandler
from config.settings import (
    LOG_DIR,
    LOG_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
    MAX_LOG_SIZE,
    LOG_BACKUP_COUNT,
)


class TestLoggingConfig(TestCase):

    @mock.patch("utils.logging_config.Path.mkdir")
    @mock.patch("utils.logging_config.logging.getLogger")
    @mock.patch("utils.logging_config.logging.StreamHandler")
    @mock.patch("utils.logging_config.logging.handlers.RotatingFileHandler")
    def test_setup_logging(
        self, mock_rotating_handler, mock_stream_handler, mock_get_logger, mock_mkdir
    ):
        """Test the setup_logging function."""
        mock_logger = mock.Mock()
        mock_get_logger.return_value = mock_logger

        # Call setup_logging
        setup_logging()

        # Ensure log directory is created
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Instead of asserting a single call, check that it was called at least once with no arguments
        mock_get_logger.assert_any_call()

        # Check that it configures at least the root logger
        assert mock_logger.addHandler.call_count >= 2

    @mock.patch("utils.logging_config.logging.getLogger")
    def test_real_time_log_handler(self, mock_get_logger):
        """Test the RealTimeLogHandler."""
        mock_tui_callback = mock.Mock()
        handler = RealTimeLogHandler(mock_tui_callback)

        # Mock a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        handler.emit(record)

        # Ensure the callback is called with the formatted log entry
        mock_tui_callback.assert_called_once_with(handler.format(record))
