import pytest
from unittest.mock import Mock
from utils.error_handler import ErrorHandler
from github.client import GitHubAPIError, RateLimitError


def test_format_error_github_api_error():
    exception = GitHubAPIError("API failure")
    result = ErrorHandler.format_error(exception)
    assert result == "GitHub API error: API failure"


def test_format_error_rate_limit_error():
    exception = RateLimitError("Rate limit exceeded")
    result = ErrorHandler.format_error(exception)
    assert (
        result
        == "GitHub API rate limit exceeded. Please try again later or use a different token."
    )


def test_format_error_value_error():
    exception = ValueError("Invalid input")
    result = ErrorHandler.format_error(exception)
    assert result == "Input error: Invalid input"


def test_format_error_file_not_found_error():
    exception = FileNotFoundError("File missing")
    result = ErrorHandler.format_error(exception)
    assert result == "File not found: File missing"


def test_format_error_permission_error():
    exception = PermissionError("Access denied")
    result = ErrorHandler.format_error(exception)
    assert result == "Permission error: Access denied"


def test_format_error_generic_error():
    exception = Exception("Generic error")
    result = ErrorHandler.format_error(exception)
    assert result == "Error: Generic error"


def test_log_exception(caplog):
    exception = ValueError("Test exception")
    with caplog.at_level("ERROR"):
        ErrorHandler.log_exception(exception)
    assert "Exception: ValueError: Test exception" in caplog.text
    assert "Traceback:" in caplog.text


def test_handle_exception_with_display_callback():
    exception = ValueError("Test exception")
    mock_display_callback = Mock()
    result = ErrorHandler.handle_exception(
        exception, display_callback=mock_display_callback
    )
    mock_display_callback.assert_called_once_with("Input error: Test exception")
    assert result is False


def test_handle_exception_without_display_callback(caplog):
    exception = ValueError("Test exception")
    with caplog.at_level("ERROR"):
        result = ErrorHandler.handle_exception(exception)
    assert "Exception: ValueError: Test exception" in caplog.text
    assert result is False
