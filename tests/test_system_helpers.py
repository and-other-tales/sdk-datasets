import pytest
import threading
import signal
import os
import time
from concurrent.futures import ThreadPoolExecutor

from utils.system_helpers import (
    register_executor,
    register_thread,
    unregister_thread,
    cleanup_resources,
    handle_signal,
    safe_thread,
    create_managed_executor,
    _threads_to_join,  # Import this internal variable for testing
)


def test_register_executor():
    executor = ThreadPoolExecutor(max_workers=1)
    registered_executor = register_executor(executor)
    assert registered_executor == executor
    cleanup_resources()


def test_register_thread():
    thread = threading.Thread(target=lambda: None)
    registered_thread = register_thread(thread)
    assert registered_thread == thread
    cleanup_resources()


def test_unregister_thread():
    thread = threading.Thread(target=lambda: None)
    register_thread(thread)
    unregister_thread(thread)
    cleanup_resources()


def test_cleanup_resources():
    executor = ThreadPoolExecutor(max_workers=1)
    register_executor(executor)
    thread = threading.Thread(target=lambda: None)
    register_thread(thread)
    cleanup_resources()
    assert not executor._threads  # Executor threads should be shut down
    assert not thread.is_alive()  # Thread should be joined


def test_cleanup_resources_with_active_thread():
    """Test cleanup_resources with a thread that's still running."""
    # Clear any existing threads that might be in the list
    _threads_to_join.clear()

    def long_running_task():
        # Sleep long enough for the test to detect it's running
        time.sleep(0.2)

    # Create and register a thread
    thread = threading.Thread(target=long_running_task)
    thread.daemon = True  # Make it daemon so test doesn't hang
    register_thread(thread)
    thread.start()

    # Ensure the thread is registered
    assert thread in _threads_to_join

    # Call cleanup_resources
    cleanup_resources()

    # Thread should still be in list since it's still running
    assert thread in _threads_to_join

    # Wait for thread to complete
    thread.join()

    # Call cleanup again after thread is done
    cleanup_resources()

    # Now the thread should be removed
    assert thread not in _threads_to_join


def test_handle_signal(monkeypatch):
    def mock_kill(pid, signum):
        assert pid == os.getpid()
        assert signum == signal.SIGTERM

    monkeypatch.setattr(os, "kill", mock_kill)
    handle_signal(signal.SIGTERM, None)


def test_safe_thread():
    """Test safe_thread decorator."""
    event_finished = threading.Event()

    @safe_thread(daemon=True)
    def dummy_function():
        time.sleep(0.1)  # Small delay to ensure thread stays alive for the test
        event_finished.set()

    thread = dummy_function()
    assert thread.is_alive(), "Thread should be alive immediately after starting"

    # Optional: Wait and verify thread completes successfully
    event_finished.wait(timeout=1)
    assert event_finished.is_set(), "Thread should have completed execution"


def test_create_managed_executor():
    executor = create_managed_executor(max_workers=2, thread_name_prefix="test")
    assert isinstance(executor, ThreadPoolExecutor)
    cleanup_resources()
