import pytest
import threading
import time
from unittest.mock import MagicMock, patch
from utils.performance import distributed_process, BackgroundTask


def double_value(x):
    return x * 2


def test_distributed_process_single_node():
    """Test distributed_process with a single node."""
    items = list(range(10))
    result = distributed_process(items, double_value, rank=0, world_size=1)
    assert result == [x * 2 for x in items], "Processing failed for single node"


def test_distributed_process_multiple_nodes():
    """Test distributed_process with multiple nodes."""
    items = list(range(10))
    rank = 0
    world_size = 2
    result = distributed_process(items, double_value, rank=rank, world_size=world_size)
    expected = [x * 2 for x in items[rank::world_size]]
    assert result == expected, f"Processing failed for node {rank} in multi-node setup"


def test_distributed_process_empty_items():
    """Test distributed_process with an empty list of items."""
    items = []
    result = distributed_process(items, double_value, rank=0, world_size=1)
    assert result == [], "Processing failed for empty items"


def test_distributed_process_progress_callback():
    """Test distributed_process with a progress callback."""
    items = list(range(10))
    progress_updates = []

    def progress_callback(progress):
        progress_updates.append(progress)

    result = distributed_process(
        items, double_value, rank=0, world_size=1, progress_callback=progress_callback
    )
    assert result == [x * 2 for x in items], "Processing failed with progress callback"
    assert len(progress_updates) > 0, "Progress callback was not called"


def test_distributed_process_invalid_rank():
    """Test distributed_process with an invalid rank."""
    items = list(range(10))
    rank = 5
    world_size = 2
    result = distributed_process(items, double_value, rank=rank, world_size=world_size)
    assert result == [], "Processing failed to handle invalid rank correctly"


def test_background_task_initialization():
    """Test BackgroundTask initialization."""
    on_complete = MagicMock()
    on_error = MagicMock()

    task = BackgroundTask(
        target=lambda: None,
        args=(1, 2),
        kwargs={"a": 3},
        on_complete=on_complete,
        on_error=on_error,
    )

    assert task.target is not None
    assert task.args == (1, 2)
    assert task.kwargs == {"a": 3}
    assert task.on_complete == on_complete
    assert task.on_error == on_error
    assert not task._is_cancelled.is_set()


def test_background_task_start_and_complete():
    """Test that BackgroundTask starts and calls on_complete."""
    result_value = "completed"
    on_complete = MagicMock()

    def target_func():
        return result_value

    task = BackgroundTask(target=target_func, on_complete=on_complete)

    future = task.start()
    # Wait for task to complete
    future.result()

    on_complete.assert_called_once_with(result_value)
    assert not task.is_running()


def test_background_task_error_handling():
    """Test that BackgroundTask handles errors."""
    on_error = MagicMock()

    def target_func():
        raise ValueError("Test error")

    task = BackgroundTask(target=target_func, on_error=on_error)

    future = task.start()
    # Wait for task to complete with error
    with pytest.raises(ValueError):
        future.result()

    on_error.assert_called_once()
    assert "Test error" in str(on_error.call_args[0][0])


def test_background_task_cancellation():
    """Test that BackgroundTask can be cancelled."""
    completed = threading.Event()

    def target_func(_cancellation_event=None):
        # Loop that checks for cancellation
        for i in range(10):
            if _cancellation_event and _cancellation_event.is_set():
                return "cancelled"
            if i == 9:
                completed.set()
            time.sleep(0.1)
        return "completed"

    task = BackgroundTask(target=target_func)
    future = task.start()

    # Cancel the task before it completes
    time.sleep(0.2)  # Let it run a bit
    assert task.stop()

    # Verify it was cancelled
    assert task.is_cancelled()
    assert not completed.is_set()  # Shouldn't have completed normally


def test_background_task_is_running():
    """Test is_running behavior."""
    running_event = threading.Event()
    completion_event = threading.Event()

    def slow_func():
        running_event.set()
        time.sleep(0.2)
        completion_event.set()
        return True

    task = BackgroundTask(target=slow_func)
    task.start()

    # Wait until the task is definitely running
    running_event.wait(timeout=1.0)
    assert task.is_running()

    # Wait for completion
    completion_event.wait(timeout=1.0)
    # Give a small delay for future to update
    time.sleep(0.1)
    assert not task.is_running()
