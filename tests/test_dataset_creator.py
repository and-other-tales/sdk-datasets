import pytest
from unittest.mock import MagicMock, patch
import threading
import time
from datasets import Dataset
from huggingface.dataset_creator import DatasetCreator


@pytest.fixture
def mock_file_processor():
    with patch("huggingface.dataset_creator.FileProcessor") as MockFileProcessor:
        yield MockFileProcessor.return_value


@pytest.fixture
def mock_metadata_generator():
    with patch(
        "huggingface.dataset_creator.MetadataGenerator"
    ) as MockMetadataGenerator:
        yield MockMetadataGenerator.return_value


@pytest.fixture
def dataset_creator(mock_file_processor, mock_metadata_generator):
    return DatasetCreator(huggingface_token="mock_token")


def test_create_dataset_success(
    mock_file_processor, mock_metadata_generator, dataset_creator
):
    mock_file_processor.process_files.return_value = [
        {"text": "Sample text", "metadata": {"key": "value"}}
    ]
    mock_metadata_generator.generate_dataset_metadata.return_value = {
        "description": "Test dataset"
    }
    mock_metadata_generator.generate_repo_structure_metadata.return_value = {
        "structure": "mock"
    }

    dataset = dataset_creator.create_dataset(
        file_data_list=[{"path": "file1.txt"}],
        dataset_name="test_dataset",
        description="Test description",
    )

    assert isinstance(dataset, Dataset)
    assert len(dataset) == 1
    assert dataset["text"][0] == "Sample text"


def test_create_dataset_no_files(mock_file_processor, dataset_creator):
    mock_file_processor.process_files.return_value = []

    dataset = dataset_creator.create_dataset(
        file_data_list=[{"path": "file1.txt"}], dataset_name="test_dataset"
    )

    assert dataset is None


def test_push_to_hub_success(dataset_creator):
    mock_dataset = MagicMock()
    mock_dataset.push_to_hub.return_value = None

    success = dataset_creator.push_to_hub(
        dataset=mock_dataset, repo_name="test_repo", private=True
    )

    assert success is True
    mock_dataset.push_to_hub.assert_called_once_with(
        "test_repo", token="mock_token", private=True, commit_message="Upload dataset"
    )


def test_push_to_hub_no_token():
    creator = DatasetCreator(huggingface_token=None)
    mock_dataset = MagicMock()

    success = creator.push_to_hub(
        dataset=mock_dataset, repo_name="test_repo", private=True
    )

    assert success is False


def test_create_and_push_dataset_success(
    mock_file_processor, mock_metadata_generator, dataset_creator
):
    # Setup
    mock_file_processor.process_files.return_value = [
        {"text": "Sample text", "metadata": {"key": "value"}}
    ]
    mock_metadata_generator.generate_dataset_metadata.return_value = {
        "description": "Test dataset"
    }
    mock_metadata_generator.generate_repo_structure_metadata.return_value = {
        "structure": "mock"
    }

    # Create a single mock object to be used consistently
    mock_dataset = MagicMock()
    mock_dataset.cast_column.return_value = (
        mock_dataset  # Make chained methods return the same mock
    )

    # Mock Dataset.from_dict to return our mock_dataset
    with patch(
        "huggingface.dataset_creator.Dataset.from_dict", return_value=mock_dataset
    ) as mock_from_dict:
        # Also mock any potential chained methods
        with patch.object(mock_dataset, "cast_column", return_value=mock_dataset):
            success, dataset = dataset_creator.create_and_push_dataset(
                file_data_list=[{"path": "file1.txt"}],
                dataset_name="test_dataset",
                description="Test description",
                private=True,
            )

    # Assertions
    assert success is True
    assert dataset == mock_dataset  # Use equality check instead of identity


def test_create_dataset_from_repository_with_cancellation(dataset_creator):
    """Test dataset creation with cancellation."""
    # Create a cancellation event that's already set
    cancel_event = threading.Event()
    cancel_event.set()

    # Create a progress callback mock
    progress_callback = MagicMock()

    # Call the method with cancellation
    result = dataset_creator.create_dataset_from_repository(
        repo_url="https://github.com/test/repo",
        dataset_name="test_dataset",
        description="Test dataset",
        progress_callback=progress_callback,
        _cancellation_event=cancel_event,
    )

    # Verify result
    assert result["success"] is False
    assert "cancelled" in result["message"].lower()

    # Verify callback was called with cancellation message
    progress_callback.assert_called_with(10, "Operation cancelled")


def test_create_dataset_from_repository_cancel_during_processing(dataset_creator):
    """Test dataset creation with cancellation during processing."""
    cancel_event = threading.Event()

    # Mock repo processing to simulate delay and check cancellation
    def mock_processing(progress_callback, _cancellation_event):
        progress_callback(30, "Processing files...")
        time.sleep(0.2)  # Introduce delay to allow cancellation event to trigger
        if _cancellation_event.is_set():
            return False
        progress_callback(50, "Finishing up...")
        return True

    with patch.object(dataset_creator, "_process_repository") as mock_process:
        mock_process.side_effect = lambda *args, **kwargs: mock_processing(
            kwargs.get("progress_callback"), kwargs.get("_cancellation_event")
        )

        progress_callback = MagicMock()
        cancel_event = threading.Event()
        results = {}

        def create_thread():
            results["result"] = dataset_creator.create_dataset_from_repository(
                repo_url="https://github.com/test/repo",
                dataset_name="test_dataset",
                description="Test dataset",
                progress_callback=progress_callback,
                _cancellation_event=cancel_event,
            )

        thread = threading.Thread(target=create_thread)
        thread.start()

        # Allow processing to start, then trigger cancellation
        time.sleep(0.1)
        cancel_event.set()

        thread.join(timeout=1.0)

        progress_callback.assert_any_call(30, "Processing files...")

        assert results["result"]["success"] is False
        assert "cancelled" in results["result"]["message"].lower()
