import pytest
import os
import uuid
from huggingface.dataset_creator import DatasetCreator
from huggingface.dataset_manager import DatasetManager


@pytest.mark.skipif(
    not os.environ.get("HUGGINGFACE_TOKEN"), reason="HUGGINGFACE_TOKEN not set"
)
def test_create_and_delete_dataset():
    """Integration test for creating and cleaning up a dataset on HuggingFace Hub."""
    # Create unique dataset name to avoid conflicts
    dataset_name = f"test_dataset_{uuid.uuid4().hex[:8]}"

    # Create real dataset creator with Hugging Face token
    dataset_creator = DatasetCreator(
        huggingface_token=os.environ.get("HUGGINGFACE_TOKEN")
    )
    dataset_manager = DatasetManager(
        huggingface_token=os.environ.get("HUGGINGFACE_TOKEN")
    )

    # Create minimal test dataset
    test_data = [
        {"text": "Sample text 1", "metadata": {"source": "test"}},
        {"text": "Sample text 2", "metadata": {"source": "test"}},
    ]

    try:
        # Create and push dataset
        success, dataset = dataset_creator.create_and_push_dataset(
            file_data_list=[],  # No actual file data needed for this test
            dataset_name=dataset_name,
            description="Test dataset for integration testing",
            private=True,  # Make it private to reduce visibility
            _test_data=test_data,  # Pass test data directly for integration testing
        )

        assert success is True
        assert dataset is not None

        # Verify it exists on Hugging Face Hub
        datasets = dataset_manager.list_datasets()
        dataset_names = [d.get("name", "") for d in datasets]
        assert dataset_name in dataset_names

    finally:
        # Clean up: delete test dataset
        dataset_manager.delete_dataset(dataset_name)
