import pytest
from unittest.mock import patch, MagicMock
from huggingface.dataset_manager import DatasetManager
import json


@pytest.fixture
def mock_hf_api():
    with patch("huggingface.dataset_manager.HfApi") as MockHfApi:
        yield MockHfApi.return_value


@pytest.fixture
def dataset_manager(mock_hf_api):
    return DatasetManager(huggingface_token="mock_token")


def test_list_datasets_authenticated_user(dataset_manager, mock_hf_api):
    mock_hf_api.whoami.return_value = {"name": "test_user"}
    mock_hf_api.list_datasets.return_value = [{"id": "dataset1"}, {"id": "dataset2"}]

    datasets = dataset_manager.list_datasets()

    assert len(datasets) == 2
    mock_hf_api.whoami.assert_called_once_with("mock_token")
    mock_hf_api.list_datasets.assert_called_once_with(author="test_user")


def test_list_datasets_specific_user(dataset_manager, mock_hf_api):
    mock_hf_api.list_datasets.return_value = [{"id": "dataset1"}]

    datasets = dataset_manager.list_datasets(username="specific_user")

    assert len(datasets) == 1
    mock_hf_api.list_datasets.assert_called_once_with(author="specific_user")


def test_list_datasets_no_token():
    manager = DatasetManager()
    datasets = manager.list_datasets()

    assert datasets == []


def test_get_dataset_info(dataset_manager, mock_hf_api):
    mock_hf_api.dataset_info.return_value = {
        "id": "dataset1",
        "description": "Test dataset",
    }

    info = dataset_manager.get_dataset_info("dataset1")

    assert info["id"] == "dataset1"
    mock_hf_api.dataset_info.assert_called_once_with("dataset1")


def test_get_dataset_info_error(dataset_manager, mock_hf_api):
    mock_hf_api.dataset_info.side_effect = Exception("Error")

    info = dataset_manager.get_dataset_info("dataset1")

    assert info is None


def test_delete_dataset(dataset_manager, mock_hf_api):
    result = dataset_manager.delete_dataset("dataset1")

    assert result is True
    mock_hf_api.delete_repo.assert_called_once_with(
        "dataset1", repo_type="dataset", token="mock_token"
    )


def test_delete_dataset_no_token():
    manager = DatasetManager()
    result = manager.delete_dataset("dataset1")

    assert result is False


def test_download_dataset_metadata_with_metadata_json(
    dataset_manager, mock_hf_api, tmp_path
):
    mock_hf_api.hf_hub_download.return_value = True

    result = dataset_manager.download_dataset_metadata("dataset1", output_dir=tmp_path)

    assert result is True
    mock_hf_api.hf_hub_download.assert_called_once_with(
        repo_id="dataset1",
        filename="metadata.json",
        repo_type="dataset",
        local_dir=tmp_path,
        token="mock_token",
    )


def test_download_dataset_metadata_no_metadata_json(
    dataset_manager, mock_hf_api, tmp_path
):
    mock_hf_api.hf_hub_download.side_effect = Exception("Not found")
    mock_hf_api.dataset_info.return_value = MagicMock(
        id="dataset1",
        description="Test dataset",
        created_at=None,
        last_modified=None,
        tags=["tag1", "tag2"],
        downloads=100,
        likes=10,
    )

    result = dataset_manager.download_dataset_metadata("dataset1", output_dir=tmp_path)

    assert result is True
    with open(tmp_path / "dataset_info.json") as f:
        metadata = json.load(f)
        assert metadata["name"] == "dataset1"
        assert metadata["tags"] == ["tag1", "tag2"]


def test_download_dataset_metadata_error(dataset_manager, mock_hf_api):
    mock_hf_api.hf_hub_download.side_effect = Exception("Error")
    mock_hf_api.dataset_info.side_effect = Exception("Error")

    result = dataset_manager.download_dataset_metadata("dataset1")

    assert result is False
