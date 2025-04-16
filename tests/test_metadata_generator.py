import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from processors.metadata_generator import MetadataGenerator


@pytest.fixture
def metadata_generator():
    return MetadataGenerator()


def test_generate_dataset_metadata_with_string_source(metadata_generator):
    source_info = "https://github.com/example/repo"
    file_count = 10
    metadata = metadata_generator.generate_dataset_metadata(source_info, file_count)

    assert metadata["source_type"] == "repository"
    assert metadata["source_name"] == "repo"
    assert metadata["file_count"] == file_count
    assert "created_at" in metadata
    assert metadata["description"] == "Dataset created from GitHub repository repo"


def test_generate_dataset_metadata_with_dict_source(metadata_generator):
    source_info = {"full_name": "example/repo"}
    file_count = 5
    metadata = metadata_generator.generate_dataset_metadata(source_info, file_count)

    assert metadata["source_type"] == "repository"
    assert metadata["source_name"] == "example/repo"
    assert metadata["file_count"] == file_count
    assert "created_at" in metadata
    assert (
        metadata["description"] == "Dataset created from GitHub repository example/repo"
    )


def test_generate_file_metadata_with_valid_file(metadata_generator, tmp_path):
    file_path = tmp_path / "test_file.txt"
    file_path.write_text("Sample content")
    file_data = {
        "local_path": str(file_path),
        "name": "test_file.txt",
        "path": "folder/test_file.txt",
        "repo": "example/repo",
        "sha": "dummysha",
        "url": "https://example.com/test_file.txt",
    }

    metadata = metadata_generator.generate_file_metadata(file_data)

    assert metadata["filename"] == "test_file.txt"
    assert metadata["path"] == "folder/test_file.txt"
    assert metadata["repo"] == "example/repo"
    assert metadata["sha"] == "dummysha"
    assert metadata["size_bytes"] == file_path.stat().st_size
    assert metadata["hash"] is not None
    assert metadata["last_modified"] is not None
    assert metadata["url"] == "https://example.com/test_file.txt"
    assert metadata["extension"] == ".txt"


def test_generate_file_metadata_with_missing_local_path(metadata_generator):
    file_data = {
        "name": "test_file.txt",
        "path": "folder/test_file.txt",
        "repo": "example/repo",
        "error": "File not found",
    }

    metadata = metadata_generator.generate_file_metadata(file_data)

    assert metadata["filename"] == "test_file.txt"
    assert metadata["path"] == "folder/test_file.txt"
    assert metadata["repo"] == "example/repo"
    assert metadata["error"] == "File not found"


@patch("processors.metadata_generator.Path.read_bytes")
@patch("processors.metadata_generator.Path.stat")
def test_generate_file_metadata_with_error(
    mock_stat, mock_read_bytes, metadata_generator
):
    mock_read_bytes.side_effect = Exception("Read error")
    file_data = {
        "local_path": "invalid/path",
        "name": "test_file.txt",
        "path": "folder/test_file.txt",
        "repo": "example/repo",
    }

    metadata = metadata_generator.generate_file_metadata(file_data)

    assert metadata["filename"] == "test_file.txt"
    assert metadata["path"] == "folder/test_file.txt"
    assert metadata["repo"] == "example/repo"
    assert "error" in metadata


def test_generate_repo_structure_metadata(metadata_generator):
    file_data_list = [
        {"repo": "repo1", "path": "folder1/file1.txt", "size": 100},
        {"repo": "repo1", "path": "folder1/file2.txt", "size": 200},
        {"repo": "repo1", "path": "folder2/file3.py", "size": 300},
        {"repo": "repo2", "path": "file4.md", "size": 400},
    ]

    metadata = metadata_generator.generate_repo_structure_metadata(file_data_list)

    assert "repo1" in metadata
    assert metadata["repo1"]["file_count"] == 3
    assert metadata["repo1"]["total_size_bytes"] == 600
    assert metadata["repo1"]["file_types"][".txt"] == 2
    assert metadata["repo1"]["file_types"][".py"] == 1
    assert "folder1" in metadata["repo1"]["directories"]
    assert "folder2" in metadata["repo1"]["directories"]

    assert "repo2" in metadata
    assert metadata["repo2"]["file_count"] == 1
    assert metadata["repo2"]["total_size_bytes"] == 400
    assert metadata["repo2"]["file_types"][".md"] == 1
    assert metadata["repo2"]["directories"] == []
