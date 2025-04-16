import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from exceptions.github_exceptions import GitHubAPIError


@pytest.fixture
def repository_fetcher():
    """Fixture to create a RepositoryFetcher with mocked client."""
    from github.repository import RepositoryFetcher

    mock_client = MagicMock()
    return RepositoryFetcher(github_token=None, client=mock_client)


def test_process_file_success(repository_fetcher, tmp_path):
    """Test _process_file when the file is successfully fetched and saved."""
    owner = "test_owner"
    repo = "test_repo"
    branch = "main"
    base_dir = tmp_path
    file_info = {
        "name": "test_file.txt",
        "path": "docs/test_file.txt",
        "sha": "abc123",
        "size": 1024,
        "html_url": "https://github.com/test_owner/test_repo/blob/main/docs/test_file.txt",
    }
    file_content = "This is a test file."

    # Mock the GitHub client to return file content
    repository_fetcher.client.get_repository_file.return_value = file_content

    result = repository_fetcher._process_file(owner, repo, file_info, branch, base_dir)

    # Verify the file was saved correctly
    saved_file = Path(base_dir) / file_info["name"]
    assert saved_file.exists()
    assert saved_file.read_text() == file_content

    # Check required fields in the result
    assert result["name"] == file_info["name"]
    assert result["path"] == file_info["path"]
    assert result["local_path"] == str(saved_file)
    assert result["size"] == file_info["size"]
    assert result["url"] == file_info["html_url"]
    assert result["sha"] == file_info["sha"]
    # These extra fields are fine
    assert result["repo"] == f"{owner}/{repo}"
    assert result["branch"] == branch


def test_process_file_github_api_error(repository_fetcher, tmp_path):
    """Test _process_file when a GitHubAPIError occurs."""
    owner = "test_owner"
    repo = "test_repo"
    branch = "main"
    base_dir = tmp_path
    file_info = {
        "name": "test_file.txt",
        "path": "docs/test_file.txt",
        "sha": "abc123",
        "size": 1024,
        "html_url": "https://github.com/test_owner/test_repo/blob/main/docs/test_file.txt",
    }

    # Mock the GitHub client to raise an error
    repository_fetcher.client.get_repository_file.side_effect = GitHubAPIError(
        "API error"
    )

    # Call the method and check that it handles the exception
    result = repository_fetcher._process_file(owner, repo, file_info, branch, base_dir)

    # Verify the file wasn't saved and the result contains the error
    saved_file = Path(base_dir) / file_info["name"]
    assert not saved_file.exists()
    assert "error" in result
    assert "API error" in result["error"]
