import pytest
import requests
import time
from unittest.mock import patch, MagicMock
from github.client import GitHubClient, GitHubAPIError, RateLimitError
from config.settings import GITHUB_API_URL, GITHUB_TIMEOUT


@pytest.fixture
def github_client():
    """Fixture to create a GitHubClient instance."""
    return GitHubClient(token="test_token")


@patch("github.client.requests.Session.get")
def test_get_success(mock_get, github_client):
    """Test successful GET request."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}
    mock_get.return_value = mock_response

    response = github_client.get("test_endpoint")
    assert response == {"key": "value"}
    mock_get.assert_called_once()


@patch("github.client.requests.Session.get")
def test_get_rate_limit_error(mock_get, github_client):
    """Test rate limit error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "rate limit exceeded"
    mock_response.headers = {"X-RateLimit-Reset": str(int(time.time()) + 60)}
    mock_get.return_value = mock_response

    with pytest.raises(RateLimitError):
        github_client.get("test_endpoint")


@patch("github.client.requests.Session.get")
def test_get_api_error(mock_get, github_client):
    """Test API error handling."""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_get.return_value = mock_response

    with pytest.raises(GitHubAPIError):
        github_client.get("test_endpoint")


@patch("github.client.requests.Session.get")
def test_get_organization_repos(mock_get, github_client):
    """Test fetching organization repositories."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"name": "repo1"}, {"name": "repo2"}]
    mock_get.return_value = mock_response

    repos = github_client.get_organization_repos("test_org")
    assert repos == [{"name": "repo1"}, {"name": "repo2"}]
    mock_get.assert_called_once_with(
        f"{GITHUB_API_URL}/orgs/test_org/repos",
        headers=github_client.headers,
        params={"page": 1, "per_page": 100},
        timeout=GITHUB_TIMEOUT,
    )


@patch("github.client.requests.Session.get")
def test_get_repository(mock_get, github_client):
    """Test fetching a single repository."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"name": "test_repo"}
    mock_get.return_value = mock_response

    repo = github_client.get_repository("test_owner", "test_repo")
    assert repo == {"name": "test_repo"}
    mock_get.assert_called_once()


@patch("github.client.requests.Session.get")
def test_get_repository_contents(mock_get, github_client):
    """Test fetching repository contents."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [{"name": "file1"}, {"name": "file2"}]
    mock_get.return_value = mock_response

    contents = github_client.get_repository_contents("test_owner", "test_repo")
    assert contents == [{"name": "file1"}, {"name": "file2"}]
    mock_get.assert_called_once()


@patch("github.client.requests.Session.get")
def test_get_repository_file(mock_get, github_client):
    """Test fetching a repository file."""
    # Create two mock responses
    mock_content_response = MagicMock()
    mock_content_response.status_code = 200
    mock_content_response.json.return_value = {
        "download_url": "http://example.com/file"
    }

    mock_download_response = MagicMock()
    mock_download_response.status_code = 200
    mock_download_response.text = "file content"

    # Configure the mock to return different responses for different calls
    mock_get.side_effect = [mock_content_response, mock_download_response]

    # Call the function under test
    file_content = github_client.get_repository_file(
        "test_owner", "test_repo", "test_path"
    )

    # Verify the result
    assert file_content == "file content"
    assert mock_get.call_count == 2
