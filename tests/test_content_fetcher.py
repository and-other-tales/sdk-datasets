import pytest
import time  # Added missing import
from unittest.mock import MagicMock, patch
import requests
import threading
from github.content_fetcher import ContentFetcher


@pytest.fixture
def mock_repo_fetcher():
    """Fixture to mock the RepositoryFetcher."""
    with patch("github.content_fetcher.RepositoryFetcher") as MockRepoFetcher:
        yield MockRepoFetcher


@pytest.fixture
def content_fetcher(mock_repo_fetcher):
    """Fixture to create a ContentFetcher instance with a mocked RepositoryFetcher."""
    return ContentFetcher(github_token="mock_token")


def test_fetch_org_repositories(content_fetcher, mock_repo_fetcher):
    """Test fetching organization repositories."""
    mock_repo_fetcher.return_value.fetch_organization_repos.return_value = [
        {"name": "repo1"},
        {"name": "repo2"},
    ]
    progress_mock = MagicMock()

    repos = content_fetcher.fetch_org_repositories(
        "mock_org", progress_callback=progress_mock
    )

    assert len(repos) == 2
    progress_mock.assert_any_call(5)
    progress_mock.assert_any_call(20)


def test_fetch_single_repository(content_fetcher, mock_repo_fetcher):
    """Test fetching a single repository."""
    mock_repo_fetcher.return_value.fetch_single_repo.return_value = {"name": "repo1"}
    progress_mock = MagicMock()

    repo = content_fetcher.fetch_single_repository(
        "https://github.com/mock_org/repo1", progress_callback=progress_mock
    )

    assert repo["name"] == "repo1"
    progress_mock.assert_called_with(50)


def test_fetch_content_for_dataset(content_fetcher, mock_repo_fetcher):
    """Test fetching content for dataset creation."""
    mock_repo_fetcher.return_value.fetch_relevant_content.return_value = [
        "file1.py",
        "file2.py",
    ]
    progress_mock = MagicMock()

    content = content_fetcher.fetch_content_for_dataset(
        "https://github.com/mock_org/repo1",
        branch="main",
        progress_callback=progress_mock,
    )

    assert len(content) == 2
    progress_mock.assert_any_call(10)
    progress_mock.assert_any_call(90)
    progress_mock.assert_any_call(100)


def test_fetch_multiple_repositories(content_fetcher, mock_repo_fetcher):
    """Test fetching content from multiple repositories."""
    mock_repo_fetcher.return_value.fetch_organization_repos.return_value = [
        {"name": "repo1", "owner": {"login": "mock_org"}, "default_branch": "main"},
        {"name": "repo2", "owner": {"login": "mock_org"}, "default_branch": "main"},
    ]
    mock_repo_fetcher.return_value.fetch_relevant_content.side_effect = [
        ["file1.py", "file2.py"],
        ["file3.py", "file4.py"],
    ]
    progress_mock = MagicMock()

    content = content_fetcher.fetch_multiple_repositories(
        "mock_org", progress_callback=progress_mock
    )

    assert len(content) == 4
    progress_mock.assert_any_call(20)
    progress_mock.assert_any_call(70)


def test_fetch_org_repositories_with_cancellation():
    """Test that org repository fetching respects cancellation."""
    with patch("requests.get") as mock_get:
        # Setup mocks
        mock_response = MagicMock()
        mock_response.json.return_value = {"public_repos": 10}
        mock_get.return_value = mock_response
        mock_response.raise_for_status = MagicMock()

        # Create cancellation event that's already set
        cancel_event = threading.Event()
        cancel_event.set()

        # Create content fetcher
        content_fetcher = ContentFetcher(github_token="test_token")
        callback = MagicMock()

        # Call with cancellation event
        result = content_fetcher.fetch_organization_repositories(
            "test_org", callback=callback, _cancellation_event=cancel_event
        )

        # Verify callback was called with cancellation message
        callback.assert_called_with(0, "Operation cancelled")
        # Result should be empty list when cancelled
        assert result == []


def test_fetch_org_repositories_cancels_midway():
    """Test cancellation during repository processing."""
    with patch("requests.get") as mock_get:
        # Mock first response (org info)
        org_response = MagicMock()
        org_response.json.return_value = {"public_repos": 200}
        org_response.raise_for_status = MagicMock()

        # Mock second response (first page of repos)
        repos_response = MagicMock()
        repos_response.json.return_value = [{"name": f"repo{i}"} for i in range(100)]
        repos_response.raise_for_status = MagicMock()

        # Define side effects for the mock requests
        def get_side_effect(*args, **kwargs):
            # First call - organization info
            if "/orgs/test_org" in args[0] and "page" not in args[0]:
                return org_response
            # Second call - first page of repos
            elif "/repos?page=1" in args[0]:
                return repos_response
            # Third call - should never happen due to cancellation
            else:
                raise StopIteration("Mock cancellation")
        
        # Setup mock get with the side effect function
        mock_get.side_effect = get_side_effect

        # Create cancellation event and callback
        cancel_event = threading.Event()
        callback = MagicMock()

        # Create content fetcher
        content_fetcher = ContentFetcher(github_token="test_token")

        # Start in another thread so we can cancel during execution
        def fetch_thread():
            try:
                content_fetcher.fetch_organization_repositories(
                    "test_org", callback=callback, _cancellation_event=cancel_event
                )
            except StopIteration:
                # Handle StopIteration explicitly to prevent it from propagating
                pass

        thread = threading.Thread(target=fetch_thread)
        thread.start()

        # Give it time to process the first batch then cancel
        time.sleep(0.1)
        cancel_event.set()

        # Wait for completion
        thread.join(timeout=1.0)

        # Verify callback was called with progress initially
        callback.assert_any_call(50, "Fetched 100/200 repositories")
