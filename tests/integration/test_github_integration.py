import pytest
import os
from github.client import GitHubClient
from github.content_fetcher import ContentFetcher


@pytest.mark.skipif(not os.environ.get("GITHUB_TOKEN"), reason="GITHUB_TOKEN not set")
def test_fetch_real_repository():
    """Integration test for fetching content from a real GitHub repository."""
    # Use a small, public repo for this test
    test_repo_url = "https://github.com/huggingface/datasets"

    # Create a real content fetcher with the GitHub token from environment
    content_fetcher = ContentFetcher(github_token=os.environ.get("GITHUB_TOKEN"))

    # Only fetch a small sample for testing purposes
    content = content_fetcher.fetch_single_repository(
        repo_url=test_repo_url,
        max_files=3,  # Limit files to not overload API
        progress_callback=None,
    )

    # Verify we got valid content
    assert isinstance(content, list)
    assert len(content) > 0
    assert all("name" in item for item in content)
    assert all("path" in item for item in content)
