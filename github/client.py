import time
import logging
import requests
import random
import threading
from requests.exceptions import RequestException, ConnectionError, ReadTimeout
from http.client import RemoteDisconnected
from urllib3.exceptions import ProtocolError
from config.settings import (
    GITHUB_API_URL,
    GITHUB_MAX_RETRIES,
    GITHUB_TIMEOUT,
    GITHUB_DOWNLOAD_RETRIES,
)

logger = logging.getLogger(__name__)


class GitHubAPIError(Exception):
    """Exception raised for GitHub API errors."""

    pass


class RateLimitError(GitHubAPIError):
    """Exception raised when GitHub API rate limit is reached."""

    pass


class GitHubClient:
    """Client for interacting with GitHub API with improved rate limiting."""

    # Class-level rate limiting
    request_lock = threading.Lock()
    last_request_time = 0
    min_request_interval = 1.0  # Minimum time between requests
    requests_per_hour = 5000  # GitHub's limit for authenticated users
    current_requests = 0
    hour_start_time = time.time()

    def __init__(self, token=None):
        self.token = token
        self.headers = {"Accept": "application/vnd.github.v3+json"}
        if token:
            # GitHub API accepts both formats but "Bearer" is more modern and standard OAuth format
            self.headers["Authorization"] = f"Bearer {token}"
        self.session = requests.Session()

    def get(self, endpoint, params=None):
        """Make a GET request to GitHub API with proper rate limiting."""
        url = f"{GITHUB_API_URL}/{endpoint.lstrip('/')}"
        retries = 0

        # Check hourly rate limit
        with GitHubClient.request_lock:
            current_time = time.time()
            elapsed_since_hour_start = current_time - GitHubClient.hour_start_time

            # Reset hourly counter if an hour has passed
            if elapsed_since_hour_start > 3600:
                GitHubClient.hour_start_time = current_time
                GitHubClient.current_requests = 0
                logger.debug("Resetting hourly rate limit counter")

            # If we're approaching the limit, slow down dramatically
            if GitHubClient.current_requests > (GitHubClient.requests_per_hour * 0.9):
                remaining_limit = (
                    GitHubClient.requests_per_hour - GitHubClient.current_requests
                )
                if remaining_limit <= 10:
                    wait_time = max((3600 - elapsed_since_hour_start), 60)
                    logger.warning(
                        f"Rate limit nearly exhausted. Waiting {wait_time:.0f}s."
                    )
                    raise RateLimitError(
                        f"GitHub API rate limit nearly exhausted. Please wait {wait_time/60:.1f} minutes before trying again."
                    )

        while retries < GITHUB_MAX_RETRIES:
            # Apply rate limiting between requests
            with GitHubClient.request_lock:
                current_time = time.time()
                elapsed = current_time - GitHubClient.last_request_time
                if elapsed < GitHubClient.min_request_interval:
                    sleep_time = GitHubClient.min_request_interval - elapsed
                    logger.debug(
                        f"Rate limiting: waiting {sleep_time:.2f}s before next request"
                    )
                    time.sleep(sleep_time)

                # Update last request time
                GitHubClient.last_request_time = time.time()
                GitHubClient.current_requests += 1

            try:
                response = self.session.get(
                    url, headers=self.headers, params=params, timeout=GITHUB_TIMEOUT
                )

                # Check remaining rate limit
                remaining = int(response.headers.get("X-RateLimit-Remaining", "1"))
                if remaining <= 100:
                    logger.warning(
                        f"GitHub API rate limit low: {remaining} requests remaining"
                    )
                    # Slow down even more when we're close to the limit
                    GitHubClient.min_request_interval = max(
                        GitHubClient.min_request_interval, 2.0
                    )

                if response.status_code == 200:
                    return response.json()
                elif (
                    response.status_code == 403
                    and "rate limit exceeded" in response.text.lower()
                ):
                    reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
                    wait_time = max(reset_time - time.time(), 0) + 5  # Add buffer

                    # If wait time is too long, notify the user instead of blocking
                    if wait_time > 120:  # More than 2 minutes
                        message = f"GitHub API rate limit exceeded. Try again after {wait_time/60:.1f} minutes."
                        logger.error(message)
                        raise RateLimitError(message)

                    logger.warning(
                        f"Rate limit exceeded. Waiting for {wait_time:.0f} seconds."
                    )

                    if retries < GITHUB_MAX_RETRIES - 1:
                        time.sleep(min(wait_time, 30))  # Wait max 30 seconds
                        retries += 1
                        continue
                    else:
                        raise RateLimitError(
                            "GitHub API rate limit exceeded. Please try again later."
                        )
                else:
                    try:
                        error_message = response.json().get("message", "Unknown error")
                    except:
                        error_message = response.text[:100]
                    logger.error(
                        f"GitHub API error: {response.status_code} - {error_message}"
                    )
                    raise GitHubAPIError(
                        f"GitHub API error: {response.status_code} - {error_message}"
                    )

            except RequestException as e:
                logger.error(f"Request error: {e}")
                if retries < GITHUB_MAX_RETRIES - 1:
                    retries += 1
                    # Exponential backoff with jitter
                    backoff_time = (2**retries) + (0.1 * random.random())
                    time.sleep(backoff_time)
                    continue
                raise GitHubAPIError(f"Failed to connect to GitHub API: {e}")

        raise GitHubAPIError("Maximum retries reached")

    def get_organization_repos(self, org_name, page=1, per_page=100):
        """Get repositories for a GitHub organization."""
        logger.info(f"Fetching repositories for organization: {org_name}")
        try:
            return self.get(
                f"orgs/{org_name}/repos", {"page": page, "per_page": per_page}
            )
        except GitHubAPIError as e:
            logger.error(f"Failed to fetch repositories for {org_name}: {e}")
            raise

    def get_repository(self, owner, repo):
        """Get a single repository."""
        logger.info(f"Fetching repository: {owner}/{repo}")
        try:
            return self.get(f"repos/{owner}/{repo}")
        except GitHubAPIError as e:
            logger.error(f"Failed to fetch repository {owner}/{repo}: {e}")
            raise

    def get_repository_contents(self, owner, repo, path="", ref=None):
        """Get contents of a repository directory."""
        logger.debug(f"Fetching contents for {owner}/{repo}/{path}")
        params = {}
        if ref:
            params["ref"] = ref

        try:
            return self.get(f"repos/{owner}/{repo}/contents/{path}", params)
        except GitHubAPIError as e:
            logger.error(f"Failed to fetch contents for {owner}/{repo}/{path}: {e}")
            raise

    def get_repository_file(self, owner, repo, path, ref=None):
        """Get the raw content of a file."""
        logger.debug(f"Fetching file content for {owner}/{repo}/{path}")
        content_data = self.get_repository_contents(owner, repo, path, ref)

        if isinstance(content_data, dict) and "download_url" in content_data:
            # Special retry logic for file downloads
            download_retries = GITHUB_DOWNLOAD_RETRIES  # More retries for downloads
            retry_count = 0

            while retry_count < download_retries:
                try:
                    # Apply rate limiting for download as well
                    with GitHubClient.request_lock:
                        current_time = time.time()
                        elapsed = current_time - GitHubClient.last_request_time
                        if elapsed < GitHubClient.min_request_interval:
                            sleep_time = GitHubClient.min_request_interval - elapsed
                            time.sleep(sleep_time)
                        GitHubClient.last_request_time = time.time()

                    download_timeout = (
                        GITHUB_TIMEOUT * 2
                    )  # Double timeout for downloads
                    response = self.session.get(
                        content_data["download_url"], timeout=download_timeout
                    )
                    response.raise_for_status()
                    return response.text
                except (
                    ConnectionError,
                    ReadTimeout,
                    RemoteDisconnected,
                    ProtocolError,
                ) as e:
                    retry_count += 1
                    if retry_count < download_retries:
                        # Exponential backoff with jitter
                        backoff_time = min(30, (2**retry_count) + (random.random() * 2))
                        logger.warning(
                            f"Connection error downloading {path}, "
                            f"retrying in {backoff_time:.2f}s ({retry_count}/{download_retries}): {e}"
                        )
                        time.sleep(backoff_time)
                    else:
                        logger.error(
                            f"Failed to download file after {download_retries} retries: {e}"
                        )
                        raise GitHubAPIError(
                            f"Failed to download file content after {download_retries} retries: {e}"
                        )
                except RequestException as e:
                    logger.error(f"Failed to download file content: {e}")
                    raise GitHubAPIError(f"Failed to download file content: {e}")

            raise GitHubAPIError(f"Maximum retries reached for downloading {path}")
        else:
            raise GitHubAPIError(f"Unexpected content data format for {path}")
