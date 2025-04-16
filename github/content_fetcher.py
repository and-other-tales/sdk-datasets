import logging
import re
import atexit
import signal
from github.repository import RepositoryFetcher
from utils.performance import async_process  # Add this import
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)

# Global executor for background tasks
_global_executor = None


def get_executor(max_workers=3):
    """Get or create a global thread pool executor."""
    global _global_executor
    if _global_executor is None:
        _global_executor = ThreadPoolExecutor(max_workers=max_workers)
    return _global_executor


def shutdown_executor():
    """Shutdown the global executor."""
    global _global_executor
    if _global_executor:
        logger.debug("Shutting down global thread pool executor")
        _global_executor.shutdown(wait=False)
        _global_executor = None


# Register shutdown function
atexit.register(shutdown_executor)

# Register signal handlers for graceful shutdown
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, lambda signum, frame: shutdown_executor())


class ContentFetcher:
    """Fetches and organizes repository content."""

    def __init__(self, github_token=None):
        self.repo_fetcher = RepositoryFetcher(github_token=github_token)
        self.github_token = github_token
        # Create GitHub client using the proper authentication
        self.github_client = self.repo_fetcher.client

    def fetch_organization_repositories(
        self, org_name, callback=None, _cancellation_event=None
    ):
        """
        Fetch repositories from an organization.

        Args:
            org_name: Organization name
            callback: Progress callback function
            _cancellation_event: Event to check for cancellation

        Returns:
            List of repositories
        """
        # Initialize progress
        total_repos = 0
        processed = 0
        page = 1
        all_repos = []

        try:
            # Use the GitHub client directly instead of direct API calls
            # This ensures proper authentication and rate limiting
            logger.info(f"Fetching repositories for organization: {org_name}")
            
            # First get the organization info to get the total repo count
            try:
                # Use proper GitHub client for authentication
                org_info = self.github_client.get(f"orgs/{org_name}")
                total_repos = org_info.get("public_repos", 0)
                
                if callback:
                    callback(0, f"Found {total_repos} repositories in {org_name}")
            except Exception as e:
                logger.error(f"Failed to get organization info for {org_name}: {e}")
                if callback:
                    callback(0, f"Error: {str(e)}")
                raise
            
            # Now fetch all repositories page by page using the authenticated client
            while True:
                # Check for cancellation
                if _cancellation_event and _cancellation_event.is_set():
                    if callback:
                        callback(
                            processed / max(1, total_repos) * 100, "Operation cancelled"
                        )
                    return []
                
                try:
                    # Get repositories for this page using the proper GitHub client
                    repos_page = self.github_client.get_organization_repos(
                        org_name, page=page, per_page=100
                    )
                    
                    if not repos_page:
                        break
                        
                    all_repos.extend(repos_page)
                    processed += len(repos_page)
                    
                    if callback:
                        callback(
                            processed / max(1, total_repos) * 100,
                            f"Fetched {processed}/{total_repos} repositories",
                        )
                    
                    # Check if we've reached the end
                    if len(repos_page) < 100:
                        break
                        
                    page += 1
                except StopIteration:
                    # Handle StopIteration for test mocks that end early
                    break
                
            return all_repos
            
        except Exception as e:
            logger.error(f"Failed to fetch repositories for organization {org_name}: {e}")
            if callback:
                callback(0, f"Error: {str(e)}")
            raise

    def fetch_org_repositories(self, org_name, progress_callback=None):
        """Fetch repositories for an organization."""
        try:
            # Start with initial progress indication
            if progress_callback:
                logger.debug("Sending initial 5% progress update")
                progress_callback(5)

            logger.debug(f"Starting repository fetch for organization: {org_name}")

            repos = self.repo_fetcher.fetch_organization_repos(org_name)

            if progress_callback:
                logger.debug("Sending 20% progress update after repository fetch")
                progress_callback(20)

            logger.debug(f"Fetched {len(repos)} repositories for {org_name}")
            return repos
        except Exception as e:
            logger.error(
                f"Failed to fetch repositories for organization {org_name}: {e}",
                exc_info=True,
            )
            raise

    def fetch_single_repository(self, repo_url, progress_callback=None):
        """Fetch a single repository."""
        try:
            repo = self.repo_fetcher.fetch_single_repo(repo_url)
            if progress_callback:
                progress_callback(50)  # Example progress update
            return repo
        except Exception as e:
            logger.error(f"Failed to fetch repository {repo_url}: {e}")
            raise

    def fetch_content_for_dataset(self, repo_data, branch=None, progress_callback=None):
        """Fetch content suitable for dataset creation."""
        if isinstance(repo_data, str):
            # Handle single repository URL
            match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", repo_data)
            if not match:
                raise ValueError(f"Invalid GitHub repository URL: {repo_data}")
            owner, repo = match.groups()
            repo = repo.rstrip(".git")
        else:
            # Handle repository dict from API
            owner = repo_data["owner"]["login"]
            repo = repo_data["name"]
            if not branch:
                branch = repo_data.get("default_branch")

        logger.info(f"Fetching content for dataset creation: {owner}/{repo}")
        try:
            # Initialize progress at 10% to show activity
            if progress_callback:
                progress_callback(10)

            content_files = self.repo_fetcher.fetch_relevant_content(
                owner, repo, branch, progress_callback=progress_callback
            )

            if progress_callback:
                progress_callback(
                    90
                )  # Better progress indication - fetching is almost done

            logger.info(
                f"Fetched {len(content_files)} relevant files from {owner}/{repo}"
            )

            # Complete progress
            if progress_callback:
                progress_callback(100)

            return content_files
        except Exception as e:
            logger.error(f"Failed to fetch content for {owner}/{repo}: {e}")
            # Make sure we indicate an error through the progress callback
            if progress_callback:
                progress_callback(-1)  # Use negative value to indicate error
            raise

    def fetch_multiple_repositories(self, org_name, progress_callback=None):
        """Fetch content from multiple repositories in an organization."""
        try:
            # Progress sections:
            # 0-20%: Fetch repositories list
            # 20-70%: Process repositories
            # 70-100%: Create dataset

            repos = self.fetch_org_repositories(org_name, progress_callback)

            if not repos:
                logger.warning(f"No repositories found for organization {org_name}")
                if progress_callback:
                    progress_callback(70)  # Skip to the end of this stage
                return []

            logger.info(f"Found {len(repos)} repositories in {org_name}")
            logger.debug(f"Repository names: {[repo['name'] for repo in repos[:5]]}...")

            if progress_callback:
                logger.debug("Updating progress to 20% after finding repositories")
                progress_callback(20)

            # Process each repo with detailed progress
            def fetch_repo_content(repo):
                try:
                    logger.debug(f"Processing repository: {repo['name']}")
                    # Don't pass progress_callback here to avoid conflicts
                    return self.fetch_content_for_dataset(repo)
                except Exception as e:
                    logger.error(
                        f"Error processing repository {repo['name']}: {e}",
                        exc_info=True,
                    )
                    return []

            all_content = []
            batch_size = min(
                5, len(repos)
            )  # Smaller batches for better progress tracking

            try:
                for i in range(0, len(repos), batch_size):
                    # Handle smaller batches
                    batch = repos[i : i + batch_size]
                    batch_names = [repo["name"] for repo in batch]
                    logger.debug(f"Processing batch {i//batch_size + 1}: {batch_names}")

                    # Use direct threading for better control
                    executor = get_executor()
                    futures = [
                        executor.submit(fetch_repo_content, repo) for repo in batch
                    ]

                    batch_content = []
                    for j, future in enumerate(futures):
                        try:
                            result = future.result(timeout=300)  # 5-minute timeout
                            repo_name = batch_names[j]
                            result_count = (
                                len(result) if isinstance(result, list) else 0
                            )
                            logger.debug(
                                f"Repository {repo_name} returned {result_count} files"
                            )

                            if isinstance(result, list):
                                batch_content.extend(result)
                            else:
                                logger.warning(
                                    f"Unexpected result type from {repo_name}: {type(result)}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error in batch processing: {e}", exc_info=True
                            )

                    all_content.extend(batch_content)

                    if progress_callback:
                        # Update progress proportionally (20-70%)
                        progress_percent = 20 + 50 * min(
                            (i + batch_size) / len(repos), 1.0
                        )
                        logger.debug(
                            f"Batch complete, updating progress to {progress_percent:.1f}%"
                        )
                        progress_callback(progress_percent)

                        # Add detailed log for this batch
                        logger.debug(
                            f"Batch {i//batch_size + 1}/{(len(repos)+batch_size-1)//batch_size}: "
                            f"Found {len(batch_content)} files"
                        )

                if progress_callback:
                    logger.debug(
                        "Repository processing complete, updating progress to 70%"
                    )
                    progress_callback(70)

                logger.info(
                    f"Fetched total of {len(all_content)} files from repositories"
                )
                return all_content
            except RuntimeError as e:
                if "cannot schedule new futures" in str(e):
                    logger.warning(
                        "Interpreter is shutting down. Stopping processing early."
                    )
                else:
                    raise
        except Exception as e:
            logger.error(
                f"Failed to fetch multiple repositories for {org_name}: {e}",
                exc_info=True,
            )
            raise
