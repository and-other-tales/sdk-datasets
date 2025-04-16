import re
import os
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from github.client import GitHubClient, GitHubAPIError
from config.settings import (
    RELEVANT_FOLDERS,
    IGNORED_DIRS,
    TEXT_FILE_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    GITHUB_DEFAULT_BRANCH,
    CACHE_DIR,
)

logger = logging.getLogger(__name__)


class RepositoryFetcher:
    """Handles fetching repositories and their contents."""

    def __init__(self, github_token=None, client=None):
        """Initialize the repository fetcher.

        Args:
            github_token (str, optional): GitHub token for authentication
            client (GitHubClient, optional): Existing GitHub client to use
        """
        self.client = client if client is not None else GitHubClient(token=github_token)
        self.cache_dir = CACHE_DIR

    def fetch_organization_repos(self, org_name):
        """Fetch all repositories for an organization."""
        logger.info(f"Fetching repositories for organization: {org_name}")
        repos = []
        page = 1

        while True:
            batch = self.client.get_organization_repos(org_name, page=page)
            if not batch:
                break

            repos.extend(batch)
            if len(batch) < 100:  # Less than max per page, we're done
                break

            page += 1

        logger.info(f"Found {len(repos)} repositories for {org_name}")
        return repos

    def fetch_single_repo(self, repo_url):
        """Fetch a single repository from its URL."""
        # Parse owner and repo from URL
        match = re.match(r"https?://github\.com/([^/]+)/([^/]+)", repo_url)
        if not match:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        owner, repo = match.groups()
        repo = repo.rstrip(".git")

        logger.info(f"Fetching repository: {owner}/{repo}")
        return self.client.get_repository(owner, repo)

    def fetch_relevant_content(self, owner, repo, branch=None, progress_callback=None):
        """
        Recursively fetch relevant content from a repository.
        Focuses on documentation, examples, samples, and cookbook folders.
        """
        if not branch:
            try:
                repo_info = self.client.get_repository(owner, repo)
                branch = repo_info.get("default_branch", GITHUB_DEFAULT_BRANCH)
            except GitHubAPIError:
                branch = GITHUB_DEFAULT_BRANCH

        logger.info(f"Fetching relevant content from {owner}/{repo} (branch: {branch})")

        # Create repository cache directory
        repo_cache_dir = self.cache_dir / owner / repo
        repo_cache_dir.mkdir(parents=True, exist_ok=True)

        # Indicate progress started
        if progress_callback:
            progress_callback(15)

        # Start with root directory
        return self._fetch_directory_content(
            owner, repo, "", branch, repo_cache_dir, progress_callback
        )

    def _fetch_directory_content(
        self, owner, repo, path, branch, base_dir, progress_callback=None
    ):
        """Recursively fetch content from a directory with improved rate limiting."""
        try:
            contents = self.client.get_repository_contents(owner, repo, path, branch)
        except GitHubAPIError as e:
            logger.error(f"Error fetching directory {path}: {e}")
            return []

        if not isinstance(contents, list):
            logger.warning(f"Expected directory content but got a file: {path}")
            return []

        # Process this directory's contents
        files_data = []
        subdirs_to_process = []

        # Indicate progress for this directory
        if progress_callback and not path:  # Only for root directory
            progress_callback(20)

        for item in contents:
            item_name = item["name"]
            item_path = item["path"]
            item_type = item["type"]

            # Skip ignored directories
            if item_type == "dir" and item_name in IGNORED_DIRS:
                continue

            # Process directories
            if item_type == "dir":
                # Check if this is a relevant directory we want to process
                if self._is_relevant_folder(item_name) or self._is_relevant_folder(
                    path
                ):
                    subdirs_to_process.append((item_path, Path(base_dir) / item_name))
                # Otherwise, check if any parent directory is relevant
                elif path and any(
                    part for part in path.split("/") if self._is_relevant_folder(part)
                ):
                    subdirs_to_process.append((item_path, Path(base_dir) / item_name))

            # Process files (only in relevant directories)
            elif item_type == "file" and (
                self._is_relevant_folder(path)
                or any(
                    part for part in path.split("/") if self._is_relevant_folder(part)
                )
            ):
                if (
                    self._is_text_file(item_name)
                    and item["size"] / 1024 / 1024 <= MAX_FILE_SIZE_MB
                ):
                    files_data.append(
                        self._process_file(owner, repo, item, branch, base_dir)
                    )

        # Update progress again after processing this directory
        if progress_callback and not path:  # Only for root directory
            progress_callback(30)

        # Use ThreadPoolExecutor with fewer workers (3 instead of 10)
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for subdir_path, subdir_local in subdirs_to_process:
                subdir_local.mkdir(parents=True, exist_ok=True)
                futures.append(
                    executor.submit(
                        self._fetch_directory_content,
                        owner,
                        repo,
                        subdir_path,
                        branch,
                        subdir_local,
                        progress_callback,
                    )
                )

            # Collect results from all futures
            for future in futures:
                try:
                    result = future.result()
                    files_data.extend(result)
                except Exception as e:
                    logger.error(f"Error in directory fetch: {e}")
                    # Continue with other directories even if one fails

        # Final progress update when finished
        if progress_callback and not path:  # Only for root directory
            progress_callback(80)

        return files_data

    def _process_file(self, owner, repo, file_info, branch, base_dir):
        """Process a single file and save it to cache."""
        try:
            file_content = self.client.get_repository_file(
                owner, repo, file_info["path"], branch
            )
            file_path = Path(base_dir) / file_info["name"]

            # Save to cache
            file_path.write_text(file_content, encoding="utf-8", errors="replace")

            return {
                "name": file_info["name"],
                "path": file_info["path"],
                "sha": file_info["sha"],
                "size": file_info["size"],
                "url": file_info["html_url"],
                "local_path": str(file_path),
                "repo": f"{owner}/{repo}",
                "branch": branch,
            }
        except Exception as e:
            logger.error(f"Error processing file {file_info['path']}: {e}")
            # For large files, create a placeholder with file info but mark as error
            try:
                # Create an error file to indicate download failure
                error_file_path = Path(base_dir) / f"{file_info['name']}.error"
                error_file_path.write_text(
                    f"Error downloading: {str(e)}", encoding="utf-8"
                )
            except Exception:
                pass

            return {
                "name": file_info["name"],
                "path": file_info["path"],
                "error": str(e),
                "repo": f"{owner}/{repo}",
                "branch": branch,
                "size": file_info.get("size", 0),
            }

    def _is_relevant_folder(self, folder_name):
        """Check if a folder is relevant (documentation, examples, etc.)."""
        folder_lower = folder_name.lower()
        return any(relevant in folder_lower for relevant in RELEVANT_FOLDERS)

    def _is_text_file(self, filename):
        """Check if a file is a text file based on extension."""
        return any(filename.lower().endswith(ext) for ext in TEXT_FILE_EXTENSIONS)

    def _is_pdf_file(self, filename):
        """Check if a file is a PDF file based on extension."""
        return filename.lower().endswith(".pdf")

    def _process_pdf_folder_structure(self, base_dir):
        """Process directory structure to extract PDF labels from folder names."""
        pdf_data = []
        base_path = Path(base_dir)
        
        # Walk the directory structure
        for root, dirs, files in os.walk(base_path):
            # Skip ignored directories
            if any(ignored in root.split(os.sep) for ignored in IGNORED_DIRS):
                continue
                
            # Extract PDF files
            for file in files:
                if file.lower().endswith('.pdf'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(base_path)
                    
                    # Extract labels from directory structure
                    path_parts = rel_path.parent.parts
                    labels = [part for part in path_parts if self._is_relevant_folder(part)]
                    
                    pdf_data.append({
                        "file_path": str(file_path),
                        "relative_path": str(rel_path),
                        "labels": labels,
                        "filename": file,
                        "directory": str(rel_path.parent)
                    })
        
        return pdf_data
