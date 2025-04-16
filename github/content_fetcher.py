import logging
import re
import atexit
import signal
import time
from github.repository import RepositoryFetcher
from utils.performance import async_process
from utils.task_tracker import TaskTracker
from concurrent.futures import ThreadPoolExecutor
import requests
import threading

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
        self.task_tracker = TaskTracker()
        
        # Status display variables
        self.status_thread = None
        self.stop_status_display = threading.Event()
        self.current_status = ""

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

    def _start_status_display(self, task_id=None):
        """
        Start a background thread to display download status in the console.
        
        Args:
            task_id (str, optional): Task ID for tracking
        """
        self.stop_status_display.clear()
        self.status_thread = threading.Thread(
            target=self._status_display_thread,
            args=(task_id,),
            daemon=True
        )
        self.status_thread.start()
        
    def _status_display_thread(self, task_id=None):
        """
        Background thread that periodically updates the console with download status.
        
        Args:
            task_id (str, optional): Task ID for tracking
        """
        try:
            while not self.stop_status_display.is_set():
                if self.repo_fetcher and hasattr(self.repo_fetcher, 'download_queue'):
                    # Get current status from the download queue
                    status_message = self.repo_fetcher.download_queue.get_status_message()
                    
                    if status_message != self.current_status:
                        # Only print status when it changes to reduce console spam
                        self.current_status = status_message
                        
                        # Clear the line and print the updated status
                        print(f"\r{' ' * 100}", end="\r")  # Clear the line
                        print(f"\r{status_message}", end="", flush=True)
                        
                        # Update task tracker if we have a task ID
                        if task_id:
                            progress = self.repo_fetcher.download_queue.get_progress()
                            self.task_tracker.update_task_progress(
                                task_id,
                                progress["percent"],
                                stage="downloading",
                                stage_progress=progress["percent"]
                            )
                
                # Sleep briefly before next update
                time.sleep(0.5)
                
        except Exception as e:
            logger.error(f"Error in status display thread: {e}")
        finally:
            # Clear the line before exiting
            print(f"\r{' ' * 100}", end="\r")
            
    def _stop_status_display(self):
        """Stop the status display thread."""
        if self.status_thread and self.status_thread.is_alive():
            self.stop_status_display.set()
            self.status_thread.join(timeout=1.0)
            
            # Print a newline to ensure next output starts on a clean line
            print()

    def fetch_content_for_dataset(self, repo_data, branch=None, progress_callback=None, _cancellation_event=None):
        """
        Fetch content suitable for dataset creation.
        
        Args:
            repo_data: Repository data (URL string or repository dict)
            branch: Branch to fetch from (optional)
            progress_callback: Function to call with progress updates
            _cancellation_event: Event that can be set to cancel the operation
            
        Returns:
            List of content files
        """
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
        task_id = None
        
        try:
            # Create task for tracking
            task_id = self.task_tracker.create_task(
                "repository_fetch",
                {"owner": owner, "repo": repo, "branch": branch},
                f"Fetching content from {owner}/{repo}"
            )
            
            # Start status display
            self._start_status_display(task_id)
            
            # Check for early cancellation
            if _cancellation_event and _cancellation_event.is_set():
                logger.info(f"Operation cancelled before starting fetch for {owner}/{repo}")
                self.task_tracker.cancel_task(task_id)
                return []
            
            # Initialize progress at 10% to show activity
            if progress_callback:
                progress_callback(10)

            content_files = self.repo_fetcher.fetch_relevant_content(
                owner, repo, branch, progress_callback=progress_callback
            )
            
            # Check for cancellation after content fetch
            if _cancellation_event and _cancellation_event.is_set():
                logger.info(f"Operation cancelled after content fetch for {owner}/{repo}")
                self.task_tracker.cancel_task(task_id)
                return []

            if progress_callback:
                progress_callback(90)  # Almost done

            logger.info(
                f"Fetched {len(content_files)} relevant files from {owner}/{repo}"
            )

            # Complete progress
            if progress_callback:
                progress_callback(100)
                
            # Complete the task
            self.task_tracker.complete_task(
                task_id, 
                success=True,
                result={"files_count": len(content_files)}
            )

            return content_files
        except Exception as e:
            logger.error(f"Failed to fetch content for {owner}/{repo}: {e}")
            
            # Update task if we have one
            if task_id:
                self.task_tracker.complete_task(
                    task_id,
                    success=False,
                    result={"error": str(e)}
                )
                
            # Make sure we indicate an error through the progress callback
            if progress_callback:
                progress_callback(-1)  # Use negative value to indicate error
            raise
        finally:
            # Always stop the status display
            self._stop_status_display()

    def fetch_multiple_repositories(self, org_name, progress_callback=None, _cancellation_event=None):
        """
        Fetch content from multiple repositories in an organization.
        
        Args:
            org_name: Name of the organization
            progress_callback: Function to call with progress updates
            _cancellation_event: Event that can be set to cancel the operation
            
        Returns:
            List of content files
        """
        task_id = None
        
        try:
            # Create task for tracking
            task_id = self.task_tracker.create_task(
                "organization_fetch",
                {"org": org_name},
                f"Fetching content from organization {org_name}"
            )
            
            # Progress sections:
            # 0-20%: Fetch repositories list and scan folder structures
            # 20-90%: Download and process files
            # 90-100%: Final processing

            # Start status display
            self._start_status_display(task_id)
            
            # Update task status
            self.task_tracker.update_task_progress(
                task_id, 
                5, 
                stage="scanning_repositories",
                stage_progress=5
            )
            
            # Check for cancellation
            if _cancellation_event and _cancellation_event.is_set():
                self.task_tracker.cancel_task(task_id)
                return []

            # Phase 1: Fetch all repositories in the organization
            repos = self.fetch_org_repositories(org_name, progress_callback)
            
            # Check for cancellation after repository fetch
            if _cancellation_event and _cancellation_event.is_set():
                logger.info("Operation cancelled during repository fetch")
                self.task_tracker.cancel_task(task_id)
                return []

            if not repos:
                logger.warning(f"No repositories found for organization {org_name}")
                if progress_callback:
                    progress_callback(70)  # Skip to the end of this stage
                self.task_tracker.complete_task(
                    task_id,
                    success=True,
                    result={"files_count": 0, "message": "No repositories found"}
                )
                return []

            logger.info(f"Found {len(repos)} repositories in {org_name}")
            logger.debug(f"Repository names: {[repo['name'] for repo in repos[:5]]}...")

            if progress_callback:
                logger.debug("Updating progress to 10% after finding repositories")
                progress_callback(10)
                
            self.task_tracker.update_task_progress(
                task_id,
                10,
                stage="scanning_repositories",
                stage_progress=50
            )

            # Phase 2: Scan all repositories to identify relevant files first
            scan_results = []
            
            try:
                # Function to scan a single repository
                def scan_repository_structure(repo):
                    try:
                        owner = repo["owner"]["login"]
                        repo_name = repo["name"]
                        branch = repo.get("default_branch")
                        
                        logger.debug(f"Scanning repository structure: {owner}/{repo_name}")
                        
                        # Scan repository structure without downloading files
                        scan_result = self.github_client.scan_repository_structure(
                            owner, repo_name, branch
                        )
                        
                        return {
                            "owner": owner,
                            "repo": repo_name,
                            "branch": branch,
                            "scan_result": scan_result
                        }
                    except Exception as e:
                        logger.error(f"Error scanning repository {repo['name']}: {e}")
                        return None
                
                # Process repositories in batches to avoid overwhelming the API
                batch_size = min(5, len(repos))
                for i in range(0, len(repos), batch_size):
                    # Check for cancellation before processing each batch
                    if _cancellation_event and _cancellation_event.is_set():
                        logger.info("Operation cancelled during repository scanning")
                        self.task_tracker.cancel_task(task_id)
                        return []
                        
                    batch = repos[i:i+batch_size]
                    logger.debug(f"Scanning batch {i//batch_size + 1}: {[r['name'] for r in batch]}")
                    
                    # Use direct threading for better control
                    executor = get_executor()
                    futures = [executor.submit(scan_repository_structure, repo) for repo in batch]
                    
                    # Collect results
                    for future in futures:
                        try:
                            # Check for cancellation during future processing
                            if _cancellation_event and _cancellation_event.is_set():
                                executor.shutdown(wait=False)
                                logger.info("Operation cancelled during repository scanning futures")
                                self.task_tracker.cancel_task(task_id)
                                return []
                                
                            result = future.result(timeout=300)  # 5-minute timeout
                            if result:
                                scan_results.append(result)
                        except Exception as e:
                            logger.error(f"Error in scan batch processing: {e}")
                    
                    # Update progress (10-20%)
                    if progress_callback:
                        scan_progress = 10 + 10 * min((i + batch_size) / len(repos), 1.0)
                        progress_callback(scan_progress)
                        
                    # Update task status
                    self.task_tracker.update_task_progress(
                        task_id,
                        scan_progress,
                        stage="scanning_repositories",
                        stage_progress=min(100, (i + batch_size) / len(repos) * 100)
                    )
                    
                # Log overall scan results
                total_relevant_files = sum(
                    r["scan_result"]["relevant_files"] for r in scan_results if r and "scan_result" in r
                )
                logger.info(
                    f"Completed scanning {len(scan_results)} repositories. "
                    f"Found {total_relevant_files} relevant files."
                )
                
                # Update task status for download phase
                self.task_tracker.update_task_progress(
                    task_id,
                    20,
                    stage="downloading_files",
                    stage_progress=0
                )
                
                # Phase 3: Build a global download queue from scan results
                download_queue = self.repo_fetcher.download_queue
                download_queue.reset()
                
                # Identify all files to download
                all_files_to_download = []
                
                for scan_result in scan_results:
                    if not scan_result or "scan_result" not in scan_result:
                        continue
                        
                    owner = scan_result["owner"]
                    repo_name = scan_result["repo"]
                    branch = scan_result["branch"]
                    structure = scan_result["scan_result"]
                    
                    # Create repository cache directory
                    repo_cache_dir = self.repo_fetcher.cache_dir / owner / repo_name
                    repo_cache_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Add files from all relevant paths in this repository
                    for path in structure.get("relevant_paths", []):
                        files = self.repo_fetcher._identify_files_to_download(
                            structure, path, owner, repo_name, branch, repo_cache_dir
                        )
                        all_files_to_download.extend(files)
                
                # Add all files to the global download queue
                if all_files_to_download:
                    download_queue.add_files(all_files_to_download)
                    logger.info(f"Added {len(all_files_to_download)} files to download queue")
                else:
                    logger.warning("No files identified for download")
                    
                # Phase 4: Download all queued files
                if download_queue.total_files > 0:
                    all_content = []
                    
                    # Process files in small batches for better progress display
                    batch_size = 5
                    while not download_queue.is_empty():
                        # Check for cancellation before each batch
                        if _cancellation_event and _cancellation_event.is_set():
                            logger.info("Operation cancelled during file download")
                            self.task_tracker.cancel_task(task_id)
                            return []
                            
                        batch = []
                        for _ in range(min(batch_size, len(download_queue.queue))):
                            file_item = download_queue.get_next_file()
                            if file_item:
                                batch.append(file_item)
                                
                        # Download this batch
                        with ThreadPoolExecutor(max_workers=3) as executor:
                            futures = []
                            for file_item in batch:
                                futures.append(executor.submit(
                                    self.repo_fetcher._download_single_file,
                                    file_item["owner"],
                                    file_item["repo"],
                                    file_item["path"],
                                    file_item["branch"],
                                    file_item["local_path"]
                                ))
                                
                            # Process results
                            for future in futures:
                                # Check for cancellation during future processing
                                if _cancellation_event and _cancellation_event.is_set():
                                    executor.shutdown(wait=False)
                                    logger.info("Operation cancelled during file download futures")
                                    self.task_tracker.cancel_task(task_id)
                                    return []
                                    
                                try:
                                    result = future.result()
                                    if result:
                                        all_content.append(result)
                                    download_queue.mark_processed()
                                except Exception as e:
                                    logger.error(f"Error downloading file: {e}")
                                    download_queue.mark_processed()
                        
                        # Update progress (20-90%)
                        progress_info = download_queue.get_progress()
                        if progress_callback:
                            download_progress = 20 + (progress_info["percent"] * 0.7)
                            progress_callback(min(90, download_progress))
                            
                        # Update task status
                        self.task_tracker.update_task_progress(
                            task_id,
                            min(90, download_progress),
                            stage="downloading_files",
                            stage_progress=progress_info["percent"]
                        )
                        
                    # Complete progress
                    if progress_callback:
                        progress_callback(90)
                        
                    logger.info(f"Downloaded {len(all_content)} files from {len(scan_results)} repositories")
                    
                    # Update task status for completion
                    self.task_tracker.update_task_progress(
                        task_id,
                        100,
                        stage="complete",
                        stage_progress=100,
                        status="completed"
                    )
                    
                    return all_content
                else:
                    # No files to download
                    logger.warning("No files to download in any repositories")
                    if progress_callback:
                        progress_callback(90)
                        
                    # Complete task
                    self.task_tracker.complete_task(
                        task_id,
                        success=True,
                        result={"files_count": 0, "message": "No relevant files found"}
                    )
                    
                    return []
                    
            except RuntimeError as e:
                if "cannot schedule new futures" in str(e):
                    logger.warning("Interpreter is shutting down. Stopping processing early.")
                    self.task_tracker.cancel_task(task_id)
                else:
                    raise
                    
        except Exception as e:
            logger.error(
                f"Failed to fetch multiple repositories for {org_name}: {e}",
                exc_info=True,
            )
            
            # Complete task as failed
            if task_id:
                self.task_tracker.complete_task(
                    task_id,
                    success=False,
                    result={"error": str(e)}
                )
                
            raise
            
        finally:
            # Always stop the status display
            self._stop_status_display()
