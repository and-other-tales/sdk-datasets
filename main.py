import sys
import os
import logging
import argparse
import signal
import traceback
from pathlib import Path
from utils.logging_config import setup_logging
from config.credentials_manager import CredentialsManager
from huggingface.dataset_manager import DatasetManager
from utils.task_tracker import TaskTracker
from utils.task_scheduler import TaskScheduler
from api.server import start_server, stop_server, is_server_running, get_server_info
from threading import Event, current_thread

# Global cancellation event for stopping ongoing tasks
global_cancellation_event = Event()

# Global logger
logger = logging.getLogger(__name__)


def run_cli():
    """Run the command-line interface."""
    print("\n===== othertales SDK Dataset Generator =====")
    print("CLI mode\n")
    print("Press Ctrl+C at any time to safely exit the application")
    
    # Initialize managers and clients
    credentials_manager = CredentialsManager()
    dataset_manager = DatasetManager(credentials_manager=credentials_manager)
    task_tracker = TaskTracker()
    github_client = None
    dataset_creator = None
    
    print("Initialization successful")
    
    # Reset cancellation event at the start
    global_cancellation_event.clear()
    
    while not global_cancellation_event.is_set() and not getattr(current_thread(), 'exit_requested', False):
        # Show dynamic menu based on server status and available resumable tasks
        server_running = is_server_running()
        resumable_tasks = task_tracker.list_resumable_tasks()
        
        print("\nMain Menu:")
        if server_running:
            print("1. Stop OpenAPI Endpoints")
        else:
            print("1. Start OpenAPI Endpoints")
        print("2. Generate Dataset")
        print("3. Manage Existing Datasets")
        
        # Only show Resume Dataset Creation if there are resumable tasks
        if resumable_tasks:
            print("4. Resume Dataset Creation")
            print("5. Scheduled Tasks & Automation")
            print("6. Configuration")
            print("7. Exit")
            max_choice = 7
        else:
            print("4. Scheduled Tasks & Automation")
            print("5. Configuration")
            print("6. Exit")
            max_choice = 6
        
        choice = input(f"\nEnter your choice (1-{max_choice}): ")
        
        if choice == "1":
            # Handle OpenAPI server
            if server_running:
                print("\n----- Stopping OpenAPI Endpoints -----")
                if stop_server():
                    print("OpenAPI Endpoints stopped successfully")
                else:
                    print("Failed to stop OpenAPI Endpoints")
            else:
                print("\n----- Starting OpenAPI Endpoints -----")
                # Get OpenAPI key
                api_key = credentials_manager.get_openapi_key()
                
                if not api_key:
                    print("OpenAPI key not configured. Please set an API key.")
                    api_key = input("Enter new OpenAPI key: ")
                    if credentials_manager.save_openapi_key(api_key):
                        print("OpenAPI key saved successfully")
                    else:
                        print("Failed to save OpenAPI key")
                        continue
                
                # Get configured server port
                server_port = credentials_manager.get_server_port()
                
                if start_server(api_key, port=server_port):
                    print("OpenAPI Endpoints started successfully")
                    print(f"Server running at: http://0.0.0.0:{server_port}")
                    print(f"API Documentation: http://0.0.0.0:{server_port}/docs")
                    print(f"OpenAPI Schema: http://0.0.0.0:{server_port}/openapi.json")
                else:
                    print("Failed to start OpenAPI Endpoints")
        
        elif choice == "2":
            print("\n----- Generate Dataset -----")
            
            # Source type
            print("\nSource Type:")
            print("1. Organization")
            print("2. Repository")
            source_type = input("Enter choice (1-2): ")
            
            if source_type == "1":
                org_name = input("Enter GitHub organization name: ")
                dataset_name = input("Enter dataset name: ")
                description = input("Enter dataset description: ")
                
                try:
                    from github.content_fetcher import ContentFetcher
                    from huggingface.dataset_creator import DatasetCreator
                    
                    # Initialize clients
                    if github_client is None:
                        github_username, github_token = credentials_manager.get_github_credentials()
                        if not github_token:
                            print("\nError: GitHub token not found. Please set your credentials first.")
                            continue
                        content_fetcher = ContentFetcher(github_token=github_token)
                    
                    if dataset_creator is None:
                        hf_username, huggingface_token = credentials_manager.get_huggingface_credentials()
                        if not huggingface_token:
                            print("\nError: Hugging Face token not found. Please set your credentials first.")
                            continue
                        dataset_creator = DatasetCreator(huggingface_token=huggingface_token)
                    
                    print(f"\nFetching repositories from organization: {org_name}")
                    
                    # Use the more reliable fetch_org_repositories method instead
                    def progress_callback(percent, message=None):
                        if message:
                            print(f"Progress: {percent:.0f}% - {message}")
                        else:
                            print(f"Progress: {percent:.0f}%")
                            
                    repos = content_fetcher.fetch_org_repositories(org_name, 
                                                                  progress_callback=lambda p: progress_callback(p))
                    
                    if not repos:
                        print(f"No repositories found for organization: {org_name}")
                        continue
                        
                    print(f"Found {len(repos)} repositories")
                    print("Fetching repository content...")
                    
                    # Use proper progress callback for content fetching
                    content = content_fetcher.fetch_multiple_repositories(org_name, 
                                                                         progress_callback=lambda p: progress_callback(p))
                    
                    if not content:
                        print("No content found in repositories")
                        continue
                        
                    print(f"Processing {len(content)} files...")
                    
                    # Create dataset
                    success, dataset = dataset_creator.create_and_push_dataset(
                        file_data_list=content,
                        dataset_name=dataset_name,
                        description=description,
                        source_info=org_name
                    )
                    
                    if success:
                        print(f"\nDataset '{dataset_name}' created successfully")
                    else:
                        print("\nFailed to create dataset")
                        
                except Exception as e:
                    print(f"\nError creating dataset: {e}")
                
            elif source_type == "2":
                repo_url = input("Enter GitHub repository URL: ")
                dataset_name = input("Enter dataset name: ")
                description = input("Enter dataset description: ")
                
                try:
                    from github.content_fetcher import ContentFetcher
                    from huggingface.dataset_creator import DatasetCreator
                    
                    # Initialize clients
                    if github_client is None:
                        github_token, _ = credentials_manager.get_github_credentials()
                        if not github_token:
                            print("\nError: GitHub token not found. Please set your credentials first.")
                            continue
                        content_fetcher = ContentFetcher(github_token=github_token)
                    
                    if dataset_creator is None:
                        _, huggingface_token = credentials_manager.get_huggingface_credentials()
                        if not huggingface_token:
                            print("\nError: Hugging Face token not found. Please set your credentials first.")
                            continue
                        dataset_creator = DatasetCreator(huggingface_token=huggingface_token)
                    
                    print(f"\nCreating dataset from repository: {repo_url}")
                    
                    # Display progress callback function
                    def progress_callback(percent, message=None):
                        if percent % 10 == 0 or percent == 100:
                            status = f"Progress: {percent:.0f}%"
                            if message:
                                status += f" - {message}"
                            print(status)
                    
                    result = dataset_creator.create_dataset_from_repository(
                        repo_url=repo_url,
                        dataset_name=dataset_name,
                        description=description,
                        progress_callback=progress_callback
                    )
                    
                    if result.get("success"):
                        print(f"\nDataset '{dataset_name}' created successfully")
                    else:
                        print(f"\nFailed to create dataset: {result.get('message', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"\nError creating dataset: {e}")
                
            else:
                print("Invalid choice")
                
        elif choice == "3":
            print("\n----- Manage Datasets -----")
            
            try:
                _, huggingface_token = credentials_manager.get_huggingface_credentials()
                
                if not huggingface_token:
                    print("\nError: Hugging Face token not found. Please set your credentials first.")
                    continue
                
                # Initialize dataset manager if needed
                if dataset_manager is None:
                    dataset_manager = DatasetManager(huggingface_token=huggingface_token,
                                                   credentials_manager=credentials_manager)
                
                print("\nFetching your datasets from Hugging Face...")
                datasets = dataset_manager.list_datasets()
                
                if not datasets:
                    print("No datasets found for your account.")
                    continue
                
                # Display datasets and options
                print(f"\nFound {len(datasets)} datasets:")
                for i, dataset in enumerate(datasets):
                    print(f"{i+1}. {dataset.get('id', 'Unknown')} - {dataset.get('lastModified', 'Unknown date')}")
                
                print("\nOptions:")
                print("1. View dataset details")
                print("2. Download dataset metadata")
                print("3. Delete a dataset")
                print("4. Return to main menu")
                
                manage_choice = input("\nEnter choice (1-4): ")
                
                if manage_choice == "1":
                    dataset_index = int(input("Enter dataset number to view: ")) - 1
                    
                    if 0 <= dataset_index < len(datasets):
                        dataset_id = datasets[dataset_index].get('id')
                        info = dataset_manager.get_dataset_info(dataset_id)
                        
                        if info:
                            print(f"\n----- Dataset: {info.id} -----")
                            print(f"Description: {info.description}")
                            print(f"Created: {info.created_at}")
                            print(f"Last modified: {info.last_modified}")
                            print(f"Downloads: {info.downloads}")
                            print(f"Likes: {info.likes}")
                            print(f"Tags: {', '.join(info.tags) if info.tags else 'None'}")
                        else:
                            print(f"Error retrieving details for dataset {dataset_id}")
                    else:
                        print("Invalid dataset number")
                
                elif manage_choice == "2":
                    dataset_index = int(input("Enter dataset number to download metadata: ")) - 1
                    
                    if 0 <= dataset_index < len(datasets):
                        dataset_id = datasets[dataset_index].get('id')
                        success = dataset_manager.download_dataset_metadata(dataset_id)
                        
                        if success:
                            print(f"\nMetadata for dataset '{dataset_id}' downloaded successfully")
                            print(f"Saved to ./dataset_metadata/{dataset_id}/")
                        else:
                            print(f"Error downloading metadata for dataset {dataset_id}")
                    else:
                        print("Invalid dataset number")
                
                elif manage_choice == "3":
                    dataset_index = int(input("Enter dataset number to delete: ")) - 1
                    
                    if 0 <= dataset_index < len(datasets):
                        dataset_id = datasets[dataset_index].get('id')
                        
                        confirm = input(f"Are you sure you want to delete dataset '{dataset_id}'? (yes/no): ")
                        if confirm.lower() == "yes":
                            success = dataset_manager.delete_dataset(dataset_id)
                            
                            if success:
                                print(f"\nDataset '{dataset_id}' deleted successfully")
                            else:
                                print(f"Error deleting dataset {dataset_id}")
                        else:
                            print("Deletion cancelled")
                    else:
                        print("Invalid dataset number")
                
                elif manage_choice == "4":
                    continue
                
                else:
                    print("Invalid choice")
                
            except Exception as e:
                print(f"\nError managing datasets: {e}")
                logging.error(f"Error in manage datasets: {e}")
                
        # Resume Dataset Creation (only available if there are resumable tasks)
        elif choice == "4" and resumable_tasks:
            print("\n----- Resume Dataset Creation -----")
            
            try:
                # Display resumable tasks
                print("\nAvailable tasks to resume:")
                for i, task in enumerate(resumable_tasks):
                    # Format task description nicely
                    task_desc = task.get("description", "Unknown task")
                    progress = task.get("progress", 0)
                    updated = task.get("updated_ago", "unknown time")
                    
                    print(f"{i+1}. {task_desc} ({progress:.0f}% complete, updated {updated})")
                
                # Get task selection
                task_index = int(input("\nEnter task number to resume (0 to cancel): ")) - 1
                
                if task_index < 0:
                    print("Resumption cancelled")
                    continue
                    
                if 0 <= task_index < len(resumable_tasks):
                    selected_task = resumable_tasks[task_index]
                    task_id = selected_task["id"]
                    task_type = selected_task["type"]
                    task_params = selected_task["params"]
                    
                    # Confirm resumption
                    confirm = input(f"Resume task: {selected_task['description']}? (yes/no): ")
                    if confirm.lower() != "yes":
                        print("Resumption cancelled")
                        continue
                    
                    print(f"\nResuming task {task_id}...")
                    
                    # Create cancellation event
                    cancellation_event = Event()
                    
                    # Handle different task types
                    if task_type == "repository":
                        # Initialize required components
                        from github.content_fetcher import ContentFetcher
                        from huggingface.dataset_creator import DatasetCreator
                        
                        # Initialize clients if needed
                        github_username, github_token = credentials_manager.get_github_credentials()
                        if not github_token:
                            print("\nError: GitHub token not found. Please set your credentials first.")
                            continue
                            
                        hf_username, huggingface_token = credentials_manager.get_huggingface_credentials()
                        if not huggingface_token:
                            print("\nError: Hugging Face token not found. Please set your credentials first.")
                            continue
                            
                        content_fetcher = ContentFetcher(github_token=github_token)
                        dataset_creator = DatasetCreator(huggingface_token=huggingface_token)
                        
                        # Progress callback function
                        def progress_callback(percent, message=None):
                            if percent % 10 == 0 or percent == 100:
                                status = f"Progress: {percent:.0f}%"
                                if message:
                                    status += f" - {message}"
                                print(status)
                        
                        # Resume repository task
                        repo_url = task_params.get("repo_url")
                        dataset_name = task_params.get("dataset_name")
                        description = task_params.get("description")
                        
                        print(f"Resuming dataset creation from repository: {repo_url}")
                        
                        result = dataset_creator.create_dataset_from_repository(
                            repo_url=repo_url,
                            dataset_name=dataset_name,
                            description=description,
                            progress_callback=progress_callback,
                            _cancellation_event=cancellation_event,
                            task_id=task_id,
                            resume_from=selected_task.get("current_stage")
                        )
                        
                        if result.get("success"):
                            print(f"\nDataset '{dataset_name}' creation resumed and completed successfully")
                        else:
                            print(f"\nFailed to resume dataset creation: {result.get('message', 'Unknown error')}")
                    
                    # Handle other task types when implemented
                    else:
                        print(f"Unsupported task type: {task_type}")
                        
                else:
                    print("Invalid task number")
                
            except Exception as e:
                print(f"\nError resuming task: {e}")
                logging.error(f"Error resuming task: {e}")
        
        # Scheduled Tasks menu (position depends on whether Resume Dataset Creation is available)
        elif (choice == "4" and not resumable_tasks) or (choice == "5" and resumable_tasks):
            print("\n----- Scheduled Tasks & Automation -----")
            
            try:
                # Initialize task scheduler
                task_scheduler = TaskScheduler()
                
                # Check if crontab is available
                if not task_scheduler.is_crontab_available():
                    print("Error: Crontab is not available on this system.")
                    print("Scheduled tasks require crontab to be installed and accessible.")
                    continue
                
                # Show scheduled tasks submenu
                print("\nScheduled Tasks Options:")
                print("1. List Scheduled Tasks")
                print("2. Add New Scheduled Task")
                print("3. Edit Scheduled Task")
                print("4. Delete Scheduled Task")
                print("5. Run Scheduled Task Now")
                print("6. Return to Main Menu")
                
                sched_choice = input("\nEnter choice (1-6): ")
                
                if sched_choice == "1":
                    # List scheduled tasks
                    tasks = task_scheduler.list_scheduled_tasks()
                    
                    if not tasks:
                        print("\nNo scheduled tasks found.")
                        continue
                    
                    print(f"\nFound {len(tasks)} scheduled tasks:")
                    for i, task in enumerate(tasks):
                        dataset = task.get("dataset_name", "Unknown")
                        schedule = task.get("schedule_description", "Unknown schedule")
                        next_run = task.get("next_run", "Unknown")
                        source = task.get("source_name", "Unknown")
                        source_type = task.get("source_type", "Unknown")
                        
                        print(f"{i+1}. {dataset} - {source_type}: {source}")
                        print(f"   Schedule: {schedule}")
                        print(f"   Next run: {next_run}")
                        print()
                
                elif sched_choice == "2":
                    # Add new scheduled task
                    print("\n--- Add New Scheduled Task ---")
                    
                    # Get source type
                    print("\nSource Type:")
                    print("1. Organization")
                    print("2. Repository")
                    source_type_choice = input("Enter choice (1-2): ")
                    
                    if source_type_choice == "1":
                        source_type = "organization"
                        source_name = input("Enter GitHub organization name: ")
                    elif source_type_choice == "2":
                        source_type = "repository"
                        source_name = input("Enter GitHub repository URL: ")
                    else:
                        print("Invalid choice")
                        continue
                    
                    # Get dataset name
                    dataset_name = input("Enter dataset name to update: ")
                    if not dataset_name:
                        print("Dataset name cannot be empty")
                        continue
                    
                    # Get schedule type
                    print("\nSchedule Type:")
                    print("1. Daily (midnight)")
                    print("2. Weekly (Sunday midnight)")
                    print("3. Bi-weekly (1st and 15th of month)")
                    print("4. Monthly (1st of month)")
                    print("5. Custom schedule")
                    schedule_choice = input("Enter choice (1-5): ")
                    
                    schedule_type = None
                    custom_params = {}
                    
                    if schedule_choice == "1":
                        schedule_type = "daily"
                    elif schedule_choice == "2":
                        schedule_type = "weekly"
                    elif schedule_choice == "3":
                        schedule_type = "biweekly"
                    elif schedule_choice == "4":
                        schedule_type = "monthly"
                    elif schedule_choice == "5":
                        schedule_type = "custom"
                        print("\nEnter custom schedule (cron format):")
                        custom_params["minute"] = input("Minute (0-59): ")
                        custom_params["hour"] = input("Hour (0-23): ")
                        custom_params["day"] = input("Day of month (1-31, * for all): ")
                        custom_params["month"] = input("Month (1-12, * for all): ")
                        custom_params["day_of_week"] = input("Day of week (0-6, 0=Sunday, * for all): ")
                    else:
                        print("Invalid choice")
                        continue
                    
                    # Create the scheduled task
                    task_id = task_scheduler.create_scheduled_task(
                        task_type="update",
                        source_type=source_type,
                        source_name=source_name,
                        dataset_name=dataset_name,
                        schedule_type=schedule_type,
                        **custom_params
                    )
                    
                    if task_id:
                        print(f"\nScheduled task created successfully (ID: {task_id})")
                    else:
                        print("\nFailed to create scheduled task")
                
                elif sched_choice == "3":
                    # Edit scheduled task
                    tasks = task_scheduler.list_scheduled_tasks()
                    
                    if not tasks:
                        print("\nNo scheduled tasks found.")
                        continue
                    
                    print(f"\nSelect a task to edit:")
                    for i, task in enumerate(tasks):
                        dataset = task.get("dataset_name", "Unknown")
                        schedule = task.get("schedule_description", "Unknown schedule")
                        source = task.get("source_name", "Unknown")
                        
                        print(f"{i+1}. {dataset} - {source} ({schedule})")
                    
                    task_index = int(input("\nEnter task number (0 to cancel): ")) - 1
                    
                    if task_index < 0:
                        print("Edit cancelled")
                        continue
                        
                    if 0 <= task_index < len(tasks):
                        selected_task = tasks[task_index]
                        task_id = selected_task["id"]
                        
                        # Get new schedule type
                        print("\nSelect new schedule type:")
                        print("1. Daily (midnight)")
                        print("2. Weekly (Sunday midnight)")
                        print("3. Bi-weekly (1st and 15th of month)")
                        print("4. Monthly (1st of month)")
                        print("5. Custom schedule")
                        schedule_choice = input("Enter choice (1-5): ")
                        
                        schedule_type = None
                        custom_params = {}
                        
                        if schedule_choice == "1":
                            schedule_type = "daily"
                        elif schedule_choice == "2":
                            schedule_type = "weekly"
                        elif schedule_choice == "3":
                            schedule_type = "biweekly"
                        elif schedule_choice == "4":
                            schedule_type = "monthly"
                        elif schedule_choice == "5":
                            schedule_type = "custom"
                            print("\nEnter custom schedule (cron format):")
                            custom_params["minute"] = input("Minute (0-59): ")
                            custom_params["hour"] = input("Hour (0-23): ")
                            custom_params["day"] = input("Day of month (1-31, * for all): ")
                            custom_params["month"] = input("Month (1-12, * for all): ")
                            custom_params["day_of_week"] = input("Day of week (0-6, 0=Sunday, * for all): ")
                        else:
                            print("Invalid choice")
                            continue
                        
                        # Update the scheduled task
                        if task_scheduler.update_scheduled_task(task_id, schedule_type, **custom_params):
                            print(f"\nScheduled task updated successfully")
                        else:
                            print("\nFailed to update scheduled task")
                    else:
                        print("Invalid task number")
                
                elif sched_choice == "4":
                    # Delete scheduled task
                    tasks = task_scheduler.list_scheduled_tasks()
                    
                    if not tasks:
                        print("\nNo scheduled tasks found.")
                        continue
                    
                    print(f"\nSelect a task to delete:")
                    for i, task in enumerate(tasks):
                        dataset = task.get("dataset_name", "Unknown")
                        schedule = task.get("schedule_description", "Unknown schedule")
                        source = task.get("source_name", "Unknown")
                        
                        print(f"{i+1}. {dataset} - {source} ({schedule})")
                    
                    task_index = int(input("\nEnter task number (0 to cancel): ")) - 1
                    
                    if task_index < 0:
                        print("Deletion cancelled")
                        continue
                        
                    if 0 <= task_index < len(tasks):
                        selected_task = tasks[task_index]
                        task_id = selected_task["id"]
                        
                        # Confirm deletion
                        confirm = input(f"Are you sure you want to delete this scheduled task? (yes/no): ")
                        if confirm.lower() != "yes":
                            print("Deletion cancelled")
                            continue
                        
                        # Delete the scheduled task
                        if task_scheduler.delete_scheduled_task(task_id):
                            print(f"\nScheduled task deleted successfully")
                        else:
                            print("\nFailed to delete scheduled task")
                    else:
                        print("Invalid task number")
                
                elif sched_choice == "5":
                    # Run scheduled task now
                    tasks = task_scheduler.list_scheduled_tasks()
                    
                    if not tasks:
                        print("\nNo scheduled tasks found.")
                        continue
                    
                    print(f"\nSelect a task to run now:")
                    for i, task in enumerate(tasks):
                        dataset = task.get("dataset_name", "Unknown")
                        source = task.get("source_name", "Unknown")
                        source_type = task.get("source_type", "Unknown")
                        
                        print(f"{i+1}. {dataset} - {source_type}: {source}")
                    
                    task_index = int(input("\nEnter task number (0 to cancel): ")) - 1
                    
                    if task_index < 0:
                        print("Run cancelled")
                        continue
                        
                    if 0 <= task_index < len(tasks):
                        selected_task = tasks[task_index]
                        task_id = selected_task["id"]
                        
                        print(f"\nRunning task in the background. Check logs for progress.")
                        if task_scheduler.run_task_now(task_id):
                            print(f"Task started successfully")
                        else:
                            print(f"Failed to start task")
                    else:
                        print("Invalid task number")
                
                elif sched_choice == "6":
                    continue
                
                else:
                    print("Invalid choice")
                
            except Exception as e:
                print(f"\nError managing scheduled tasks: {e}")
                logging.error(f"Error in scheduled tasks menu: {e}")
        
        # Configuration menu (position depends on whether Resume Dataset Creation is available)
        elif (choice == "5" and not resumable_tasks) or (choice == "6" and resumable_tasks):
            print("\n----- Configuration -----")
            print("1. API Credentials")
            print("2. Server & Dataset Configuration")
            print("3. Return to main menu")
            
            config_choice = input("\nEnter choice (1-3): ")
            
            if config_choice == "1":
                print("\n--- API Credentials ---")
                print("1. Set GitHub Credentials")
                print("2. Set Hugging Face Credentials")
                print("3. Set OpenAPI Key")
                print("4. Return to previous menu")
                
                cred_choice = input("\nEnter choice (1-4): ")
                
                if cred_choice == "1":
                    github_username = input("Enter GitHub username: ")
                    github_token = input("Enter GitHub token (will not be shown): ")
                    
                    try:
                        credentials_manager.save_github_credentials(github_username, github_token)
                        print("GitHub credentials saved successfully")
                        
                        # Ask if user wants to edit HuggingFace credentials too
                        edit_hf = input("\nWould you like to enter or amend your Hugging Face credentials too? (Y/n): ")
                        if edit_hf.lower() != "n":
                            hf_username = input("Enter Hugging Face username: ")
                            hf_token = input("Enter Hugging Face token (will not be shown): ")
                            
                            try:
                                credentials_manager.save_huggingface_credentials(hf_username, hf_token)
                                print("Hugging Face credentials saved successfully")
                            except Exception as e:
                                print(f"Error saving Hugging Face credentials: {e}")
                                
                    except Exception as e:
                        print(f"Error saving GitHub credentials: {e}")
                        
                elif cred_choice == "2":
                    hf_username = input("Enter Hugging Face username: ")
                    hf_token = input("Enter Hugging Face token (will not be shown): ")
                    
                    try:
                        credentials_manager.save_huggingface_credentials(hf_username, hf_token)
                        print("Hugging Face credentials saved successfully")
                        
                        # Ask if user wants to edit GitHub credentials too
                        edit_github = input("\nWould you like to enter or amend your GitHub credentials too? (Y/n): ")
                        if edit_github.lower() != "n":
                            github_username = input("Enter GitHub username: ")
                            github_token = input("Enter GitHub token (will not be shown): ")
                            
                            try:
                                credentials_manager.save_github_credentials(github_username, github_token)
                                print("GitHub credentials saved successfully")
                            except Exception as e:
                                print(f"Error saving GitHub credentials: {e}")
                                
                    except Exception as e:
                        print(f"Error saving Hugging Face credentials: {e}")
                
                elif cred_choice == "3":
                    openapi_key = input("Enter OpenAPI key (will not be shown): ")
                    
                    try:
                        credentials_manager.save_openapi_key(openapi_key)
                        print("OpenAPI key saved successfully")
                    except Exception as e:
                        print(f"Error saving OpenAPI key: {e}")
                        
                elif cred_choice == "4":
                    continue
                    
                else:
                    print("Invalid choice")
                    
                # Return to configuration menu
                continue
                
            elif config_choice == "2":
                print("\n--- Server & Dataset Configuration ---")
                
                # Show current settings
                server_port = credentials_manager.get_server_port()
                temp_dir = credentials_manager.get_temp_dir()
                cache_size = task_tracker.get_cache_size()
                
                print(f"1. Set API Server Port (current: {server_port})")
                print(f"2. Set Temporary Storage Location (current: {temp_dir})")
                print(f"3. Delete Cache & Temporary Files ({cache_size} MB)")
                print("4. Return to previous menu")
                
                server_choice = input("\nEnter choice (1-4): ")
                
                if server_choice == "1":
                    try:
                        new_port = int(input("Enter new server port (1024-65535): "))
                        if 1024 <= new_port <= 65535:
                            if credentials_manager.save_server_port(new_port):
                                print(f"Server port updated to {new_port}")
                            else:
                                print("Failed to update server port")
                        else:
                            print("Invalid port number. Must be between 1024 and 65535.")
                    except ValueError:
                        print("Invalid input. Port must be a number.")
                
                elif server_choice == "2":
                    new_dir = input("Enter new temporary storage location: ")
                    try:
                        path = Path(new_dir)
                        if credentials_manager.save_temp_dir(str(path.absolute())):
                            print(f"Temporary storage location updated to {path.absolute()}")
                        else:
                            print("Failed to update temporary storage location")
                    except Exception as e:
                        print(f"Error updating temporary storage location: {e}")
                
                elif server_choice == "3":
                    confirm = input(f"Are you sure you want to delete all cache and temporary files ({cache_size} MB)? (Y/N): ")
                    if confirm.lower() == "y":
                        if task_tracker.clear_cache():
                            print("Cache and temporary files deleted successfully")
                        else:
                            print("Failed to delete cache and temporary files")
                    else:
                        print("Cache deletion cancelled")
                
                elif server_choice == "4":
                    continue
                
                else:
                    print("Invalid choice")
            
            elif config_choice == "3":
                continue
                
            else:
                print("Invalid choice")
                
        # Exit application (position depends on whether Resume Dataset Creation is available)
        elif (choice == "6" and not resumable_tasks) or (choice == "7" and resumable_tasks):
            # Check if the server is running before exiting
            if is_server_running():
                print("\nStopping OpenAPI Endpoints before exiting...")
                stop_server()
            print("\nExiting application. Goodbye!")
            break
            
        else:
            # Dynamic message based on max_choice
            print(f"Invalid choice. Please enter a number between 1 and {max_choice}.")


def run_update(args):
    """
    Run an automatic update task based on command line arguments.
    Used for scheduled updates.
    
    Args:
        args: Command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    logger = logging.getLogger("update")
    logger.info(f"Starting automatic update with args: {args}")
    
    # Reset cancellation event at the start
    global_cancellation_event.clear()
    
    # Create a local cancellation event that links to the global one
    cancellation_event = Event()
    
    # Function to check for cancellation
    def check_cancelled():
        if global_cancellation_event.is_set():
            cancellation_event.set()
            return True
        return False
    
    try:
        # Initialize required components
        credentials_manager = CredentialsManager()
        task_tracker = TaskTracker()
        
        # Create task to track progress
        task_id = args.task_id if args.task_id else None
        
        # Check for GitHub credentials
        github_username, github_token = credentials_manager.get_github_credentials()
        if not github_token:
            logger.error("GitHub token not found. Please set credentials first.")
            return 1
            
        # Check for Hugging Face credentials
        hf_username, huggingface_token = credentials_manager.get_huggingface_credentials()
        if not huggingface_token:
            logger.error("Hugging Face token not found. Please set credentials first.")
            return 1
            
        # Initialize content fetcher and dataset creator
        from github.content_fetcher import ContentFetcher
        from huggingface.dataset_creator import DatasetCreator
        
        content_fetcher = ContentFetcher(github_token=github_token)
        dataset_creator = DatasetCreator(huggingface_token=huggingface_token)
        
        # Handle organization update
        if args.organization:
            org_name = args.organization
            dataset_name = args.dataset_name
            
            logger.info(f"Updating dataset '{dataset_name}' from organization: {org_name}")
            
            # Create task for tracking
            if not task_id:
                task_id = task_tracker.create_task(
                    "organization_update",
                    {"org": org_name, "dataset_name": dataset_name},
                    f"Updating dataset '{dataset_name}' from organization {org_name}"
                )
                
            # Define progress callback
            def progress_callback(percent, message=None):
                # Check for cancellation
                if check_cancelled():
                    if message:
                        logger.info(f"Cancelled at {percent:.0f}% - {message}")
                    else:
                        logger.info(f"Cancelled at {percent:.0f}%")
                    return
                
                if message:
                    logger.info(f"Progress: {percent:.0f}% - {message}")
                else:
                    logger.info(f"Progress: {percent:.0f}%")
                    
                if task_id:
                    task_tracker.update_task_progress(task_id, percent)
            
            # Fetch repositories
            repos = content_fetcher.fetch_org_repositories(org_name, progress_callback=progress_callback)
            
            # Check for cancellation
            if check_cancelled():
                logger.info("Operation cancelled by user")
                if task_id:
                    task_tracker.cancel_task(task_id)
                return 1
            
            if not repos:
                logger.warning(f"No repositories found for organization: {org_name}")
                if task_id:
                    task_tracker.complete_task(task_id, success=False, 
                                              result={"error": "No repositories found"})
                return 1
                
            logger.info(f"Found {len(repos)} repositories in {org_name}")
            
            # Fetch content
            content = content_fetcher.fetch_multiple_repositories(
                org_name, 
                progress_callback=progress_callback,
                _cancellation_event=cancellation_event
            )
            
            # Check for cancellation
            if check_cancelled():
                logger.info("Operation cancelled by user")
                if task_id:
                    task_tracker.cancel_task(task_id)
                return 1
            
            if not content:
                logger.warning("No content found in repositories")
                if task_id:
                    task_tracker.complete_task(task_id, success=False, 
                                              result={"error": "No content found"})
                return 1
                
            logger.info(f"Processing {len(content)} files...")
            
            # Create or update dataset
            success, dataset = dataset_creator.create_and_push_dataset(
                file_data_list=content,
                dataset_name=dataset_name,
                description=f"SDK documentation from {org_name} GitHub organization",
                source_info=org_name,
                update_existing=True
            )
            
            # Final cancellation check
            if check_cancelled():
                logger.info("Operation cancelled by user during dataset creation")
                if task_id:
                    task_tracker.cancel_task(task_id)
                return 1
            
            if success:
                logger.info(f"Dataset '{dataset_name}' updated successfully")
                if task_id:
                    task_tracker.complete_task(task_id, success=True)
                return 0
            else:
                logger.error("Failed to update dataset")
                if task_id:
                    task_tracker.complete_task(task_id, success=False, 
                                              result={"error": "Failed to update dataset"})
                return 1
                
        # Handle repository update
        elif args.repository:
            repo_url = args.repository
            dataset_name = args.dataset_name
            
            logger.info(f"Updating dataset '{dataset_name}' from repository: {repo_url}")
            
            # Create task for tracking
            if not task_id:
                task_id = task_tracker.create_task(
                    "repository_update",
                    {"repo_url": repo_url, "dataset_name": dataset_name},
                    f"Updating dataset '{dataset_name}' from repository {repo_url}"
                )
                
            # Define progress callback
            def progress_callback(percent, message=None):
                # Check for cancellation
                if check_cancelled():
                    if message:
                        logger.info(f"Cancelled at {percent:.0f}% - {message}")
                    else:
                        logger.info(f"Cancelled at {percent:.0f}%")
                    return
                
                if message:
                    logger.info(f"Progress: {percent:.0f}% - {message}")
                else:
                    logger.info(f"Progress: {percent:.0f}%")
                    
                if task_id:
                    task_tracker.update_task_progress(task_id, percent)
                    
            # Create or update dataset
            result = dataset_creator.create_dataset_from_repository(
                repo_url=repo_url,
                dataset_name=dataset_name,
                description=f"SDK documentation from {repo_url}",
                progress_callback=progress_callback,
                _cancellation_event=cancellation_event,
                update_existing=True
            )
            
            # Check for cancellation
            if check_cancelled():
                logger.info("Operation cancelled by user")
                if task_id:
                    task_tracker.cancel_task(task_id)
                return 1
            
            if result.get("success"):
                logger.info(f"Dataset '{dataset_name}' updated successfully")
                if task_id:
                    task_tracker.complete_task(task_id, success=True)
                return 0
            else:
                logger.error(f"Failed to update dataset: {result.get('message', 'Unknown error')}")
                if task_id:
                    task_tracker.complete_task(task_id, success=False, 
                                              result={"error": result.get('message', 'Unknown error')})
                return 1
                
        else:
            logger.error("No organization or repository specified")
            return 1
            
    except Exception as e:
        logger.error(f"Error during update: {e}", exc_info=True)
        if task_id:
            task_tracker.complete_task(task_id, success=False, result={"error": str(e)})
        return 1

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(sig, frame):
        """Handle signals like CTRL+C by setting the cancellation event."""
        if sig == signal.SIGINT:
            print("\n\nReceived interrupt signal (Ctrl+C). Cancelling operations and shutting down...")
        elif sig == signal.SIGTERM:
            print("\n\nReceived termination signal. Cancelling operations and shutting down...")
        
        # Set the cancellation event to stop ongoing tasks
        global_cancellation_event.set()
        
        # Set a flag to exit after current operation
        current_thread().exit_requested = True
        
        # Make sure we don't handle the same signal again (let default handler take over if needed)
        signal.signal(sig, signal.SIG_DFL)
        
        # Don't exit immediately - let the application handle the shutdown gracefully
        # The application will check the cancellation event and exit cleanly
    
    # Set up the signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Add an exit flag to the main thread
    current_thread().exit_requested = False

def clean_shutdown():
    """Perform a clean shutdown of the application."""
    logger.info("Performing clean shutdown...")
    
    # Stop server if running
    if is_server_running():
        print("\nStopping OpenAPI Endpoints...")
        stop_server()
    
    # Cancel any running background threads
    from github.content_fetcher import shutdown_executor
    shutdown_executor()
    
    print("\nApplication has been shut down.")

def main():
    """Main entry point for the application."""
    setup_logging()
    
    # Setup signal handlers
    setup_signal_handlers()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="SDK Dataset Generator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Update command
    update_parser = subparsers.add_parser("update", help="Update an existing dataset")
    update_parser.add_argument("--organization", help="GitHub organization name")
    update_parser.add_argument("--repository", help="GitHub repository URL")
    update_parser.add_argument("--dataset-name", required=True, help="Dataset name to update")
    update_parser.add_argument("--task-id", help="Task ID for tracking")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Handle command-line mode
        if args.command == "update":
            result = run_update(args)
            clean_shutdown()
            return result
        else:
            # No command or unknown command, run interactive CLI
            run_cli()
            clean_shutdown()
            return 0
    except KeyboardInterrupt:
        # This should now be caught by our signal handler first,
        # but keep this as a fallback
        logger.info("KeyboardInterrupt received in main()")
        clean_shutdown()
        print("\nApplication terminated by user.")
        return 0
    except Exception as e:
        print(f"\nError: Application failed: {e}")
        logger.critical(f"Application failed with error: {e}")
        logger.debug(f"Traceback: {traceback.format_exc()}")
        clean_shutdown()
        return 1


if __name__ == "__main__":
    sys.exit(main())
