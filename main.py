import sys
import os
import logging
from utils.logging_config import setup_logging
from config.credentials_manager import CredentialsManager
from huggingface.dataset_manager import DatasetManager


def run_cli():
    """Run the command-line interface."""
    print("\n===== othertales SDK Dataset Generator =====")
    print("CLI mode\n")
    
    # Initialize managers and clients
    credentials_manager = CredentialsManager()
    dataset_manager = DatasetManager(credentials_manager=credentials_manager)
    github_client = None
    dataset_creator = None
    
    print("Credentials manager initialized successfully")
    
    while True:
        print("\nMain Menu:")
        print("1. Generate Dataset")
        print("2. Manage Existing Datasets")
        print("3. Manage Credentials")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ")
        
        if choice == "1":
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
                    
                    print(f"\nFetching repositories from organization: {org_name}")
                    repos = content_fetcher.fetch_organization_repositories(org_name)
                    
                    if not repos:
                        print(f"No repositories found for organization: {org_name}")
                        continue
                        
                    print(f"Found {len(repos)} repositories")
                    print("Fetching repository content...")
                    
                    content = content_fetcher.fetch_multiple_repositories(org_name)
                    
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
                
        elif choice == "2":
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
                
        elif choice == "3":
            print("\n----- Manage Credentials -----")
            print("1. Set GitHub Credentials")
            print("2. Set Hugging Face Credentials")
            print("3. Return to main menu")
            
            cred_choice = input("\nEnter choice (1-3): ")
            
            if cred_choice == "1":
                github_username = input("Enter GitHub username: ")
                github_token = input("Enter GitHub token (will not be shown): ")
                
                try:
                    credentials_manager.save_github_credentials(github_username, github_token)
                    print("GitHub credentials saved successfully")
                except Exception as e:
                    print(f"Error saving GitHub credentials: {e}")
                    
            elif cred_choice == "2":
                hf_username = input("Enter Hugging Face username: ")
                hf_token = input("Enter Hugging Face token (will not be shown): ")
                
                try:
                    credentials_manager.save_huggingface_credentials(hf_username, hf_token)
                    print("Hugging Face credentials saved successfully")
                except Exception as e:
                    print(f"Error saving Hugging Face credentials: {e}")
                    
            elif cred_choice == "3":
                continue
                
            else:
                print("Invalid choice")
                
        elif choice == "4":
            print("\nExiting application. Goodbye!")
            break
            
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


def main():
    """Main entry point for the application."""
    setup_logging()
    
    try:
        run_cli()
        return 0
    except KeyboardInterrupt:
        print("\n\nApplication terminated by user.")
        return 0
    except Exception as e:
        print(f"\nError: Application failed: {e}")
        logging.critical(f"Application failed with error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
