import requests
import json
import sys
import os

# Base URL for the API
BASE_URL = "http://localhost:8080"

# Your API key
API_KEY = "your-api-key-here"


def make_api_request(endpoint, payload):
    """Make an authenticated request to the API."""
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = requests.post(f"{BASE_URL}/{endpoint}", json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def generate_dataset():
    """Generate a dataset from a GitHub repository."""
    payload = {
        "source_type": "repository",  # or "organization"
        "source_name": "https://github.com/username/repo",  # or organization name
        "dataset_name": "example-dataset",
        "description": "An example dataset generated via API",
    }
    
    print("Generating dataset...")
    result = make_api_request("generate", payload)
    
    if result and result.get("success"):
        print(f"Success: {result.get('message')}")
    else:
        print(f"Failed: {result.get('message')}")


def view_dataset_info(dataset_id):
    """View information about a dataset."""
    payload = {"action": "view", "dataset_id": dataset_id}
    
    print(f"Getting info for dataset '{dataset_id}'...")
    result = make_api_request("modify", payload)
    
    if result and result.get("success"):
        print(f"Dataset: {result['data']['id']}")
        print(f"Description: {result['data']['description']}")
        print(f"Created: {result['data']['created_at']}")
        print(f"Last modified: {result['data']['last_modified']}")
        print(f"Downloads: {result['data']['downloads']}")
        print(f"Likes: {result['data']['likes']}")
        print(f"Tags: {', '.join(result['data']['tags']) if result['data']['tags'] else 'None'}")
    else:
        print(f"Failed: {result.get('message')}")


def download_dataset_metadata(dataset_id):
    """Download metadata for a dataset."""
    payload = {"action": "download", "dataset_id": dataset_id}
    
    print(f"Downloading metadata for dataset '{dataset_id}'...")
    result = make_api_request("modify", payload)
    
    if result and result.get("success"):
        print(f"Success: {result.get('message')}")
        print(f"Saved to: {result['data']['path']}")
    else:
        print(f"Failed: {result.get('message')}")


def delete_dataset(dataset_id):
    """Delete a dataset."""
    payload = {"action": "delete", "dataset_id": dataset_id}
    
    print(f"Deleting dataset '{dataset_id}'...")
    result = make_api_request("modify", payload)
    
    if result and result.get("success"):
        print(f"Success: {result.get('message')}")
    else:
        print(f"Failed: {result.get('message')}")


def print_usage():
    """Print usage information."""
    print("Usage:")
    print("  python api_client_example.py generate")
    print("  python api_client_example.py view <dataset_id>")
    print("  python api_client_example.py download <dataset_id>")
    print("  python api_client_example.py delete <dataset_id>")


if __name__ == "__main__":
    # Validate API key
    if API_KEY == "your-api-key-here":
        print("Please set your API key in the script before running")
        sys.exit(1)
    
    # Check schema endpoint for OpenAPI
    try:
        schema_response = requests.get(f"{BASE_URL}/openapi.json")
        if schema_response.status_code != 200:
            print("API server not available or not responding to OpenAPI schema requests")
            print(f"Status code: {schema_response.status_code}")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API server. Make sure it's running.")
        sys.exit(1)
    
    # Process command line arguments
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "generate":
        generate_dataset()
    elif command == "view" and len(sys.argv) == 3:
        view_dataset_info(sys.argv[2])
    elif command == "download" and len(sys.argv) == 3:
        download_dataset_metadata(sys.argv[2])
    elif command == "delete" and len(sys.argv) == 3:
        delete_dataset(sys.argv[2])
    else:
        print_usage()
        sys.exit(1)