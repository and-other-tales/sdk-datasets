# OtherTales Dataset Generator for SDK Code Compliant LLM Training

A powerful tool to create machine learning datasets from GitHub repositories and push them to Hugging Face Hub.

![OtherTales Dataset Generator for SDK Code Compliant LLM Training](https://img.shields.io/badge/othertales-SDK%20Dataset%20Generator-blue)

## Overview

> **Note:** This application has been simplified to a CLI-only interface from the previous TUI implementation.

This application allows you to:
- Extract content from GitHub repositories
- Process various file types (Python, Markdown, Jupyter notebooks, JSON, etc.)
- Create structured datasets suitable for machine learning
- Push datasets to Hugging Face Hub
- Manage your datasets with a simple CLI interface

## Features

- **Multi-format Support**: Process code, markdown, JSON, and Jupyter notebooks
- **Command-line Interface**: Simple and straightforward interface for dataset creation and management
- **API Access**: Both GitHub and Hugging Face integrations
- **Background Processing**: Efficient parallel processing with cancellation support
- **Comprehensive Metadata**: Automatic extraction of repository structure metadata

## Installation

### Prerequisites

- Python 3.7+
- Git
- GitHub and Hugging Face accounts (for API tokens)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/datasets.git
cd datasets
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your credentials:
```bash
# Run the application
python main.py
# Select "3. Manage Credentials" from the main menu

# Or programmatically
python -c "from config.credentials_manager import CredentialsManager; cm = CredentialsManager(); cm.save_github_credentials('your_username', 'your_token'); cm.save_huggingface_credentials('your_hf_username', 'your_hf_token')"
```

## Usage

### Command-line Interface

Launch the application:

```bash
python main.py
```

This will open the main menu where you can:
- Generate datasets from GitHub repositories or organizations
- Manage existing datasets on Hugging Face
- Manage API credentials
- Exit the application

### Python API

You can also use the library directly in your Python scripts or notebooks:

```python
from github.content_fetcher import ContentFetcher
from huggingface.dataset_creator import DatasetCreator
from config.credentials_manager import CredentialsManager

# Setup credentials
credentials = CredentialsManager()
github_username, github_token = credentials.get_github_credentials()
hf_username, hf_token = credentials.get_huggingface_credentials()

# Initialize components
content_fetcher = ContentFetcher(github_token=github_token)
dataset_creator = DatasetCreator(huggingface_token=hf_token)

# Fetch content from a repository
repo_url = "https://github.com/huggingface/datasets"
file_data = content_fetcher.fetch_content_for_dataset(repo_url)

# Create and push a dataset
dataset = dataset_creator.create_dataset_from_repository(
    repo_url=repo_url,
    dataset_name="my-new-dataset",
    description="Dataset created from GitHub",
)
```

Check the `notebooks` directory for complete usage examples.

### Docker

You can also run the application using Docker:

```bash
# Build and run with Docker Compose
export GITHUB_TOKEN=your_github_token
export HUGGINGFACE_TOKEN=your_huggingface_token
docker-compose up
```

## Project Structure

- `github/`: GitHub API integration and repository content fetching
- `huggingface/`: Hugging Face Hub integration for dataset creation and management
- `processors/`: File processors for different formats (Python, markdown, notebooks, etc.)
- `config/`: Configuration and credential management
- `ui/`: *(Deprecated)* Previous UI implementation
- `utils/`: Utility functions for performance and system management
- `tests/`: Test suite for the entire application
- `notebooks/`: Example Jupyter notebooks

## Testing

Run the test suite:

```bash
pytest
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
