# OtherTales SDK Dataset Generator for SDK Code Compliant LLM Training

A CLI application that creates datasets compatible with the Hugging Face Datasets Library for fine-tuning and training modes, ensuring code-generation or modification tasks are SDK Compliant ("SDK Aware").

![OtherTales SDK Dataset Generator](https://img.shields.io/badge/othertales-SDK%20Dataset%20Generator-blue)

## Overview

> **Note:** This application has been simplified to a CLI-only interface from the previous TUI implementation.

The **othertales SDK Dataset Generator** addresses the challenge of models producing outdated or invalid code due to incomplete or stale SDK/API knowledge. While RAG agents and commercial search services can help, they often incur high token costs. This tool operates entirely locally—no LLMs or API tokens are required for core dataset creation—by fetching and processing official documentation and examples directly from GitHub.

With a single CLI command, users can point to a GitHub repository or organization, and the tool will automatically:

- Discover and extract content from folders like `docs/`, `examples/`, `samples/`, `cookbooks/`, and notebooks.
- Process multi-format files (Python, Markdown, Jupyter, JSON, etc.) into structured, training-ready datasets.
- Push or store the resulting datasets for integration into Hugging Face Hub or local SMT fine-tuning pipelines.

The generated datasets guarantee 100% accurate and verified coverage of the latest SDK/API documentation for the libraries most critical to your development workflow.

## Key Features

- **SDK Compliant Datasets**: Automatically generate datasets that reflect the most up-to-date SDK/API requirements, eliminating guesswork.
- **Local, Token-Free Processing**: No LLM calls or commercial API tokens needed for dataset creation—processing happens entirely on your machine.
- **Repository & Organization Support**: Input either a single GitHub repo URL (e.g., `https://github.com/langchain-ai/langgraph`) or an organization URL (e.g., `https://github.com/langchain-ai`) to fetch all public repos.
- **Multi-Format Extraction**: Searches for documentation, examples, samples, cookbooks, and notebooks across each target repository.
- **Hugging Face Datasets Integration**: Outputs compatible with the Hugging Face Datasets Library for seamless pushing to the Hub.
- **SMT Fine-Tuning Ready**: Tailor-made datasets for sequence-to-sequence, code generation, and code modification fine-tuning tasks.
- **OpenAPI Tools Server**: Built-in server exposing OpenAPI-compatible endpoints, enabling models with tools support (e.g., via OpenWebUI or custom hosts) to request dataset generation within chat completions. Configuration requires only GitHub/Hugging Face usernames and tokens as environment variables.
- **Concurrent Operation**: The OpenAPI server runs alongside direct CLI tasks, available on demand with minimal configuration.
- **Task Tracking & Resuming**: Keep track of dataset generation jobs, resume interrupted tasks, and modify existing datasets and metadata.
- **Scheduling & Cron Support**: Automate and schedule recurring dataset creation or update tasks to stay in sync with upstream repository changes.

## Installation

### Prerequisites

- Python 3.7+
- Git
- GitHub and Hugging Face accounts (for API tokens if pushing to the Hub or using the OpenAPI server)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/othertales-sdk-dataset-generator.git
   cd othertales-sdk-dataset-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your credentials:
   ```bash
   # Via CLI menu
   python main.py
   # Select "3. Manage Credentials"

   # Or programmatically
   python -c "from config.credentials_manager import CredentialsManager; cm = CredentialsManager(); cm.save_github_credentials('your_username','your_token'); cm.save_huggingface_credentials('your_hf_username','your_hf_token')"
   ```

## Usage

### Command-line Interface

Generate datasets from a repository:

```bash
python main.py
# Select "1. Generate Dataset"
# Enter a repository or organization URL when prompted
```

The tool will fetch all public repositories (if an organization URL is provided), extract relevant documentation and examples, and create Hugging Face compatible datasets.

### Python API

Use the library directly in your Python scripts or notebooks:

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

# Fetch and create a dataset
repo_url = "https://github.com/huggingface/datasets"
dataset = content_fetcher.fetch_and_create_dataset(
    repo_url=repo_url,
    dataset_name="my-sdk-aware-dataset",
    description="Up-to-date SDK documentation dataset",
)
```

## Docker

Run with Docker Compose:
```bash
export GITHUB_TOKEN=your_github_token
export HUGGINGFACE_TOKEN=your_hf_token
docker-compose up
```

## Project Structure

- `github/`: GitHub API integration and repository content fetching
- `huggingface/`: Hugging Face Hub integration for dataset creation and management
- `processors/`: File processors for different formats (Python, Markdown, notebooks, etc.)
- `config/`: Configuration and credential management
- `server/`: OpenAPI-compatible tools server implementation
- `utils/`: Utility functions for performance and system management
- `tests/`: Test suite for the entire application
- `notebooks/`: Example Jupyter notebooks

## Testing

Run the test suite:
```bash
pytest
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/awesome-feature`)
3. Commit your changes (`git commit -m 'Add awesome feature'`)
4. Push to the branch (`git push origin feature/awesome-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
