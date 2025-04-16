import os
import logging
from pathlib import Path

# Base directories
APP_DIR = Path(os.path.expanduser("~/.github_hf_dataset_creator"))
CACHE_DIR = APP_DIR / "cache"
LOG_DIR = APP_DIR / "logs"
CONFIG_DIR = APP_DIR / "config"
TEMP_DIR = APP_DIR / "temp"  # For temporary files during dataset creation

# Ensure directories exist
for directory in [APP_DIR, CACHE_DIR, LOG_DIR, CONFIG_DIR, TEMP_DIR]:
    directory.mkdir(exist_ok=True, parents=True)
    
# Server settings (defaults, can be overridden in config)
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080

# GitHub settings
GITHUB_API_URL = "https://api.github.com"
GITHUB_MAX_RETRIES = 3
GITHUB_TIMEOUT = 30
GITHUB_DEFAULT_BRANCH = "main"
GITHUB_DOWNLOAD_RETRIES = 5  # Specific retry count for file downloads

# Repository content settings
RELEVANT_FOLDERS = [
    "doc",
    "docs",
    "documentation",
    "example",
    "examples",
    "sample",
    "samples",
    "cookbook",
    "cookbooks",
    "tutorial",
    "tutorials",
    "guide",
    "guides",
]
IGNORED_DIRS = [".git", "node_modules", "__pycache__", "build", "dist"]
MAX_FILE_SIZE_MB = 10
TEXT_FILE_EXTENSIONS = [
    ".md",
    ".txt",
    ".py",
    ".js",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".html",
    ".css",
    ".json",
    ".yaml",
    ".yml",
    ".rst",
    ".ipynb",
]

# Hugging Face settings
HF_DATASET_TEMPLATE = {
    "metadata": {
        "creator": "github_hf_dataset_creator",
        "timestamp": "",
        "source": "",
        "version": "1.0",
    },
    "config": {"features": {"text": "string", "metadata": "dict"}},
}
HF_DEFAULT_REPO_TYPE = "dataset"

# Logging settings
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO
LOG_FILE = LOG_DIR / "app.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
LOG_BACKUP_COUNT = 3

# Parallel processing settings
PARALLEL_MAX_WORKERS = None  # None means use CPU count - 1
PARALLEL_CHUNK_SIZE = 10  # Number of items to process at once in parallel
ASYNC_MAX_WORKERS = None  # None means use CPU count * 2 for IO-bound tasks
