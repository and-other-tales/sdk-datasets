import os
import dotenv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def load_environment_variables():
    """
    Load environment variables from .env file and system environment.

    Returns:
        dict: Dictionary of environment variables
    """
    # Try to load from .env file if it exists
    env_file = Path(".env")
    if env_file.exists():
        dotenv.load_dotenv()
        logger.info("Loaded environment variables from .env file")

    # Get environment variables relevant to the application
    env_vars = {
        "github_token": os.environ.get("GITHUB_TOKEN", ""),
        "github_username": os.environ.get("GITHUB_USERNAME", ""),
        "huggingface_token": os.environ.get("HUGGINGFACE_TOKEN", ""),
        "huggingface_username": os.environ.get("HUGGINGFACE_USERNAME", ""),
    }

    return env_vars
