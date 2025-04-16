import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from utils.env_loader import load_environment_variables


@pytest.fixture
def mock_env_file():
    """Fixture to mock the .env file."""
    with patch("pathlib.Path.exists", return_value=True):
        yield


def test_load_environment_variables_from_env_file(mock_env_file):
    """Test loading environment variables from a .env file."""
    with patch("dotenv.load_dotenv") as mock_load_dotenv:
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "env_github_token",
                "GITHUB_USERNAME": "env_github_user",
                "HUGGINGFACE_TOKEN": "env_hf_token",
                "HUGGINGFACE_USERNAME": "env_hf_user",
            },
        ):
            env_vars = load_environment_variables()
            mock_load_dotenv.assert_called_once()
            assert env_vars["github_token"] == "env_github_token"
            assert env_vars["github_username"] == "env_github_user"
            assert env_vars["huggingface_token"] == "env_hf_token"
            assert env_vars["huggingface_username"] == "env_hf_user"


def test_load_environment_variables_from_system_env():
    """Test loading environment variables from system environment."""
    with patch("pathlib.Path.exists", return_value=False):
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "system_github_token",
                "GITHUB_USERNAME": "system_github_user",
                "HUGGINGFACE_TOKEN": "system_hf_token",
                "HUGGINGFACE_USERNAME": "system_hf_user",
            },
        ):
            env_vars = load_environment_variables()
            assert env_vars["github_token"] == "system_github_token"
            assert env_vars["github_username"] == "system_github_user"
            assert env_vars["huggingface_token"] == "system_hf_token"
            assert env_vars["huggingface_username"] == "system_hf_user"


def test_load_environment_variables_no_env_file():
    """Test behavior when no .env file is present."""
    with patch("pathlib.Path.exists", return_value=False):
        with patch.dict(
            os.environ,
            {
                "GITHUB_TOKEN": "system_github_token",
                "GITHUB_USERNAME": "system_github_user",
                "HUGGINGFACE_TOKEN": "system_hf_token",
                "HUGGINGFACE_USERNAME": "system_hf_user",
            },
        ):
            env_vars = load_environment_variables()
            assert env_vars["github_token"] == "system_github_token"
            assert env_vars["github_username"] == "system_github_user"
            assert env_vars["huggingface_token"] == "system_hf_token"
            assert env_vars["huggingface_username"] == "system_hf_user"
