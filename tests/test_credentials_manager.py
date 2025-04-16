import pytest
import json
import keyring
from pathlib import Path
from unittest.mock import patch, MagicMock
from config.credentials_manager import CredentialsManager


@pytest.fixture
def mock_config_file(tmp_path):
    """Fixture to mock the configuration file."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({"github_username": "", "huggingface_username": ""})
    )
    return config_file


@pytest.fixture
def credentials_manager(mock_config_file):
    """Fixture to create a CredentialsManager instance with a mocked config file."""
    with patch("config.credentials_manager.CONFIG_DIR", mock_config_file.parent), patch(
        "config.credentials_manager.load_environment_variables", return_value={}
    ), patch(
        "config.credentials_manager.CredentialsManager.CONFIG_FILE", mock_config_file
    ):
        cm = CredentialsManager()
        # Clear any potential side effects from initialization
        mock_config_file.write_text(
            json.dumps({"github_username": "", "huggingface_username": ""})
        )
        yield cm


def test_ensure_config_file_exists(credentials_manager, mock_config_file):
    """Test that the configuration file is created if it doesn't exist."""
    assert mock_config_file.exists()
    with open(mock_config_file, "r") as f:
        config = json.load(f)
    assert config == {"github_username": "", "huggingface_username": ""}


def test_save_github_credentials(credentials_manager, mock_config_file):
    """Test saving GitHub credentials."""
    with patch("keyring.set_password") as mock_set_password:
        credentials_manager.save_github_credentials("test_user", "test_token")
        mock_set_password.assert_called_once_with(
            credentials_manager.SERVICE_NAME,
            credentials_manager.GITHUB_KEY,
            "test_token",
        )
    with open(mock_config_file, "r") as f:
        config = json.load(f)
    assert config["github_username"] == "test_user"


def test_save_huggingface_credentials(credentials_manager, mock_config_file):
    """Test saving Hugging Face credentials."""
    with patch("keyring.set_password") as mock_set_password:
        credentials_manager.save_huggingface_credentials("hf_user", "hf_token")
        mock_set_password.assert_called_once_with(
            credentials_manager.SERVICE_NAME,
            credentials_manager.HUGGINGFACE_KEY,
            "hf_token",
        )
    with open(mock_config_file, "r") as f:
        config = json.load(f)
    assert config["huggingface_username"] == "hf_user"


def test_get_github_credentials(credentials_manager):
    """Test retrieving GitHub credentials."""
    # Update the config file manually to ensure empty username
    config = {"github_username": "", "huggingface_username": ""}
    credentials_manager._save_config(config)

    with patch("keyring.get_password", return_value="test_token") as mock_get_password:
        username, token = credentials_manager.get_github_credentials()
        mock_get_password.assert_called_once_with(
            credentials_manager.SERVICE_NAME, credentials_manager.GITHUB_KEY
        )
        assert username == ""
        assert token == "test_token"


def test_get_huggingface_credentials(credentials_manager):
    """Test retrieving Hugging Face credentials."""
    # Update the config file manually to ensure empty username
    config = {"github_username": "", "huggingface_username": ""}
    credentials_manager._save_config(config)

    with patch("keyring.get_password", return_value="hf_token") as mock_get_password:
        username, token = credentials_manager.get_huggingface_credentials()
        mock_get_password.assert_called_once_with(
            credentials_manager.SERVICE_NAME, credentials_manager.HUGGINGFACE_KEY
        )
        assert username == ""
        assert token == "hf_token"


def test_extract_usernames_from_env(credentials_manager, mock_config_file):
    """Test extracting usernames from environment variables."""
    # Reset the config file
    mock_config_file.write_text(
        json.dumps({"github_username": "", "huggingface_username": ""})
    )

    env_vars = {
        "github_token": "test_github_token",
        "github_username": "test_github_user",
        "huggingface_token": "test_hf_token",
        "huggingface_username": "test_hf_user",
    }

    # We need to patch both the load_environment_variables function AND
    # update the env_vars attribute on the credentials_manager
    with patch(
        "config.credentials_manager.load_environment_variables", return_value=env_vars
    ):
        # Directly set the env_vars attribute to ensure it's used by the method
        credentials_manager.env_vars = env_vars
        credentials_manager._extract_usernames_from_env()

    # Verify that the config file was updated with the environment usernames
    with open(mock_config_file, "r") as f:
        config = json.load(f)
    assert config["github_username"] == "test_github_user"
    assert config["huggingface_username"] == "test_hf_user"
