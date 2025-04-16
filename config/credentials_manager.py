import json
import os
import logging
from pathlib import Path
from config.settings import CONFIG_DIR
from utils.env_loader import load_environment_variables

# Try to import keyring but provide fallback if not available
try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False

logger = logging.getLogger(__name__)


class CredentialsManager:
    """Manages secure storage and retrieval of API credentials."""

    SERVICE_NAME = "github_hf_dataset_creator"
    GITHUB_KEY = "github_token"
    HUGGINGFACE_KEY = "huggingface_token"
    CONFIG_FILE = CONFIG_DIR / "config.json"

    def __init__(self):
        self._ensure_config_file_exists()
        # Load environment variables
        self.env_vars = load_environment_variables()
        # Extract usernames from tokens if available
        self._extract_usernames_from_env()

    def _ensure_config_file_exists(self):
        """Ensure the configuration file exists with default values."""
        if not self.CONFIG_FILE.exists():
            default_config = {"github_username": "", "huggingface_username": ""}
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.CONFIG_FILE.write_text(json.dumps(default_config, indent=2))
            logger.info(f"Created default configuration file at {self.CONFIG_FILE}")

    def _extract_usernames_from_env(self):
        """Try to update usernames in config if we have tokens in env variables."""
        try:
            config = self._load_config()
            updated = False

            # If GitHub token exists in env but no username is configured
            if self.env_vars.get("github_token") and not config.get("github_username"):
                github_username = self.env_vars.get("github_username")
                if github_username:
                    config["github_username"] = github_username
                    updated = True

            # Similarly for HuggingFace
            if self.env_vars.get("huggingface_token") and not config.get(
                "huggingface_username"
            ):
                hf_username = self.env_vars.get("huggingface_username")
                if hf_username:
                    config["huggingface_username"] = hf_username
                    updated = True

            if updated:
                self._save_config(config)
        except Exception as e:
            logger.error(f"Error extracting usernames from env: {e}")

    def save_github_credentials(self, username, token):
        """Save GitHub credentials."""
        try:
            config = self._load_config()
            config["github_username"] = username
            
            # Save token in config file if keyring not available
            if not HAS_KEYRING:
                config["github_token"] = token
                logger.warning("Keyring not available, storing token in config file (less secure)")
            
            self._save_config(config)
            
            # Try to use keyring if available
            if HAS_KEYRING:
                try:
                    keyring.set_password(self.SERVICE_NAME, self.GITHUB_KEY, token)
                except Exception as e:
                    logger.warning(f"Keyring failed, storing token in config file: {e}")
                    config["github_token"] = token
                    self._save_config(config)
                    
            logger.info(f"Saved GitHub credentials for user {username}")
        except Exception as e:
            logger.error(f"Failed to save GitHub credentials: {e}")

    def save_huggingface_credentials(self, username, token):
        """Save Hugging Face credentials."""
        try:
            config = self._load_config()
            config["huggingface_username"] = username
            
            # Save token in config file if keyring not available
            if not HAS_KEYRING:
                config["huggingface_token"] = token
                logger.warning("Keyring not available, storing token in config file (less secure)")
                
            self._save_config(config)
            
            # Try to use keyring if available
            if HAS_KEYRING:
                try:
                    keyring.set_password(self.SERVICE_NAME, self.HUGGINGFACE_KEY, token)
                except Exception as e:
                    logger.warning(f"Keyring failed, storing token in config file: {e}")
                    config["huggingface_token"] = token
                    self._save_config(config)
                    
            logger.info(f"Saved Hugging Face credentials for user {username}")
        except Exception as e:
            logger.error(f"Failed to save Hugging Face credentials: {e}")

    def get_github_credentials(self):
        """Get GitHub credentials with environment variable fallback."""
        config = self._load_config()
        username = config.get("github_username", "")
        token = None

        # Try to get token from keyring if available
        if HAS_KEYRING:
            try:
                token = keyring.get_password(self.SERVICE_NAME, self.GITHUB_KEY)
            except Exception as e:
                logger.warning(f"Error accessing keyring: {e}")

        # If not found in keyring, try config file
        if not token and "github_token" in config:
            token = config.get("github_token")
            logger.info("Using GitHub token from config file")

        # If still not found, check environment variable
        if not token and self.env_vars.get("github_token"):
            token = self.env_vars.get("github_token")
            logger.info("Using GitHub token from environment variables")

        return username, token

    def get_huggingface_credentials(self):
        """Get Hugging Face credentials with environment variable fallback."""
        config = self._load_config()
        username = config.get("huggingface_username", "")
        token = None

        # Try to get token from keyring if available
        if HAS_KEYRING:
            try:
                token = keyring.get_password(self.SERVICE_NAME, self.HUGGINGFACE_KEY)
            except Exception as e:
                logger.warning(f"Error accessing keyring: {e}")

        # If not found in keyring, try config file
        if not token and "huggingface_token" in config:
            token = config.get("huggingface_token")
            logger.info("Using HuggingFace token from config file")

        # If still not found, check environment variable
        if not token and self.env_vars.get("huggingface_token"):
            token = self.env_vars.get("huggingface_token")
            logger.info("Using HuggingFace token from environment variables")

        return username, token

    def _load_config(self):
        """Load configuration from file."""
        try:
            if self.CONFIG_FILE.exists():
                return json.loads(self.CONFIG_FILE.read_text())
            return {"github_username": "", "huggingface_username": ""}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {"github_username": "", "huggingface_username": ""}

    def _save_config(self, config):
        """Save configuration to file."""
        try:
            self.CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.CONFIG_FILE.write_text(json.dumps(config, indent=2))
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
