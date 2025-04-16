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
    """Manages secure storage and retrieval of API credentials and application settings."""

    SERVICE_NAME = "github_hf_dataset_creator"
    GITHUB_KEY = "github_token"
    HUGGINGFACE_KEY = "huggingface_token"
    OPENAPI_KEY = "openapi_key"
    CONFIG_FILE = CONFIG_DIR / "config.json"
    
    # Default settings
    DEFAULT_SERVER_PORT = 8080
    DEFAULT_TEMP_DIR = str(Path(os.path.expanduser("~/.github_hf_dataset_creator/temp")))

    def __init__(self):
        self._ensure_config_file_exists()
        # Load environment variables
        self.env_vars = load_environment_variables()
        # Extract usernames from tokens if available
        self._extract_usernames_from_env()

    def _ensure_config_file_exists(self):
        """Ensure the configuration file exists with default values."""
        if not self.CONFIG_FILE.exists():
            default_config = {
                "github_username": "", 
                "huggingface_username": "",
                "server_port": self.DEFAULT_SERVER_PORT,
                "temp_dir": self.DEFAULT_TEMP_DIR
            }
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
        
    def save_openapi_key(self, key):
        """Save OpenAPI API key."""
        try:
            config = self._load_config()
            
            # Save key in config file if keyring not available
            if not HAS_KEYRING:
                config["openapi_key"] = key
                logger.warning("Keyring not available, storing API key in config file (less secure)")
                
            self._save_config(config)
            
            # Try to use keyring if available
            if HAS_KEYRING:
                try:
                    keyring.set_password(self.SERVICE_NAME, self.OPENAPI_KEY, key)
                except Exception as e:
                    logger.warning(f"Keyring failed, storing API key in config file: {e}")
                    config["openapi_key"] = key
                    self._save_config(config)
                    
            logger.info("Saved OpenAPI API key")
            return True
        except Exception as e:
            logger.error(f"Failed to save OpenAPI API key: {e}")
            return False

    def get_openapi_key(self):
        """Get OpenAPI API key."""
        config = self._load_config()
        key = None

        # Try to get key from keyring if available
        if HAS_KEYRING:
            try:
                key = keyring.get_password(self.SERVICE_NAME, self.OPENAPI_KEY)
            except Exception as e:
                logger.warning(f"Error accessing keyring: {e}")

        # If not found in keyring, try config file
        if not key and "openapi_key" in config:
            key = config.get("openapi_key")
            logger.info("Using OpenAPI key from config file")

        # If still not found, check environment variable
        if not key and self.env_vars.get("openapi_key"):
            key = self.env_vars.get("openapi_key")
            logger.info("Using OpenAPI key from environment variables")

        return key
        
    def get_server_port(self):
        """Get configured server port."""
        config = self._load_config()
        return config.get("server_port", self.DEFAULT_SERVER_PORT)
    
    def save_server_port(self, port):
        """Save server port configuration."""
        try:
            config = self._load_config()
            config["server_port"] = int(port)
            self._save_config(config)
            logger.info(f"Saved server port: {port}")
            return True
        except Exception as e:
            logger.error(f"Failed to save server port: {e}")
            return False
    
    def get_temp_dir(self):
        """Get configured temporary directory path."""
        config = self._load_config()
        return config.get("temp_dir", self.DEFAULT_TEMP_DIR)
    
    def save_temp_dir(self, dir_path):
        """Save temporary directory configuration."""
        try:
            # Ensure directory exists
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
            config = self._load_config()
            config["temp_dir"] = dir_path
            self._save_config(config)
            logger.info(f"Saved temporary directory: {dir_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save temporary directory: {e}")
            return False

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
