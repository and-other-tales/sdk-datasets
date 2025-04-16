import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Ensure the package root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.server import (
    app,
    set_api_key,
    verify_api_key,
    start_server,
    stop_server,
    is_server_running,
    get_server_info,
)
from fastapi.testclient import TestClient
from fastapi import HTTPException


class TestAPIServer(unittest.TestCase):
    """Tests for the FastAPI server implementation."""

    def setUp(self):
        """Set up the test environment."""
        self.test_api_key = "test-api-key"
        set_api_key(self.test_api_key)
        self.client = TestClient(app)

    def test_api_key_validation(self):
        """Test API key validation."""
        # Test with valid API key
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        response = self.client.get("/", headers=headers)
        self.assertEqual(response.status_code, 200)

        # Test with invalid API key
        headers = {"Authorization": "Bearer invalid-key"}
        response = self.client.get("/generate", headers=headers)
        self.assertEqual(response.status_code, 401)

        # Test with missing API key
        response = self.client.get("/generate")
        self.assertEqual(response.status_code, 403)
        
    def test_root_endpoint(self):
        """Test the root endpoint."""
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        response = self.client.get("/", headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
        
    def test_status_endpoint(self):
        """Test the status endpoint."""
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        response = self.client.get("/status", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertIn("host", data)
        self.assertIn("port", data)
        self.assertIn("version", data)

    @patch("api.server.ContentFetcher")
    @patch("api.server.DatasetCreator")
    @patch("api.server.CredentialsManager")
    def test_generate_endpoint(self, mock_creds, mock_creator, mock_fetcher):
        """Test the generate endpoint."""
        # Setup mocks
        mock_creds_instance = MagicMock()
        mock_creds_instance.get_github_credentials.return_value = ("user", "token")
        mock_creds_instance.get_huggingface_credentials.return_value = ("user", "token")
        mock_creds.return_value = mock_creds_instance

        # Mock successful dataset creation
        mock_creator_instance = MagicMock()
        mock_creator_instance.create_dataset_from_repository.return_value = {"success": True}
        mock_creator.return_value = mock_creator_instance

        # Test repository source type
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        payload = {
            "source_type": "repository",
            "source_name": "https://github.com/test/repo",
            "dataset_name": "test-dataset",
            "description": "Test description",
        }
        response = self.client.post("/generate", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    @patch("api.server.DatasetManager")
    @patch("api.server.CredentialsManager")
    def test_modify_endpoint(self, mock_creds, mock_dataset_manager):
        """Test the modify endpoint."""
        # Setup mocks
        mock_creds_instance = MagicMock()
        mock_creds_instance.get_huggingface_credentials.return_value = ("user", "token")
        mock_creds.return_value = mock_creds_instance

        # Mock dataset manager
        mock_manager_instance = MagicMock()
        mock_info = MagicMock()
        mock_info.id = "test-dataset"
        mock_info.description = "Test description"
        mock_info.created_at = "2023-01-01"
        mock_info.last_modified = "2023-01-02"
        mock_info.downloads = 10
        mock_info.likes = 5
        mock_info.tags = ["test"]
        mock_manager_instance.get_dataset_info.return_value = mock_info
        mock_manager_instance.download_dataset_metadata.return_value = True
        mock_manager_instance.delete_dataset.return_value = True
        mock_dataset_manager.return_value = mock_manager_instance

        # Test view action
        headers = {"Authorization": f"Bearer {self.test_api_key}"}
        payload = {"action": "view", "dataset_id": "test-dataset"}
        response = self.client.post("/modify", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        self.assertEqual(response.json()["data"]["id"], "test-dataset")

        # Test download action
        payload = {"action": "download", "dataset_id": "test-dataset"}
        response = self.client.post("/modify", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

        # Test delete action
        payload = {"action": "delete", "dataset_id": "test-dataset"}
        response = self.client.post("/modify", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

        # Test invalid action
        payload = {"action": "invalid", "dataset_id": "test-dataset"}
        response = self.client.post("/modify", json=payload, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["success"])

    def test_server_management(self):
        """Test server management functions."""
        # Test starting server
        with patch("threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance
            
            result = start_server("test-key")
            mock_thread_instance.start.assert_called_once()
            self.assertIsInstance(result, dict)
            self.assertEqual(result["status"], "running")
            self.assertIn("openapi_url", result)
            self.assertIn("api_docs_url", result)

        # Test server status
        self.assertTrue(is_server_running())

        # Test stopping server
        self.assertTrue(stop_server())
        
        # Server should now be reported as not running
        from api.server import server_status
        server_status.running = False
        self.assertFalse(is_server_running())
        
    def test_get_server_info(self):
        """Test the get_server_info function."""
        from api.server import server_status
        
        # Test when server is running
        server_status.running = True
        server_status.host = "127.0.0.1"
        server_status.port = 8080
        
        info = get_server_info()
        self.assertIsInstance(info, dict)
        self.assertEqual(info["status"], "running")
        self.assertEqual(info["host"], "127.0.0.1")
        self.assertEqual(info["port"], 8080)
        self.assertEqual(info["api_docs_url"], "http://127.0.0.1:8080/docs")
        self.assertEqual(info["openapi_url"], "http://127.0.0.1:8080/openapi.json")
        
        # Test when server is not running
        server_status.running = False
        
        info = get_server_info()
        self.assertEqual(info["status"], "stopped")
        self.assertIsNone(info["api_docs_url"])
        self.assertIsNone(info["openapi_url"])


if __name__ == "__main__":
    unittest.main()