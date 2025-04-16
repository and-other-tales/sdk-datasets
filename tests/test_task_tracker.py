import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
from datetime import datetime

# Ensure the package root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.task_tracker import TaskTracker


class TestTaskTracker(unittest.TestCase):
    """Tests for the task tracker functionality."""

    def setUp(self):
        """Set up the test environment."""
        # Create a tracker instance
        self.tracker = TaskTracker()
        
        # Mock the tasks directory
        self.tasks_dir_patcher = patch('utils.task_tracker.TASKS_DIR')
        self.mock_tasks_dir = self.tasks_dir_patcher.start()
        
        # Mock the cache directory
        self.cache_dir_patcher = patch('utils.task_tracker.CACHE_DIR')
        self.mock_cache_dir = self.cache_dir_patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        self.tasks_dir_patcher.stop()
        self.cache_dir_patcher.stop()

    def test_create_task(self):
        """Test creating a new task."""
        # Mock datetime for predictable task_id
        mock_datetime = MagicMock()
        mock_datetime.now().strftime.return_value = "20230101_120000"
        mock_datetime.now().isoformat.return_value = "2023-01-01T12:00:00"
        
        with patch('utils.task_tracker.datetime', mock_datetime), \
             patch('builtins.open', mock_open()), \
             patch('json.dump') as mock_json_dump:
            
            # Call the method
            task_id = self.tracker.create_task(
                task_type="repository",
                params={"repo_url": "https://github.com/test/repo"},
                description="Test repository dataset creation"
            )
            
            # Verify the result
            self.assertEqual(task_id, "repository_20230101_120000")
            
            # Check json dump was called with the right data
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            task_data = args[0]
            
            self.assertEqual(task_data["id"], "repository_20230101_120000")
            self.assertEqual(task_data["type"], "repository")
            self.assertEqual(task_data["params"], {"repo_url": "https://github.com/test/repo"})
            self.assertEqual(task_data["description"], "Test repository dataset creation")
            self.assertEqual(task_data["status"], "created")
            self.assertEqual(task_data["progress"], 0)

    def test_update_task_progress(self):
        """Test updating task progress."""
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_tasks_dir.__truediv__.return_value = mock_file
        
        # Mock task data
        mock_task_data = {
            "id": "task123",
            "type": "repository",
            "progress": 0,
            "status": "in_progress",
            "current_stage": None,
            "stages": []
        }
        
        # Mock datetime
        mock_datetime = MagicMock()
        mock_datetime.now().isoformat.return_value = "2023-01-01T12:30:00"
        
        with patch('utils.task_tracker.datetime', mock_datetime), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))), \
             patch('json.load', return_value=mock_task_data), \
             patch('json.dump') as mock_json_dump:
            
            # Call the method with a new stage
            result = self.tracker.update_task_progress(
                task_id="task123",
                progress=25,
                stage="downloading",
                stage_progress=50
            )
            
            # Verify the result
            self.assertTrue(result)
            
            # Check json dump was called with the right data
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            updated_data = args[0]
            
            self.assertEqual(updated_data["progress"], 25)
            self.assertEqual(updated_data["current_stage"], "downloading")
            self.assertEqual(updated_data["stage_progress"], 50)
            self.assertEqual(updated_data["stage_started_at"], "2023-01-01T12:30:00")
            
            # Now test updating progress within the same stage
            mock_json_dump.reset_mock()
            mock_task_data["current_stage"] = "downloading"
            mock_task_data["stage_progress"] = 50
            
            with patch('json.load', return_value=mock_task_data):
                result = self.tracker.update_task_progress(
                    task_id="task123",
                    progress=50,
                    stage="downloading",
                    stage_progress=75
                )
                
                self.assertTrue(result)
                
                # Check stage wasn't reset
                args, _ = mock_json_dump.call_args
                updated_data = args[0]
                self.assertEqual(updated_data["progress"], 50)
                self.assertEqual(updated_data["current_stage"], "downloading")
                self.assertEqual(updated_data["stage_progress"], 75)
                self.assertNotIn("stage_started_at", updated_data)

    def test_complete_task(self):
        """Test completing a task."""
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_tasks_dir.__truediv__.return_value = mock_file
        
        # Mock task data
        mock_task_data = {
            "id": "task123",
            "type": "repository",
            "progress": 75,
            "status": "in_progress",
            "current_stage": "processing",
            "stages": [{"name": "downloading", "completed_at": "2023-01-01T12:15:00"}]
        }
        
        # Mock datetime
        mock_datetime = MagicMock()
        mock_datetime.now().isoformat.return_value = "2023-01-01T13:00:00"
        
        with patch('utils.task_tracker.datetime', mock_datetime), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))), \
             patch('json.load', return_value=mock_task_data), \
             patch('json.dump') as mock_json_dump:
            
            # Call the method
            result = self.tracker.complete_task(
                task_id="task123",
                success=True,
                result={"dataset_url": "https://huggingface.co/datasets/test-dataset"}
            )
            
            # Verify the result
            self.assertTrue(result)
            
            # Check json dump was called with the right data
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            updated_data = args[0]
            
            self.assertEqual(updated_data["status"], "completed")
            self.assertEqual(updated_data["completed_at"], "2023-01-01T13:00:00")
            self.assertEqual(updated_data["progress"], 100)
            self.assertEqual(updated_data["result"], {"dataset_url": "https://huggingface.co/datasets/test-dataset"})
            self.assertIsNone(updated_data["current_stage"])
            self.assertEqual(len(updated_data["stages"]), 2)
            self.assertEqual(updated_data["stages"][1]["name"], "processing")

    def test_cancel_task(self):
        """Test cancelling a task."""
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_tasks_dir.__truediv__.return_value = mock_file
        
        # Mock task data
        mock_task_data = {
            "id": "task123",
            "type": "repository",
            "progress": 50,
            "status": "in_progress"
        }
        
        # Mock datetime
        mock_datetime = MagicMock()
        mock_datetime.now().isoformat.return_value = "2023-01-01T13:00:00"
        
        with patch('utils.task_tracker.datetime', mock_datetime), \
             patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))), \
             patch('json.load', return_value=mock_task_data), \
             patch('json.dump') as mock_json_dump:
            
            # Call the method
            result = self.tracker.cancel_task("task123")
            
            # Verify the result
            self.assertTrue(result)
            
            # Check json dump was called with the right data
            mock_json_dump.assert_called_once()
            args, _ = mock_json_dump.call_args
            updated_data = args[0]
            
            self.assertEqual(updated_data["status"], "cancelled")
            self.assertEqual(updated_data["cancelled_at"], "2023-01-01T13:00:00")

    def test_get_task(self):
        """Test getting task details."""
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_tasks_dir.__truediv__.return_value = mock_file
        
        # Mock task data
        mock_task_data = {
            "id": "task123",
            "type": "repository",
            "progress": 75,
            "status": "in_progress"
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))), \
             patch('json.load', return_value=mock_task_data):
            
            # Call the method
            task = self.tracker.get_task("task123")
            
            # Verify the result
            self.assertEqual(task, mock_task_data)
            
            # Test non-existent task
            mock_file.exists.return_value = False
            task = self.tracker.get_task("nonexistent")
            self.assertIsNone(task)

    def test_list_resumable_tasks(self):
        """Test listing resumable tasks."""
        # Mock task files
        task1 = MagicMock()
        task1.name = "task1.json"
        task2 = MagicMock()
        task2.name = "task2.json"
        self.mock_tasks_dir.glob.return_value = [task1, task2]
        
        # Mock task data
        task1_data = {
            "id": "task1",
            "type": "repository",
            "progress": 75,
            "status": "in_progress",
            "created_at": "2023-01-01T12:00:00",
            "updated_at": "2023-01-01T12:30:00"
        }
        
        task2_data = {
            "id": "task2",
            "type": "organization",
            "progress": 25,
            "status": "in_progress",
            "created_at": "2023-01-01T13:00:00",
            "updated_at": "2023-01-01T13:10:00"
        }
        
        # Mock datetime calculations
        now = datetime(2023, 1, 1, 14, 0, 0)
        
        # Mock the datetime.now() and fromisoformat() calls
        mock_datetime = MagicMock()
        mock_datetime.now.return_value = now
        mock_datetime.fromisoformat = datetime.fromisoformat
        
        # Set up the mocks for open calls with different task data
        mock_file_opens = {
            "task1.json": mock_open(read_data=json.dumps(task1_data)),
            "task2.json": mock_open(read_data=json.dumps(task2_data))
        }
        
        def side_effect(file_name, *args, **kwargs):
            return mock_file_opens[file_name.name]()
        
        with patch('utils.task_tracker.datetime', mock_datetime), \
             patch('builtins.open', side_effect=side_effect), \
             patch('json.load', side_effect=[task1_data, task2_data]):
            
            # Call the method
            tasks = self.tracker.list_resumable_tasks()
            
            # Verify the result
            self.assertEqual(len(tasks), 2)
            # Should be sorted by updated_at (most recent first)
            self.assertEqual(tasks[0]["id"], "task2")
            self.assertEqual(tasks[1]["id"], "task1")
            # Check that time_ago fields were added
            self.assertEqual(tasks[0]["updated_ago"], "50 minutes ago")
            self.assertEqual(tasks[1]["updated_ago"], "90 minutes ago")

    def test_get_cache_size(self):
        """Test getting cache size."""
        # Mock os.walk to return file structure
        mock_walk_data = [
            ('/cache', ['dir1'], ['file1.txt']),
            ('/cache/dir1', [], ['file2.txt', 'file3.txt'])
        ]
        
        # Mock file sizes
        mock_file_sizes = {
            '/cache/file1.txt': 1024 * 1024,  # 1 MB
            '/cache/dir1/file2.txt': 2 * 1024 * 1024,  # 2 MB
            '/cache/dir1/file3.txt': 3 * 1024 * 1024   # 3 MB
        }
        
        with patch('os.walk', return_value=mock_walk_data), \
             patch('os.path.join', side_effect=lambda a, b: f"{a}/{b}"), \
             patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', side_effect=lambda path: mock_file_sizes[path]):
            
            # Call the method
            cache_size = self.tracker.get_cache_size()
            
            # Verify the result - should be 6 MB total
            self.assertEqual(cache_size, 6)

    def test_clear_cache(self):
        """Test clearing the cache."""
        # Mock cache directory
        self.mock_cache_dir.exists.return_value = True
        mock_dir = MagicMock()
        mock_file = MagicMock()
        mock_dir.is_dir.return_value = True
        mock_file.is_dir.return_value = False
        self.mock_cache_dir.iterdir.return_value = [mock_dir, mock_file]
        
        with patch('shutil.rmtree') as mock_rmtree:
            # Call the method
            result = self.tracker.clear_cache()
            
            # Verify the result
            self.assertTrue(result)
            mock_rmtree.assert_called_once_with(mock_dir)
            mock_file.unlink.assert_called_once()
            
            # Test when cache directory doesn't exist
            self.mock_cache_dir.exists.return_value = False
            result = self.tracker.clear_cache()
            self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()