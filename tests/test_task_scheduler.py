import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
from datetime import datetime

# Ensure the package root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.task_scheduler import TaskScheduler


class TestTaskScheduler(unittest.TestCase):
    """Tests for the task scheduler functionality."""

    def setUp(self):
        """Set up the test environment."""
        # Mock the CronTab class
        self.crontab_patcher = patch('utils.task_scheduler.CronTab')
        self.mock_crontab_class = self.crontab_patcher.start()
        self.mock_crontab = MagicMock()
        self.mock_crontab_class.return_value = self.mock_crontab
        
        # Create a scheduler instance with mocked crontab
        self.scheduler = TaskScheduler(username="test_user")
        
        # Mock the schedules directory
        self.schedules_dir_patcher = patch('utils.task_scheduler.SCHEDULES_DIR')
        self.mock_schedules_dir = self.schedules_dir_patcher.start()
        self.mock_schedules_dir.glob.return_value = []

    def tearDown(self):
        """Clean up after each test."""
        self.crontab_patcher.stop()
        self.schedules_dir_patcher.stop()

    def test_list_scheduled_tasks(self):
        """Test listing scheduled tasks."""
        # Mock task file data
        mock_task_data = {
            "id": "task123",
            "type": "update",
            "source_type": "repository",
            "source_name": "https://github.com/test/repo",
            "dataset_name": "test-dataset",
            "schedule_type": "daily",
            "created_at": "2023-01-01T00:00:00"
        }
        
        # Set up mock file glob and open
        mock_file = MagicMock()
        mock_file.name = "task123.json"
        self.mock_schedules_dir.glob.return_value = [mock_file]
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))):
            # Mock next run time calculation for cron job
            mock_job = MagicMock()
            mock_schedule = MagicMock()
            mock_next_run = MagicMock()
            mock_next_run.strftime.return_value = "2023-01-02 00:00:00"
            mock_schedule.get_next.return_value = mock_next_run
            mock_job.schedule.return_value = mock_schedule
            mock_job.command = "--task-id task123"
            self.mock_crontab.__iter__.return_value = [mock_job]
            
            # Call the method
            tasks = self.scheduler.list_scheduled_tasks()
            
            # Verify the result
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["id"], "task123")
            self.assertEqual(tasks[0]["next_run"], "2023-01-02 00:00:00")

    def test_get_schedule_description(self):
        """Test getting human-readable schedule descriptions."""
        # Create a mock cron job with different patterns
        daily_job = MagicMock()
        daily_job.__str__.return_value = "0 0 * * *"
        
        weekly_job = MagicMock()
        weekly_job.__str__.return_value = "0 0 * * 0"
        
        monthly_job = MagicMock()
        monthly_job.__str__.return_value = "0 0 1 * *"
        
        custom_job = MagicMock()
        custom_job.__str__.return_value = "30 12 * * 1,3,5"
        
        # Test each pattern
        self.assertEqual(self.scheduler._get_schedule_description(daily_job), "Daily at midnight")
        self.assertEqual(self.scheduler._get_schedule_description(weekly_job), "Weekly on Sunday at midnight")
        self.assertEqual(self.scheduler._get_schedule_description(monthly_job), "Monthly on the 1st at midnight")
        self.assertEqual(self.scheduler._get_schedule_description(custom_job), "Custom schedule: 30 12 * * 1,3,5")

    def test_create_scheduled_task(self):
        """Test creating a scheduled task."""
        # Set up mocks for cron job
        mock_job = MagicMock()
        self.mock_crontab.new.return_value = mock_job
        
        # Mock datetime for predictable task_id
        mock_datetime = MagicMock()
        mock_datetime.now().strftime.return_value = "20230101_120000"
        
        with patch('utils.task_scheduler.datetime', mock_datetime), \
             patch('builtins.open', mock_open()), \
             patch('json.dump') as mock_json_dump:
            
            # Call the method
            task_id = self.scheduler.create_scheduled_task(
                task_type="update",
                source_type="repository",
                source_name="https://github.com/test/repo",
                dataset_name="test-dataset",
                schedule_type="daily"
            )
            
            # Verify the result
            self.assertEqual(task_id, "scheduled_update_20230101_120000")
            mock_job.setall.assert_called_with("0 0 * * *")
            self.mock_crontab.write.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_delete_scheduled_task(self):
        """Test deleting a scheduled task."""
        # Set up mocks for cron job
        mock_job = MagicMock()
        mock_job.command = "--task-id task123"
        self.mock_crontab.__iter__.return_value = [mock_job]
        
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_schedules_dir.__truediv__.return_value = mock_file
        
        # Call the method
        result = self.scheduler.delete_scheduled_task("task123")
        
        # Verify the result
        self.assertTrue(result)
        self.mock_crontab.remove.assert_called_once_with(mock_job)
        self.mock_crontab.write.assert_called_once()
        mock_file.unlink.assert_called_once()

    def test_update_scheduled_task(self):
        """Test updating a scheduled task."""
        # Set up mocks for cron job
        mock_job = MagicMock()
        mock_job.command = "--task-id task123"
        self.mock_crontab.__iter__.return_value = [mock_job]
        
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_schedules_dir.__truediv__.return_value = mock_file
        
        # Mock task data
        mock_task_data = {
            "id": "task123",
            "schedule_type": "daily"
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))), \
             patch('json.load', return_value=mock_task_data), \
             patch('json.dump') as mock_json_dump:
            
            # Call the method
            result = self.scheduler.update_scheduled_task(
                task_id="task123",
                schedule_type="weekly"
            )
            
            # Verify the result
            self.assertTrue(result)
            mock_job.setall.assert_called_with("0 0 * * 0")
            self.mock_crontab.write.assert_called_once()
            mock_json_dump.assert_called_once()

    def test_get_task_details(self):
        """Test getting task details."""
        # Mock task file
        mock_file = MagicMock()
        mock_file.exists.return_value = True
        self.mock_schedules_dir.__truediv__.return_value = mock_file
        
        # Mock task data
        mock_task_data = {
            "id": "task123",
            "schedule_type": "daily"
        }
        
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_task_data))), \
             patch('json.load', return_value=mock_task_data):
            
            # Call the method
            result = self.scheduler.get_task_details("task123")
            
            # Verify the result
            self.assertEqual(result, mock_task_data)

    def test_is_crontab_available(self):
        """Test checking if crontab is available."""
        # Test when crontab is available
        result = self.scheduler.is_crontab_available()
        self.assertTrue(result)
        
        # Test when crontab is not available
        self.scheduler.crontab = None
        result = self.scheduler.is_crontab_available()
        self.assertFalse(result)

    def test_run_task_now(self):
        """Test running a task immediately."""
        # Mock task details
        mock_task_data = {
            "id": "task123",
            "command": "python main.py --task-id task123"
        }
        
        with patch.object(self.scheduler, 'get_task_details', return_value=mock_task_data), \
             patch('subprocess.Popen') as mock_popen:
            
            # Call the method
            result = self.scheduler.run_task_now("task123")
            
            # Verify the result
            self.assertTrue(result)
            mock_popen.assert_called_once()


if __name__ == "__main__":
    unittest.main()