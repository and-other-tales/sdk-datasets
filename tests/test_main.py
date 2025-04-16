import pytest
from unittest.mock import patch, MagicMock, call
from main import main, run_cli


@patch("main.setup_logging")
@patch("main.run_cli")
def test_main_success(mock_run_cli, mock_setup_logging):
    # Call the main function
    result = main()

    # Assert setup_logging was called
    mock_setup_logging.assert_called_once()

    # Assert run_cli was called
    mock_run_cli.assert_called_once()

    # Assert the main function returns 0 on success
    assert result == 0


@patch("main.setup_logging")
@patch("main.run_cli", side_effect=Exception("Test exception"))
def test_main_failure(mock_run_cli, mock_setup_logging):
    # Call the main function
    result = main()

    # Assert setup_logging was called
    mock_setup_logging.assert_called_once()

    # Assert run_cli was called
    mock_run_cli.assert_called_once()

    # Assert the main function returns 1 on failure
    assert result == 1


@patch("builtins.print")
@patch("builtins.input")
@patch("main.CredentialsManager")
@patch("main.DatasetManager")
def test_cli_menu_exit(mock_dataset_manager, mock_creds_manager, mock_input, mock_print):
    # Set up mock input to choose 'Exit' option
    mock_input.return_value = "4"  # Exit option
    
    # Set up mock credentials manager
    mock_cm_instance = MagicMock()
    mock_cm_instance.get_github_credentials.return_value = ("user", "token")
    mock_cm_instance.get_huggingface_credentials.return_value = ("user", "token")
    mock_creds_manager.return_value = mock_cm_instance
    
    # Run the CLI
    run_cli()
    
    # Verify credentials manager was initialized
    mock_creds_manager.assert_called_once()
    
    # Verify the exit message was printed
    mock_print.assert_any_call("\nExiting application. Goodbye!")


@patch("builtins.print")
@patch("builtins.input")
@patch("main.CredentialsManager")
@patch("main.DatasetManager")
def test_cli_manage_credentials(mock_dataset_manager, mock_creds_manager, mock_input, mock_print):
    # Set up mock inputs for credential management flow
    # First choose credentials, then GitHub creds, then exit
    mock_input.side_effect = ["3", "1", "testuser", "testtoken", "4"]
    
    # Set up mock credentials manager
    mock_cm_instance = MagicMock()
    mock_cm_instance.get_github_credentials.return_value = ("user", "token")
    mock_cm_instance.get_huggingface_credentials.return_value = ("user", "token")
    mock_creds_manager.return_value = mock_cm_instance
    
    # Run the CLI
    run_cli()
    
    # Verify credentials manager methods were called
    mock_cm_instance.save_github_credentials.assert_called_once_with("testuser", "testtoken")
    
    # Verify success message was printed
    mock_print.assert_any_call("GitHub credentials saved successfully")
