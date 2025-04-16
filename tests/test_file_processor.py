import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


@pytest.fixture
def file_processor():
    """Fixture to create a FileProcessor instance."""
    from processors.file_processor import FileProcessor

    return FileProcessor()


@pytest.fixture
def mock_file_data():
    """Fixture to create mock file data."""
    return {
        "name": "example.txt",
        "path": "/repo/example.txt",
        "repo": "test_repo",
        "local_path": "/tmp/example.txt",
        "size": 1024,
        "url": "https://github.com/test_repo/example.txt",
    }


def test_process_file_text(file_processor, mock_file_data):
    with patch("pathlib.Path.exists", return_value=True), patch(
        "pathlib.Path.read_text", return_value="Sample text content"
    ):
        result = file_processor.process_file(mock_file_data)
        assert result is not None
        assert result["text"] == "Sample text content"
        assert result["metadata"]["name"] == "example.txt"


def test_process_file_missing_local_path(file_processor, mock_file_data):
    del mock_file_data["local_path"]
    result = file_processor.process_file(mock_file_data)
    assert "error" in result
    assert "Missing local_path" in result["error"]


def test_process_file_nonexistent_path(file_processor, mock_file_data):
    with patch("pathlib.Path.exists", return_value=False):
        result = file_processor.process_file(mock_file_data)
        assert "error" in result
        assert "File does not exist" in result["error"]


def test_process_file_error_in_file_data(file_processor, mock_file_data):
    mock_file_data["error"] = "Some error"
    with patch("pathlib.Path.exists", return_value=False):
        result = file_processor.process_file(mock_file_data)
        assert "error" in result


def test_process_files_parallel(file_processor, mock_file_data):
    mock_file_data_list = [mock_file_data.copy() for _ in range(5)]

    def mock_process(data):
        return {"text": f"Processed {data['name']}", "metadata": data}

    with patch.object(file_processor, "process_file", side_effect=mock_process):
        results = file_processor.process_files(mock_file_data_list)

    assert len(results) == 5
    assert all("text" in result for result in results)


def test_process_markdown(file_processor):
    file_path = Path("/tmp/example.md")
    file_data = {"name": "example.md", "path": "/tmp/example.md"}
    with patch("pathlib.Path.read_text", return_value="# Markdown Title"):
        result = file_processor.process_markdown(file_path, file_data)
        assert result["metadata"]["format"] == "markdown"


def test_process_json(file_processor):
    file_path = Path("/tmp/example.json")
    file_data = {"name": "example.json", "path": "/tmp/example.json"}
    with patch("pathlib.Path.read_text", return_value='{"key": "value"}'):
        result = file_processor.process_json(file_path, file_data)
        assert result["metadata"]["format"] == "json"
        assert isinstance(result["structured_data"], dict)


def test_process_notebook(file_processor):
    file_path = Path("/tmp/example.ipynb")
    file_data = {"name": "example.ipynb", "path": "/tmp/example.ipynb"}
    notebook_content = {
        "cells": [
            {"cell_type": "markdown", "source": ["# Markdown Cell"]},
            {"cell_type": "code", "source": ["print('Hello, world!')"]},
        ]
    }
    with patch("pathlib.Path.read_text", return_value=json.dumps(notebook_content)):
        result = file_processor.process_notebook(file_path, file_data)
        assert result["metadata"]["format"] == "notebook"
        assert "cells" in result


def test_process_pdf(file_processor):
    file_path = Path("/tmp/example.pdf")
    file_data = {"name": "example.pdf", "path": "/tmp/example.pdf"}
    with patch("pathlib.Path.exists", return_value=True):
        result = file_processor.process_pdf(file_path, file_data)
        assert result["metadata"]["format"] == "pdf"
