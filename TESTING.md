# Testing Guide

This document provides guidelines for writing and running tests for the Dataset Creator project.

## Running Tests

To run all tests:

```bash
pytest
```

To run specific test files:

```bash
pytest tests/test_file_processor.py
```

To run specific test functions:

```bash
pytest tests/test_file_processor.py::test_process_markdown
```

## Test Coverage

To generate test coverage reports:

```bash
pytest --cov=./ --cov-report=html
```

This will create a directory called `htmlcov` containing an HTML report of your test coverage. Open `htmlcov/index.html` in your browser to view the report.

## Types of Tests

### Unit Tests

These test individual functions and classes in isolation. Unit tests should be fast and cover the basic functionality of each component.

### Integration Tests

These test the interaction between multiple components. Integration tests verify that different parts of the system work together correctly.

### End-to-End Tests

These test the entire application from start to finish, including user interface interactions and external dependencies (like GitHub API and Hugging Face API).

## Writing Tests

- Each test should have a clear name describing what it's testing
- Use fixtures for setting up common test data
- Mock external dependencies to isolate your tests
- Test both success and failure cases

### Example Test Structure

```python
import pytest
from unittest.mock import MagicMock, patch

def test_function_success_case():
    # Setup
    ...
    
    # Exercise
    ...
    
    # Verify
    ...

def test_function_failure_case():
    # Setup
    ...
    
    # Exercise
    ...
    
    # Verify
    ...
```

## Continuous Integration

Our CI pipeline runs tests automatically on every pull request and push to main branches. The pipeline:

1. Runs unit tests on Python 3.9, 3.10, and 3.12
2. Checks code coverage
3. Performs code quality checks (linting, formatting, etc.)

All tests must pass and maintain the code coverage threshold for pull requests to be merged.
