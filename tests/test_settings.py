import os
import logging
from pathlib import Path
import pytest

from config.settings import (
    APP_DIR,
    CACHE_DIR,
    LOG_DIR,
    CONFIG_DIR,
    GITHUB_API_URL,
    GITHUB_MAX_RETRIES,
    GITHUB_TIMEOUT,
    GITHUB_DEFAULT_BRANCH,
    GITHUB_DOWNLOAD_RETRIES,
    RELEVANT_FOLDERS,
    IGNORED_DIRS,
    MAX_FILE_SIZE_MB,
    TEXT_FILE_EXTENSIONS,
    HF_DATASET_TEMPLATE,
    HF_DEFAULT_REPO_TYPE,
    LOG_FORMAT,
    LOG_LEVEL,
    LOG_FILE,
    MAX_LOG_SIZE,
    LOG_BACKUP_COUNT,
    PARALLEL_MAX_WORKERS,
    PARALLEL_CHUNK_SIZE,
    ASYNC_MAX_WORKERS,
)


def test_directories_exist():
    for directory in [APP_DIR, CACHE_DIR, LOG_DIR, CONFIG_DIR]:
        assert directory.exists() and directory.is_dir()


def test_github_settings():
    assert GITHUB_API_URL == "https://api.github.com"
    assert GITHUB_MAX_RETRIES == 3
    assert GITHUB_TIMEOUT == 30
    assert GITHUB_DEFAULT_BRANCH == "main"
    assert GITHUB_DOWNLOAD_RETRIES == 5


def test_repository_content_settings():
    assert "docs" in RELEVANT_FOLDERS
    assert ".git" in IGNORED_DIRS
    assert MAX_FILE_SIZE_MB == 10
    assert ".py" in TEXT_FILE_EXTENSIONS


def test_hf_dataset_template():
    assert HF_DATASET_TEMPLATE["metadata"]["creator"] == "github_hf_dataset_creator"
    assert HF_DATASET_TEMPLATE["config"]["features"]["text"] == "string"
    assert HF_DEFAULT_REPO_TYPE == "dataset"


def test_logging_settings():
    assert LOG_FORMAT == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    assert LOG_LEVEL == logging.INFO
    assert LOG_FILE == LOG_DIR / "app.log"
    assert MAX_LOG_SIZE == 10 * 1024 * 1024
    assert LOG_BACKUP_COUNT == 3


def test_parallel_processing_settings():
    assert PARALLEL_CHUNK_SIZE == 10
    assert PARALLEL_MAX_WORKERS is None
    assert ASYNC_MAX_WORKERS is None
