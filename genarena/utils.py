# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Utility functions for genarena."""

import os
import re
import threading
from datetime import datetime


# Thread lock for directory creation
_dir_lock = threading.Lock()


def sanitize_name(name: str) -> str:
    """
    Sanitize a model name for use in file paths.

    Replaces special characters (/, \\, :, etc.) with underscores.

    Args:
        name: The original name (e.g., model path like "org/model-name")

    Returns:
        Sanitized name safe for file system use
    """
    # Replace common path separators and special characters
    sanitized = re.sub(r'[/\\:*?"<>|]', '_', name)
    # Replace multiple underscores with single one
    sanitized = re.sub(r'_+', '_', sanitized)
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    return sanitized


def ensure_dir(path: str) -> None:
    """
    Create a directory if it doesn't exist (thread-safe).

    Args:
        path: Directory path to create
    """
    with _dir_lock:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    """
    Generate a timestamp string in YYYYMMDD_HHMM format.

    Returns:
        Timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


def get_sorted_model_pair(model_a: str, model_b: str) -> tuple[str, str, bool]:
    """
    Sort a model pair alphabetically to ensure consistent file naming.

    This ensures that battles between model_a and model_b always use
    the same log file regardless of which model is passed as first argument.

    Args:
        model_a: First model name
        model_b: Second model name

    Returns:
        Tuple of (sorted_first, sorted_second, swapped) where swapped is True
        if the order was changed
    """
    if model_a <= model_b:
        return model_a, model_b, False
    else:
        return model_b, model_a, True


def get_battle_log_filename(model_a: str, model_b: str) -> str:
    """
    Generate a consistent log filename for a model pair.

    Args:
        model_a: First model name
        model_b: Second model name

    Returns:
        Filename in format "<sorted_model_a>_vs_<sorted_model_b>.jsonl"
    """
    first, second, _ = get_sorted_model_pair(model_a, model_b)
    return f"{sanitize_name(first)}_vs_{sanitize_name(second)}.jsonl"


def iso_timestamp() -> str:
    """
    Generate an ISO 8601 timestamp string.

    Returns:
        ISO format timestamp string (e.g., "2026-01-16T10:30:00Z")
    """
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
