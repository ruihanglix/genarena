# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Auto commit module for GenArena.

This module provides automatic commit and push functionality
that is triggered after command execution.
"""

import logging
from typing import Callable, TypeVar

from genarena.sync.git_ops import (
    is_git_initialized,
    has_uncommitted_changes,
    git_commit,
    git_push,
    git_remote_get_url,
)

logger = logging.getLogger(__name__)

# Type variable for generic decorator
T = TypeVar("T")


def auto_commit_and_push(arena_dir: str, command_name: str) -> None:
    """
    Automatically commit and push changes after a command execution.

    This function is designed to be called after commands that modify
    arena_dir content (e.g., run, merge, delete). It silently skips
    if Git is not initialized, and only warns on failure without
    interrupting the main command flow.

    Args:
        arena_dir: Path to the arena directory
        command_name: Name of the command that triggered this auto-commit
    """
    # Skip silently if Git is not initialized
    if not is_git_initialized(arena_dir):
        return

    # Check if there are uncommitted changes
    if not has_uncommitted_changes(arena_dir):
        logger.debug(f"No changes to commit after {command_name}")
        return

    # Try to commit
    try:
        success, msg = git_commit(arena_dir, command_name=command_name)
        if success:
            if "Nothing to commit" not in msg:
                logger.info(f"Auto-committed changes: {msg}")
        else:
            logger.warning(f"Auto-commit failed: {msg}")
            return
    except Exception as e:
        logger.warning(f"Auto-commit failed with exception: {e}")
        return

    # Check if remote is configured and try to push
    remote_url = git_remote_get_url(arena_dir)
    if not remote_url:
        logger.debug("No remote configured, skipping auto-push")
        return

    # Try to push
    try:
        success, msg = git_push(arena_dir)
        if success:
            logger.info(f"Auto-pushed changes: {msg}")
        else:
            logger.warning(f"Auto-push failed: {msg}")
    except Exception as e:
        logger.warning(f"Auto-push failed with exception: {e}")


def with_auto_commit(command_name: str):
    """
    Decorator that adds auto-commit functionality to command functions.

    The decorated function must have 'arena_dir' as an argument or
    in its args namespace.

    Args:
        command_name: Name of the command for commit message

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., int]) -> Callable[..., int]:
        def wrapper(*args, **kwargs) -> int:
            # Execute the original command
            result = func(*args, **kwargs)

            # Only auto-commit if the command succeeded (return code 0)
            if result == 0:
                # Try to get arena_dir from args
                arena_dir = None

                # Check kwargs first
                if "arena_dir" in kwargs:
                    arena_dir = kwargs["arena_dir"]
                # Check if first arg is argparse.Namespace
                elif args and hasattr(args[0], "arena_dir"):
                    arena_dir = args[0].arena_dir

                if arena_dir:
                    auto_commit_and_push(arena_dir, command_name)

            return result

        return wrapper

    return decorator
