# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Git operations module for GenArena.

This module provides Git version control functionality for arena data,
including initialization, commit, remote configuration, and push operations.
"""

import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Patterns to exclude from Git tracking (models directories contain large image files)
GITIGNORE_PATTERNS = [
    "# GenArena: Exclude model output images (large files)",
    "*/models/",
    "",
    "# Python cache",
    "__pycache__/",
    "*.pyc",
    "",
    "# OS files",
    ".DS_Store",
    "Thumbs.db",
]


def _run_git_command(
    arena_dir: str,
    args: list,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess:
    """
    Run a git command in the arena directory.

    Args:
        arena_dir: Path to the arena directory
        args: Git command arguments (without 'git' prefix)
        check: If True, raise exception on non-zero exit code
        capture_output: If True, capture stdout and stderr

    Returns:
        CompletedProcess instance

    Raises:
        subprocess.CalledProcessError: If check=True and command fails
    """
    cmd = ["git"] + args
    return subprocess.run(
        cmd,
        cwd=arena_dir,
        check=check,
        capture_output=capture_output,
        text=True,
    )


def is_git_initialized(arena_dir: str) -> bool:
    """
    Check if the arena directory is a Git repository.

    Args:
        arena_dir: Path to the arena directory

    Returns:
        True if Git is initialized, False otherwise
    """
    git_dir = os.path.join(arena_dir, ".git")
    return os.path.isdir(git_dir)


def git_init(arena_dir: str) -> tuple[bool, str]:
    """
    Initialize a Git repository in the arena directory.

    Args:
        arena_dir: Path to the arena directory

    Returns:
        Tuple of (success, message)
    """
    if is_git_initialized(arena_dir):
        return True, "Git repository already initialized"

    # Ensure directory exists
    os.makedirs(arena_dir, exist_ok=True)

    try:
        result = _run_git_command(arena_dir, ["init"])
        logger.info(f"Initialized Git repository in {arena_dir}")

        # Ensure .gitignore is set up
        ensure_gitignore(arena_dir)

        return True, "Git repository initialized successfully"
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to initialize Git repository: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg


def ensure_gitignore(arena_dir: str) -> tuple[bool, str]:
    """
    Create or update .gitignore file to exclude models directories.

    Args:
        arena_dir: Path to the arena directory

    Returns:
        Tuple of (success, message)
    """
    gitignore_path = os.path.join(arena_dir, ".gitignore")

    existing_content = ""
    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r", encoding="utf-8") as f:
            existing_content = f.read()

    # Check if the key pattern already exists
    key_pattern = "*/models/"
    if key_pattern in existing_content:
        return True, ".gitignore already contains required patterns"

    # Append patterns to existing content
    new_content = existing_content
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"

    if new_content:
        new_content += "\n"

    new_content += "\n".join(GITIGNORE_PATTERNS)

    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    logger.info(f"Updated .gitignore in {arena_dir}")
    return True, ".gitignore updated successfully"


def git_add_all(arena_dir: str) -> tuple[bool, str]:
    """
    Stage all changes in the arena directory (respecting .gitignore).

    Args:
        arena_dir: Path to the arena directory

    Returns:
        Tuple of (success, message)
    """
    if not is_git_initialized(arena_dir):
        return False, "Git repository not initialized"

    try:
        _run_git_command(arena_dir, ["add", "-A"])
        return True, "All changes staged"
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to stage changes: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg


def has_uncommitted_changes(arena_dir: str) -> bool:
    """
    Check if there are uncommitted changes in the repository.

    Args:
        arena_dir: Path to the arena directory

    Returns:
        True if there are uncommitted changes, False otherwise
    """
    if not is_git_initialized(arena_dir):
        return False

    try:
        # Check for staged changes
        result = _run_git_command(arena_dir, ["diff", "--cached", "--quiet"], check=False)
        if result.returncode != 0:
            return True

        # Check for unstaged changes
        result = _run_git_command(arena_dir, ["diff", "--quiet"], check=False)
        if result.returncode != 0:
            return True

        # Check for untracked files (that aren't ignored)
        result = _run_git_command(
            arena_dir,
            ["ls-files", "--others", "--exclude-standard"],
            check=False
        )
        if result.stdout.strip():
            return True

        return False
    except Exception as e:
        logger.warning(f"Error checking for uncommitted changes: {e}")
        return False


def git_commit(
    arena_dir: str,
    message: Optional[str] = None,
    command_name: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Commit staged changes.

    Args:
        arena_dir: Path to the arena directory
        message: Custom commit message (optional)
        command_name: Name of the command that triggered this commit (for auto-commit)

    Returns:
        Tuple of (success, message)
    """
    if not is_git_initialized(arena_dir):
        return False, "Git repository not initialized"

    # Stage all changes first
    success, msg = git_add_all(arena_dir)
    if not success:
        return False, msg

    # Check if there's anything to commit
    result = _run_git_command(arena_dir, ["diff", "--cached", "--quiet"], check=False)
    if result.returncode == 0:
        return True, "Nothing to commit, working tree clean"

    # Generate commit message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if message:
        commit_msg = message
    elif command_name:
        commit_msg = f"[genarena] Auto commit after {command_name} at {timestamp}"
    else:
        commit_msg = f"[genarena] Auto commit at {timestamp}"

    try:
        _run_git_command(arena_dir, ["commit", "-m", commit_msg])
        logger.info(f"Committed changes: {commit_msg}")
        return True, f"Committed: {commit_msg}"
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to commit: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg


def git_remote_get_url(arena_dir: str, remote_name: str = "origin") -> Optional[str]:
    """
    Get the URL of a remote repository.

    Args:
        arena_dir: Path to the arena directory
        remote_name: Name of the remote (default: origin)

    Returns:
        Remote URL or None if not configured
    """
    if not is_git_initialized(arena_dir):
        return None

    try:
        result = _run_git_command(
            arena_dir,
            ["remote", "get-url", remote_name],
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def git_remote_add(
    arena_dir: str,
    url: str,
    remote_name: str = "origin",
    force: bool = False,
) -> tuple[bool, str]:
    """
    Configure a remote repository.

    Args:
        arena_dir: Path to the arena directory
        url: Remote repository URL
        remote_name: Name of the remote (default: origin)
        force: If True, overwrite existing remote URL

    Returns:
        Tuple of (success, message)
    """
    if not is_git_initialized(arena_dir):
        return False, "Git repository not initialized"

    existing_url = git_remote_get_url(arena_dir, remote_name)

    if existing_url:
        if existing_url == url:
            return True, f"Remote '{remote_name}' already configured with this URL"

        if not force:
            return False, (
                f"Remote '{remote_name}' already exists with URL: {existing_url}. "
                f"Use --force to overwrite."
            )

        # Remove existing remote
        try:
            _run_git_command(arena_dir, ["remote", "remove", remote_name])
        except subprocess.CalledProcessError as e:
            return False, f"Failed to remove existing remote: {e.stderr}"

    # Add remote
    try:
        _run_git_command(arena_dir, ["remote", "add", remote_name, url])
        logger.info(f"Added remote '{remote_name}': {url}")
        return True, f"Remote '{remote_name}' configured: {url}"
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to add remote: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg


def git_push(
    arena_dir: str,
    remote_name: str = "origin",
    branch: Optional[str] = None,
    set_upstream: bool = True,
) -> tuple[bool, str]:
    """
    Push commits to the remote repository.

    Args:
        arena_dir: Path to the arena directory
        remote_name: Name of the remote (default: origin)
        branch: Branch name (default: current branch)
        set_upstream: If True, set upstream tracking

    Returns:
        Tuple of (success, message)
    """
    if not is_git_initialized(arena_dir):
        return False, "Git repository not initialized"

    # Check if remote is configured
    remote_url = git_remote_get_url(arena_dir, remote_name)
    if not remote_url:
        return False, f"Remote '{remote_name}' not configured. Use 'genarena git remote --url <url>' first."

    # Get current branch if not specified
    if not branch:
        try:
            result = _run_git_command(arena_dir, ["branch", "--show-current"])
            branch = result.stdout.strip()
            if not branch:
                # Might be on a detached HEAD, try to get default branch
                branch = "main"
        except subprocess.CalledProcessError:
            branch = "main"

    # Push
    try:
        push_args = ["push"]
        if set_upstream:
            push_args.extend(["-u", remote_name, branch])
        else:
            push_args.extend([remote_name, branch])

        _run_git_command(arena_dir, push_args)
        logger.info(f"Pushed to {remote_name}/{branch}")
        return True, f"Pushed to {remote_name}/{branch}"
    except subprocess.CalledProcessError as e:
        error_msg = f"Failed to push: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg


def git_sync(arena_dir: str) -> tuple[bool, str]:
    """
    Commit all changes and push to remote (one-click sync).

    Args:
        arena_dir: Path to the arena directory

    Returns:
        Tuple of (success, message)
    """
    if not is_git_initialized(arena_dir):
        return False, "Git repository not initialized"

    messages = []

    # Commit changes
    success, msg = git_commit(arena_dir)
    messages.append(msg)

    if not success and "Nothing to commit" not in msg:
        return False, msg

    # Push to remote
    success, msg = git_push(arena_dir)
    messages.append(msg)

    if not success:
        # If push fails due to no remote, still return partial success
        if "not configured" in msg:
            return True, f"{messages[0]} (push skipped: {msg})"
        return False, msg

    return True, " | ".join(messages)
