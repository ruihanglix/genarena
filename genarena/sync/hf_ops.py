# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Huggingface operations module for GenArena.

This module provides functionality for uploading and downloading
arena data to/from Huggingface Dataset repositories.
"""

import logging
import os
import time
import functools
from typing import Any, Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

# Type variable for retry decorator
T = TypeVar("T")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_RETRY_BACKOFF = 2.0  # Exponential backoff multiplier


def retry_on_failure(
    max_retries: int = DEFAULT_MAX_RETRIES,
    delay: float = DEFAULT_RETRY_DELAY,
    backoff: float = DEFAULT_RETRY_BACKOFF,
    exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator that retries a function on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_retries + 1} attempts: {e}"
                        )

            # Re-raise the last exception
            raise last_exception  # type: ignore

        return wrapper
    return decorator

# Environment variable for HF token
HF_TOKEN_ENV = "HF_TOKEN"


def get_hf_token() -> Optional[str]:
    """
    Get the Huggingface token from environment variable.

    Returns:
        Token string or None if not set
    """
    return os.environ.get(HF_TOKEN_ENV)


def require_hf_token() -> str:
    """
    Get the Huggingface token, raising an error if not set.

    Returns:
        Token string

    Raises:
        ValueError: If HF_TOKEN environment variable is not set
    """
    token = get_hf_token()
    if not token:
        raise ValueError(
            f"Environment variable {HF_TOKEN_ENV} is not set. "
            f"Please set it with your Huggingface token: "
            f"export {HF_TOKEN_ENV}='your_token_here'"
        )
    return token


def validate_dataset_repo(repo_id: str, token: Optional[str] = None) -> tuple[bool, str]:
    """
    Validate that a repository exists and is a Dataset type.

    Args:
        repo_id: Repository ID (e.g., "username/repo-name")
        token: Huggingface token (optional for public repos)

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError

        api = HfApi(token=token)

        try:
            repo_info = api.repo_info(repo_id=repo_id, repo_type="dataset")
            return True, f"Valid Dataset repository: {repo_id}"
        except RepositoryNotFoundError:
            # Try to check if it exists as a different type
            try:
                # Check if it's a model repo
                api.repo_info(repo_id=repo_id, repo_type="model")
                return False, (
                    f"Repository '{repo_id}' exists but is a Model repository, not a Dataset. "
                    f"Please create a Dataset repository on Huggingface."
                )
            except RepositoryNotFoundError:
                pass

            try:
                # Check if it's a space repo
                api.repo_info(repo_id=repo_id, repo_type="space")
                return False, (
                    f"Repository '{repo_id}' exists but is a Space repository, not a Dataset. "
                    f"Please create a Dataset repository on Huggingface."
                )
            except RepositoryNotFoundError:
                pass

            return False, (
                f"Repository '{repo_id}' does not exist. "
                f"Please create a Dataset repository on Huggingface first: "
                f"https://huggingface.co/new-dataset"
            )

    except ImportError:
        return False, (
            "huggingface_hub package is not installed. "
            "Please install it with: pip install huggingface_hub"
        )
    except Exception as e:
        return False, f"Error validating repository: {e}"


def list_repo_files(
    repo_id: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> tuple[bool, list[str], str]:
    """
    List all files in a Huggingface Dataset repository.

    Args:
        repo_id: Repository ID
        token: Huggingface token (optional for public repos)
        revision: Branch/revision name

    Returns:
        Tuple of (success, file_list, message)
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )

        return True, list(files), f"Found {len(files)} files in {repo_id}"

    except Exception as e:
        return False, [], f"Error listing repository files: {e}"


def get_repo_file_info(
    repo_id: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> tuple[bool, list[dict], str]:
    """
    Get detailed file information from a Huggingface Dataset repository.

    Args:
        repo_id: Repository ID
        token: Huggingface token (optional for public repos)
        revision: Branch/revision name

    Returns:
        Tuple of (success, file_info_list, message)
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)

        repo_info = api.repo_info(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            files_metadata=True,
        )

        file_infos = []
        if repo_info.siblings:
            for sibling in repo_info.siblings:
                file_infos.append({
                    "path": sibling.rfilename,
                    "size": sibling.size,
                    "blob_id": sibling.blob_id,
                })

        return True, file_infos, f"Found {len(file_infos)} files in {repo_id}"

    except Exception as e:
        return False, [], f"Error getting repository info: {e}"


def upload_file(
    repo_id: str,
    local_path: str,
    remote_path: str,
    token: str,
    commit_message: Optional[str] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    repo_type: str = "dataset",
) -> tuple[bool, str]:
    """
    Upload a single file to a Huggingface repository with retry support.

    Args:
        repo_id: Repository ID
        local_path: Local file path
        remote_path: Path in the repository
        token: Huggingface token
        commit_message: Optional commit message
        max_retries: Maximum number of retry attempts on failure
        repo_type: Repository type ("dataset", "model", or "space")

    Returns:
        Tuple of (success, message)
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    if not commit_message:
        commit_message = f"Upload {remote_path}"

    @retry_on_failure(
        max_retries=max_retries,
        delay=DEFAULT_RETRY_DELAY,
        backoff=DEFAULT_RETRY_BACKOFF,
    )
    def _do_upload() -> None:
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
        )

    try:
        _do_upload()
        return True, f"Uploaded {remote_path}"
    except Exception as e:
        return False, f"Error uploading file: {e}"


def upload_files_batch(
    repo_id: str,
    file_mappings: list[tuple[str, str]],
    token: str,
    commit_message: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Upload multiple files in a single commit.

    Args:
        repo_id: Repository ID
        file_mappings: List of (local_path, remote_path) tuples
        token: Huggingface token
        commit_message: Optional commit message

    Returns:
        Tuple of (success, message)
    """
    try:
        from huggingface_hub import HfApi, CommitOperationAdd

        api = HfApi(token=token)

        if not commit_message:
            commit_message = f"Upload {len(file_mappings)} files"

        operations = [
            CommitOperationAdd(
                path_in_repo=remote_path,
                path_or_fileobj=local_path,
            )
            for local_path, remote_path in file_mappings
        ]

        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
        )

        return True, f"Uploaded {len(file_mappings)} files"

    except Exception as e:
        return False, f"Error uploading files: {e}"


def download_file(
    repo_id: str,
    remote_path: str,
    local_path: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> tuple[bool, str]:
    """
    Download a single file from a Huggingface Dataset repository.

    Args:
        repo_id: Repository ID
        remote_path: Path in the repository
        local_path: Local file path to save to
        token: Huggingface token (optional for public repos)
        revision: Branch/revision name

    Returns:
        Tuple of (success, message)
    """
    try:
        from huggingface_hub import hf_hub_download

        # Ensure local directory exists
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download to a temp location first, then move
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=remote_path,
            repo_type="dataset",
            revision=revision,
            token=token,
            local_dir=os.path.dirname(local_path),
            local_dir_use_symlinks=False,
        )

        # If downloaded to a different path, copy to expected location
        if downloaded_path != local_path:
            import shutil
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            shutil.copy2(downloaded_path, local_path)

        return True, f"Downloaded {remote_path}"

    except Exception as e:
        return False, f"Error downloading file: {e}"


def check_file_exists(
    repo_id: str,
    remote_path: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> bool:
    """
    Check if a file exists in the repository.

    Args:
        repo_id: Repository ID
        remote_path: Path in the repository
        token: Huggingface token (optional for public repos)
        revision: Branch/revision name

    Returns:
        True if file exists
    """
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )

        return remote_path in files

    except Exception:
        return False


def format_file_size(size_bytes: Optional[int]) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Human-readable size string
    """
    if size_bytes is None:
        return "Unknown"

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0

    return f"{size_bytes:.1f} PB"


# =============================================================================
# High-level operations
# =============================================================================

def upload_arena_data(
    arena_dir: str,
    repo_id: str,
    subsets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    experiments: Optional[list[str]] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[bool, str]:
    """
    Upload arena data to a Huggingface Dataset repository.

    This function:
    1. Validates the repository exists and is a Dataset type
    2. Collects files to upload based on filters
    3. Packs directories into ZIP files
    4. Uploads files with progress indication and retry on failure

    Supports resume upload: by default (overwrite=False), already uploaded files
    are automatically skipped, enabling resumable uploads after connection failures.

    Args:
        arena_dir: Path to the arena directory
        repo_id: Huggingface repository ID
        subsets: List of subsets to upload (None = all)
        models: List of models to upload (None = all)
        experiments: List of experiments (exp_name) to upload (None = all)
        overwrite: If True, overwrite existing files; if False, skip existing (resume mode)
        show_progress: If True, show progress bar
        max_retries: Maximum number of retry attempts per file on failure

    Returns:
        Tuple of (success, message)
    """
    from genarena.sync.packer import (
        collect_upload_tasks,
        pack_model_dir,
        pack_exp_dir,
        TempPackingContext,
        TaskType,
    )

    # Get token
    try:
        token = require_hf_token()
    except ValueError as e:
        return False, str(e)

    # Validate repository
    valid, msg = validate_dataset_repo(repo_id, token)
    if not valid:
        return False, msg

    logger.info(f"Uploading to repository: {repo_id}")

    # Collect upload tasks
    tasks = collect_upload_tasks(arena_dir, subsets, models, experiments)
    if not tasks:
        return False, "No files to upload. Check arena_dir and filters."

    logger.info(f"Found {len(tasks)} items to scan")

    # Get existing files in repo (for overwrite check)
    existing_files = set()
    if not overwrite:
        logger.info("Checking existing files in remote repository...")
        success, files, _ = list_repo_files(repo_id, token)
        if success:
            existing_files = set(files)
            logger.info(f"Found {len(existing_files)} files in remote repository")

    # Pre-scan: categorize tasks into to_upload and to_skip
    to_upload = []
    to_skip = []
    for task in tasks:
        if not overwrite and task.remote_path in existing_files:
            to_skip.append(task)
        else:
            to_upload.append(task)

    # Display scan summary
    logger.info(f"Scan summary: {len(to_upload)} to upload, {len(to_skip)} already exist (will skip)")

    if to_skip:
        logger.info("Already uploaded (will skip):")
        for task in to_skip[:10]:
            logger.info(f"  ✓ {task.remote_path}")
        if len(to_skip) > 10:
            logger.info(f"  ... and {len(to_skip) - 10} more")

    if to_upload:
        logger.info("To be uploaded:")
        for task in to_upload[:10]:
            logger.info(f"  → {task.remote_path}")
        if len(to_upload) > 10:
            logger.info(f"  ... and {len(to_upload) - 10} more")

    if not to_upload:
        return True, f"All {len(to_skip)} files already exist in repository. Nothing to upload."

    # Process tasks (only those that need uploading)
    uploaded = 0
    skipped = len(to_skip)  # Pre-count skipped
    failed = 0
    errors = []

    # Setup progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            to_upload = tqdm(to_upload, desc="Uploading", unit="file")
        except ImportError:
            pass

    with TempPackingContext() as ctx:
        for task in to_upload:
            try:
                if task.task_type == TaskType.MODEL_ZIP:
                    # Pack model directory
                    zip_path = ctx.get_temp_zip_path(task.remote_path)
                    success, msg = pack_model_dir(task.local_path, zip_path)
                    if not success:
                        errors.append(f"{task.name}: {msg}")
                        failed += 1
                        continue

                    # Upload ZIP with retry
                    success, msg = upload_file(
                        repo_id, zip_path, task.remote_path, token,
                        commit_message=f"[genarena] Upload model: {task.subset}/{task.name}",
                        max_retries=max_retries,
                    )

                elif task.task_type == TaskType.EXP_ZIP:
                    # Pack experiment directory
                    zip_path = ctx.get_temp_zip_path(task.remote_path)
                    success, msg = pack_exp_dir(task.local_path, zip_path)
                    if not success:
                        errors.append(f"{task.name}: {msg}")
                        failed += 1
                        continue

                    # Upload ZIP with retry
                    success, msg = upload_file(
                        repo_id, zip_path, task.remote_path, token,
                        commit_message=f"[genarena] Upload experiment: {task.subset}/{task.name}",
                        max_retries=max_retries,
                    )

                elif task.task_type == TaskType.SMALL_FILE:
                    # Upload small file directly with retry
                    success, msg = upload_file(
                        repo_id, task.local_path, task.remote_path, token,
                        commit_message=f"[genarena] Upload {task.name}",
                        max_retries=max_retries,
                    )

                else:
                    success = False
                    msg = f"Unknown task type: {task.task_type}"

                if success:
                    uploaded += 1
                    logger.debug(f"Uploaded: {task.remote_path}")
                else:
                    errors.append(f"{task.name}: {msg}")
                    failed += 1

            except Exception as e:
                errors.append(f"{task.name}: {e}")
                failed += 1

    # Summary
    summary = f"Uploaded: {uploaded}, Skipped: {skipped}, Failed: {failed}"
    if errors:
        summary += f"\nErrors:\n" + "\n".join(f"  - {e}" for e in errors[:5])
        if len(errors) > 5:
            summary += f"\n  ... and {len(errors) - 5} more errors"

    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    summary += f"\n\nRepository URL: {repo_url}"

    success = failed == 0 or uploaded > 0
    return success, summary


def pull_arena_data(
    arena_dir: str,
    repo_id: str,
    subsets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    experiments: Optional[list[str]] = None,
    revision: str = "main",
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[bool, str]:
    """
    Pull arena data from a Huggingface Dataset repository.

    This function:
    1. Validates the repository exists and is a Dataset type
    2. Lists files in the repository
    3. Filters based on subsets/models
    4. Downloads and unpacks ZIP files

    Args:
        arena_dir: Path to the local arena directory
        repo_id: Huggingface repository ID
        subsets: List of subsets to download (None = all)
        models: List of models to download (None = all)
        experiments: List of experiments (exp_name) to download (None = all)
        revision: Branch/revision to download from
        overwrite: If True, overwrite existing files
        show_progress: If True, show progress bar

    Returns:
        Tuple of (success, message)
    """
    import tempfile
    import shutil
    from genarena.sync.packer import (
        collect_download_tasks,
        unpack_zip,
        TaskType,
    )

    # Get token (optional for public repos)
    token = get_hf_token()

    # Validate repository
    valid, msg = validate_dataset_repo(repo_id, token)
    if not valid:
        return False, msg

    logger.info(f"Pulling from repository: {repo_id} (revision: {revision})")

    # List files in repository
    success, repo_files, msg = list_repo_files(repo_id, token, revision)
    if not success:
        return False, msg

    if not repo_files:
        return False, "Repository is empty"

    # Collect download tasks
    tasks = collect_download_tasks(repo_files, arena_dir, subsets, models, experiments)
    if not tasks:
        return False, "No matching files to download. Check filters."

    logger.info(f"Found {len(tasks)} items to download")

    # Process tasks
    downloaded = 0
    skipped = 0
    failed = 0
    errors = []

    # Setup progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            tasks = tqdm(tasks, desc="Downloading", unit="file")
        except ImportError:
            pass

    # Create temp directory for downloads
    temp_dir = tempfile.mkdtemp(prefix="genarena_pull_")

    try:
        for task in tasks:
            try:
                if task.task_type in (TaskType.MODEL_ZIP, TaskType.EXP_ZIP):
                    # Download ZIP to temp location
                    temp_zip = os.path.join(temp_dir, os.path.basename(task.remote_path))
                    success, msg = download_file(
                        repo_id, task.remote_path, temp_zip, token, revision
                    )

                    if not success:
                        errors.append(f"{task.name}: {msg}")
                        failed += 1
                        continue

                    # Unpack ZIP
                    success, msg = unpack_zip(temp_zip, task.local_path, overwrite)
                    if not success:
                        errors.append(f"{task.name}: {msg}")
                        failed += 1
                        continue

                    downloaded += 1
                    logger.debug(f"Downloaded and unpacked: {task.remote_path}")

                elif task.task_type == TaskType.SMALL_FILE:
                    # Check if file exists and skip if not overwriting
                    if os.path.exists(task.local_path) and not overwrite:
                        logger.debug(f"Skipping existing: {task.local_path}")
                        skipped += 1
                        continue

                    # Download file directly
                    success, msg = download_file(
                        repo_id, task.remote_path, task.local_path, token, revision
                    )

                    if success:
                        downloaded += 1
                        logger.debug(f"Downloaded: {task.remote_path}")
                    else:
                        errors.append(f"{task.name}: {msg}")
                        failed += 1

            except Exception as e:
                errors.append(f"{task.name}: {e}")
                failed += 1

    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Summary
    summary = f"Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}"
    if errors:
        summary += f"\nErrors:\n" + "\n".join(f"  - {e}" for e in errors[:5])
        if len(errors) > 5:
            summary += f"\n  ... and {len(errors) - 5} more errors"

    success = failed == 0 or downloaded > 0
    return success, summary


def list_repo_contents(
    repo_id: str,
    revision: str = "main",
) -> tuple[bool, str]:
    """
    List contents of a Huggingface Dataset repository.

    Displays files organized by subset with size information.

    Args:
        repo_id: Huggingface repository ID
        revision: Branch/revision name

    Returns:
        Tuple of (success, formatted_output)
    """
    # Get token (optional for public repos)
    token = get_hf_token()

    # Validate repository
    valid, msg = validate_dataset_repo(repo_id, token)
    if not valid:
        return False, msg

    # Get file info
    success, file_infos, msg = get_repo_file_info(repo_id, token, revision)
    if not success:
        return False, msg

    if not file_infos:
        return True, f"Repository '{repo_id}' is empty"

    # Organize by subset
    subsets: dict[str, list[dict]] = {}
    other_files: list[dict] = []

    for info in file_infos:
        path = info["path"]
        parts = path.split("/")

        if len(parts) >= 2:
            subset = parts[0]
            if subset not in subsets:
                subsets[subset] = []
            subsets[subset].append(info)
        else:
            other_files.append(info)

    # Format output
    lines = [
        f"Repository: {repo_id}",
        f"Revision: {revision}",
        f"Total files: {len(file_infos)}",
        "",
    ]

    total_size = sum(f.get("size", 0) or 0 for f in file_infos)
    lines.append(f"Total size: {format_file_size(total_size)}")
    lines.append("")

    for subset in sorted(subsets.keys()):
        files = subsets[subset]
        subset_size = sum(f.get("size", 0) or 0 for f in files)

        lines.append(f"=== {subset} ({len(files)} files, {format_file_size(subset_size)}) ===")

        # Organize by type
        models = []
        experiments = []
        other = []

        for f in files:
            path = f["path"]
            if "/models/" in path:
                models.append(f)
            elif "/pk_logs/" in path:
                experiments.append(f)
            else:
                other.append(f)

        if models:
            lines.append("  Models:")
            for f in sorted(models, key=lambda x: x["path"]):
                size = format_file_size(f.get("size"))
                name = os.path.basename(f["path"])
                lines.append(f"    - {name} ({size})")

        if experiments:
            lines.append("  Experiments:")
            for f in sorted(experiments, key=lambda x: x["path"]):
                size = format_file_size(f.get("size"))
                name = os.path.basename(f["path"])
                lines.append(f"    - {name} ({size})")

        if other:
            lines.append("  Other:")
            for f in sorted(other, key=lambda x: x["path"]):
                size = format_file_size(f.get("size"))
                name = f["path"].split("/", 1)[1] if "/" in f["path"] else f["path"]
                lines.append(f"    - {name} ({size})")

        lines.append("")

    if other_files:
        lines.append("=== Other files ===")
        for f in sorted(other_files, key=lambda x: x["path"]):
            size = format_file_size(f.get("size"))
            lines.append(f"  - {f['path']} ({size})")

    return True, "\n".join(lines)
