# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Deploy operations for GenArena.

Handles uploading arena data to HuggingFace for Spaces deployment.
Unlike `hf upload`, this uploads images directly (not as ZIP) for CDN access.
Parquet benchmark data is downloaded from rhli/genarena during Docker build.
"""

import logging
import os
from multiprocessing import Pool
from typing import Optional

logger = logging.getLogger(__name__)


# Default multiprocessing settings
DEFAULT_NUM_WORKERS = 16
DEFAULT_WORKER_TIMEOUT = 300  # seconds


def upload_for_deploy(
    arena_dir: str,
    arena_repo: str,
    space_repo: str,
    subsets: Optional[list[str]] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    max_retries: int = 3,
    num_workers: int = DEFAULT_NUM_WORKERS,
    worker_timeout: int = DEFAULT_WORKER_TIMEOUT,
) -> tuple[bool, str]:
    """
    Upload all data needed for HuggingFace Spaces deployment.

    This uploads:
    1. Arena data (pk_logs, models, state.json) to arena_repo (Dataset)
       - Images are uploaded directly (not as ZIP) for CDN access
       - Follows symlinks to upload actual image files
    2. Deploy files (Dockerfile, app.py, README.md) to space_repo

    Note: Parquet benchmark data is NOT uploaded. It is downloaded from
    rhli/genarena during Docker build in the Space.

    Args:
        arena_dir: Local arena directory
        arena_repo: HF Dataset repo for arena data
        space_repo: HF Space repo for deployment
        subsets: Subsets to upload (None = all)
        overwrite: Overwrite existing files
        show_progress: Show progress bar
        max_retries: Max retries per file
        num_workers: Number of parallel workers for upload (default: 16)
        worker_timeout: Timeout in seconds for each worker (default: 300)

    Returns:
        Tuple of (success, message)
    """
    from genarena.sync.hf_ops import (
        require_hf_token,
        validate_dataset_repo,
    )

    # Get token
    try:
        token = require_hf_token()
    except ValueError as e:
        return False, str(e)

    messages = []

    # 1. Upload arena data to Dataset repo (images as individual files, not ZIP)
    logger.info(f"Uploading arena data to {arena_repo}...")
    valid, msg = validate_dataset_repo(arena_repo, token)
    if not valid:
        return False, f"Arena repo validation failed: {msg}"

    success, msg = upload_arena_data_for_cdn(
        arena_dir=arena_dir,
        repo_id=arena_repo,
        subsets=subsets,
        overwrite=overwrite,
        show_progress=show_progress,
        token=token,
        num_workers=num_workers,
        worker_timeout=worker_timeout,
    )
    if not success:
        return False, f"Arena upload failed: {msg}"
    messages.append(f"Arena data: {msg}")

    # 2. Upload deploy files to Space repo
    logger.info(f"Uploading deploy files to {space_repo}...")
    success, msg = upload_deploy_files(
        space_repo=space_repo,
        overwrite=overwrite,
        token=token,
    )
    if not success:
        return False, f"Deploy files upload failed: {msg}"
    messages.append(f"Deploy files: {msg}")

    return True, "\n".join(messages)


def collect_files_follow_symlinks(
    base_dir: str,
    path_prefix: str = "",
) -> list[tuple[str, str]]:
    """
    Collect all files under base_dir, following symlinks.

    Args:
        base_dir: Directory to scan
        path_prefix: Prefix for remote paths

    Returns:
        List of (local_path, remote_path) tuples
    """
    files = []

    if not os.path.isdir(base_dir):
        return files

    # Use os.walk with followlinks=True to traverse symlinks
    for root, dirs, filenames in os.walk(base_dir, followlinks=True):
        # Skip hidden directories and special directories
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != "raw_outputs"]

        rel_root = os.path.relpath(root, base_dir)
        if rel_root == ".":
            rel_root = ""

        for filename in filenames:
            if filename.startswith("."):
                continue

            local_path = os.path.join(root, filename)

            # Build remote path
            if rel_root:
                remote_path = f"{path_prefix}/{rel_root}/{filename}" if path_prefix else f"{rel_root}/{filename}"
            else:
                remote_path = f"{path_prefix}/{filename}" if path_prefix else filename

            # Normalize path separators
            remote_path = remote_path.replace("\\", "/")

            files.append((local_path, remote_path))

    return files


def _upload_batch_worker(args: tuple) -> tuple[int, int]:
    """
    Worker function for uploading a single batch.

    Args:
        args: Tuple of (batch_index, batch, repo_id, token, total_batches, max_retries)

    Returns:
        Tuple of (uploaded_count, failed_count)
    """
    from huggingface_hub import HfApi, CommitOperationAdd

    batch_index, batch, repo_id, token, total_batches, max_retries = args

    api = HfApi(token=token)

    operations = []
    failed_read = 0
    for local_path, remote_path in batch:
        try:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=remote_path,
                    path_or_fileobj=local_path,
                )
            )
        except Exception as e:
            logger.warning(f"Failed to read {local_path}: {e}")
            failed_read += 1

    if not operations:
        return 0, failed_read

    # Try to commit batch with retries
    for attempt in range(max_retries):
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                commit_message=f"[genarena deploy] Upload batch {batch_index + 1}/{total_batches}",
            )
            return len(operations), failed_read
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Batch {batch_index + 1} failed (attempt {attempt + 1}), retrying: {e}")
            else:
                logger.error(f"Batch {batch_index + 1} failed after {max_retries} attempts: {e}")
                return 0, len(operations) + failed_read

    return 0, len(operations) + failed_read


def upload_arena_data_for_cdn(
    arena_dir: str,
    repo_id: str,
    subsets: Optional[list[str]] = None,
    overwrite: bool = False,
    show_progress: bool = True,
    token: Optional[str] = None,
    num_workers: int = DEFAULT_NUM_WORKERS,
    worker_timeout: int = DEFAULT_WORKER_TIMEOUT,
) -> tuple[bool, str]:
    """
    Upload arena data with images as individual files (not ZIP) for CDN access.

    This function follows symlinks to upload actual image files.
    Models directory often contains symlinks to external image directories.

    Directory structure uploaded:
        {subset}/models/{exp_name}/{model}/{index}.png  (individual images)
        {subset}/pk_logs/{exp_name}/*.jsonl             (battle logs)
        {subset}/arena/state.json                       (ELO state)

    Args:
        arena_dir: Path to the arena directory
        repo_id: HuggingFace repository ID
        subsets: List of subsets to upload (None = all)
        overwrite: If True, overwrite existing files
        show_progress: If True, show progress bar
        token: HuggingFace token
        num_workers: Number of parallel workers for upload (default: 16)
        worker_timeout: Timeout in seconds for each worker (default: 300)

    Returns:
        Tuple of (success, message)
    """
    from huggingface_hub import HfApi

    if token is None:
        from genarena.sync.hf_ops import require_hf_token
        token = require_hf_token()

    api = HfApi(token=token)

    # Validate arena directory
    if not os.path.isdir(arena_dir):
        return False, f"Arena directory not found: {arena_dir}"

    # Discover subsets
    available_subsets = [
        d for d in os.listdir(arena_dir)
        if os.path.isdir(os.path.join(arena_dir, d)) and not d.startswith(".")
    ]

    if subsets:
        target_subsets = [s for s in subsets if s in available_subsets]
    else:
        target_subsets = available_subsets

    if not target_subsets:
        return False, "No subsets found to upload"

    logger.info(f"Target subsets: {target_subsets}")

    # Collect all files to upload (following symlinks)
    all_files: list[tuple[str, str]] = []

    for subset in target_subsets:
        subset_dir = os.path.join(arena_dir, subset)
        logger.info(f"Scanning subset: {subset}")

        # Collect files from models/, pk_logs/, arena/
        for subdir in ["models", "pk_logs", "arena"]:
            subdir_path = os.path.join(subset_dir, subdir)
            if os.path.isdir(subdir_path):
                files = collect_files_follow_symlinks(subdir_path, f"{subset}/{subdir}")
                all_files.extend(files)
                logger.info(f"  {subdir}: {len(files)} files")

    if not all_files:
        return False, "No files found to upload"

    logger.info(f"Total files to upload: {len(all_files)}")

    # Filter by extension (only upload relevant files)
    valid_extensions = {".png", ".jpg", ".jpeg", ".webp", ".json", ".jsonl"}
    all_files = [
        (local, remote) for local, remote in all_files
        if os.path.splitext(local)[1].lower() in valid_extensions
    ]
    logger.info(f"Files after extension filtering: {len(all_files)}")

    # Filter out files in subdirectories under models/<exp>/<model>/
    # Expected structure: {subset}/models/{exp}/{model}/{file}
    # Files deeper than this (e.g., {subset}/models/{exp}/{model}/subfolder/{file}) should be skipped
    def is_valid_model_path(remote: str) -> bool:
        parts = remote.split("/")
        # Non-models paths are always valid
        if len(parts) < 2 or parts[1] != "models":
            return True
        # For models paths, expect exactly: subset/models/exp/model/file (5 parts)
        return len(parts) == 5

    before_depth_filter = len(all_files)
    all_files = [(local, remote) for local, remote in all_files if is_valid_model_path(remote)]
    depth_filtered = before_depth_filter - len(all_files)
    if depth_filtered > 0:
        logger.info(f"Skipped {depth_filtered} files in model subdirectories")
    logger.info(f"Files after filtering: {len(all_files)}")

    # Get existing files in repo (for skip check)
    existing_files: set[str] = set()
    if not overwrite:
        try:
            existing_files = set(api.list_repo_files(repo_id=repo_id, repo_type="dataset"))
            logger.info(f"Existing files in repo: {len(existing_files)}")
        except Exception:
            pass

    # Filter out existing files
    if not overwrite:
        original_count = len(all_files)
        all_files = [
            (local, remote) for local, remote in all_files
            if remote not in existing_files
        ]
        skipped = original_count - len(all_files)
        logger.info(f"Skipping {skipped} existing files, {len(all_files)} to upload")
    else:
        skipped = 0

    if not all_files:
        return True, f"All files already exist. Skipped {skipped} files."

    # Upload in batches using create_commit with multiprocessing
    batch_size = 500  # HuggingFace recommends smaller batches for large files
    max_retries = 3

    # Create batches
    batches = []
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i:i + batch_size]
        batches.append(batch)

    total_batches = len(batches)
    logger.info(f"Uploading {total_batches} batches with {num_workers} workers (timeout: {worker_timeout}s per worker)")

    # Prepare worker arguments
    worker_args = [
        (i, batch, repo_id, token, total_batches, max_retries)
        for i, batch in enumerate(batches)
    ]

    total_uploaded = 0
    total_failed = 0

    # Use multiprocessing pool
    with Pool(processes=num_workers) as pool:
        if show_progress:
            try:
                from tqdm import tqdm
                results = list(tqdm(
                    pool.imap_unordered(_upload_batch_worker, worker_args),
                    total=total_batches,
                    desc="Uploading batches",
                    unit="batch",
                ))
            except ImportError:
                results = []
                for args in worker_args:
                    try:
                        result = pool.apply_async(_upload_batch_worker, (args,))
                        uploaded, failed = result.get(timeout=worker_timeout)
                        results.append((uploaded, failed))
                    except Exception as e:
                        logger.error(f"Worker timeout or error: {e}")
                        results.append((0, len(args[1])))
        else:
            results = []
            for args in worker_args:
                try:
                    result = pool.apply_async(_upload_batch_worker, (args,))
                    uploaded, failed = result.get(timeout=worker_timeout)
                    results.append((uploaded, failed))
                except Exception as e:
                    logger.error(f"Worker timeout or error: {e}")
                    results.append((0, len(args[1])))

    # Aggregate results
    for uploaded, failed in results:
        total_uploaded += uploaded
        total_failed += failed

    return True, f"Uploaded {total_uploaded}, skipped {skipped}, failed {total_failed} files"


def upload_deploy_files(
    space_repo: str,
    overwrite: bool = False,
    token: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Upload deploy files (Dockerfile, app.py, README.md) to Space repo.

    Args:
        space_repo: HF Space repo ID
        overwrite: Overwrite existing files
        token: HF token

    Returns:
        Tuple of (success, message)
    """
    from huggingface_hub import HfApi

    from genarena.sync.hf_ops import upload_file

    if token is None:
        from genarena.sync.hf_ops import require_hf_token

        token = require_hf_token()

    api = HfApi(token=token)

    # Get deploy directory
    deploy_dir = os.path.dirname(os.path.abspath(__file__))
    deploy_dir = os.path.join(os.path.dirname(deploy_dir), "deploy")

    if not os.path.isdir(deploy_dir):
        return False, f"Deploy directory not found: {deploy_dir}"

    # Files to upload
    deploy_files = [
        ("Dockerfile", "Dockerfile"),
        ("app.py", "genarena/deploy/app.py"),
        ("README.md", "README.md"),
    ]

    # Get existing files
    existing_files: set[str] = set()
    if not overwrite:
        try:
            existing_files = set(
                api.list_repo_files(repo_id=space_repo, repo_type="space")
            )
        except Exception:
            pass

    uploaded = 0
    skipped = 0
    failed = 0

    for local_name, remote_path in deploy_files:
        local_path = os.path.join(deploy_dir, local_name)
        if not os.path.isfile(local_path):
            logger.warning(f"Deploy file not found: {local_path}")
            continue

        if not overwrite and remote_path in existing_files:
            skipped += 1
            continue

        success, msg = upload_file(
            repo_id=space_repo,
            local_path=local_path,
            remote_path=remote_path,
            token=token,
            commit_message=f"Upload {remote_path}",
            repo_type="space",
        )
        if success:
            uploaded += 1
        else:
            failed += 1
            logger.warning(f"Failed to upload {remote_path}: {msg}")

    # Also upload the genarena package files needed for the Space
    # We need to upload the entire genarena package
    success, msg = upload_genarena_package(space_repo, token, overwrite)
    if not success:
        return False, f"Failed to upload genarena package: {msg}"

    return True, f"Uploaded {uploaded}, skipped {skipped}, failed {failed} deploy files. {msg}"


def upload_genarena_package(
    space_repo: str,
    token: str,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """
    Upload the genarena package to the Space repo.

    Args:
        space_repo: HF Space repo ID
        token: HF token
        overwrite: Overwrite existing files

    Returns:
        Tuple of (success, message)
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Get genarena package directory
    genarena_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(genarena_dir)

    try:
        # Upload pyproject.toml
        pyproject_path = os.path.join(project_root, "pyproject.toml")
        if os.path.isfile(pyproject_path):
            api.upload_file(
                repo_id=space_repo,
                path_or_fileobj=pyproject_path,
                path_in_repo="pyproject.toml",
                repo_type="space",
                commit_message="Upload pyproject.toml",
            )

        # Upload genarena package using upload_folder
        api.upload_folder(
            repo_id=space_repo,
            folder_path=genarena_dir,
            path_in_repo="genarena",
            repo_type="space",
            commit_message="[genarena deploy] Upload genarena package",
            allow_patterns=["**/*.py", "**/*.html", "**/*.css", "**/*.js"],
            ignore_patterns=["**/__pycache__/**", "**/.pytest_cache/**"],
        )

        return True, "Package uploaded successfully"
    except Exception as e:
        logger.error(f"Failed to upload package: {e}")
        return False, str(e)
