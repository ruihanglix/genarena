# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Initialization operations for GenArena.

This module provides functionality for one-click initialization of arena
directories, including downloading benchmark data and official arena data
from HuggingFace repositories.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Default repository configurations
DEFAULT_BENCHMARK_REPO = "rhli/genarena"
DEFAULT_ARENA_REPO = "rhli/genarena-battlefield"


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.1f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


def discover_repo_subsets(
    repo_id: str,
    token: Optional[str] = None,
    revision: str = "main",
) -> list[str]:
    """
    Discover available subsets in a HuggingFace repository.

    Looks for directories containing parquet files or known subset patterns.

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace token (optional for public repos)
        revision: Repository revision/branch

    Returns:
        List of subset names found in the repository
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    try:
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        )
    except Exception as e:
        logger.warning(f"Failed to list repo files: {e}")
        return []

    # Find directories that contain parquet files
    subsets: set[str] = set()
    for f in files:
        # Look for patterns like: <subset>/data-*.parquet or <subset>/*.parquet
        if f.endswith(".parquet"):
            parts = f.split("/")
            if len(parts) >= 2:
                # First directory is the subset name
                subset = parts[0]
                # Skip hidden directories and common non-subset directories
                if not subset.startswith(".") and subset not in ("data", "raw"):
                    subsets.add(subset)

    return sorted(subsets)


def download_benchmark_data(
    data_dir: str,
    repo_id: str = DEFAULT_BENCHMARK_REPO,
    subsets: Optional[list[str]] = None,
    revision: str = "main",
    overwrite: bool = False,
    show_progress: bool = True,
) -> tuple[bool, str, dict]:
    """
    Download benchmark Parquet data from HuggingFace.

    Expected repository structure:
        <subset>/data-00000-of-00001.parquet
        <subset>/data-00001-of-00001.parquet
        ...

    Downloads to:
        data_dir/<subset>/data-*.parquet

    Args:
        data_dir: Local directory to save data
        repo_id: HuggingFace repository ID
        subsets: List of subsets to download (None = all available)
        revision: Repository revision/branch
        overwrite: If True, overwrite existing files
        show_progress: If True, show progress information

    Returns:
        Tuple of (success, message, stats_dict)
    """
    from huggingface_hub import HfApi, hf_hub_download

    from genarena.sync.hf_ops import get_hf_token

    token = get_hf_token()
    api = HfApi(token=token)

    stats = {
        "downloaded_files": 0,
        "skipped_files": 0,
        "failed_files": 0,
        "total_bytes": 0,
        "subsets": {},
    }

    # Discover available subsets if not specified
    if subsets is None:
        logger.info(f"Discovering subsets in {repo_id}...")
        subsets = discover_repo_subsets(repo_id, token, revision)
        if not subsets:
            return False, f"No subsets found in repository {repo_id}", stats
        logger.info(f"Found subsets: {', '.join(subsets)}")

    # List all files in the repo
    try:
        all_files = list(api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
        ))
    except Exception as e:
        return False, f"Failed to list repository files: {e}", stats

    # Filter files for requested subsets
    files_to_download: list[tuple[str, str]] = []  # (remote_path, local_path)

    for subset in subsets:
        subset_files = [
            f for f in all_files
            if f.startswith(f"{subset}/") and f.endswith(".parquet")
        ]

        if not subset_files:
            logger.warning(f"No parquet files found for subset '{subset}'")
            continue

        stats["subsets"][subset] = {
            "files": len(subset_files),
            "bytes": 0,
            "downloaded": 0,
            "skipped": 0,
        }

        for remote_path in subset_files:
            # Construct local path: data_dir/<subset>/filename.parquet
            local_path = os.path.join(data_dir, remote_path)
            files_to_download.append((remote_path, local_path))

    if not files_to_download:
        return False, "No parquet files found for the specified subsets", stats

    # Create data directory
    os.makedirs(data_dir, exist_ok=True)

    # Download files
    errors: list[str] = []

    if show_progress:
        try:
            from tqdm import tqdm
            files_iter = tqdm(files_to_download, desc="Downloading", unit="file")
        except ImportError:
            files_iter = files_to_download
    else:
        files_iter = files_to_download

    for remote_path, local_path in files_iter:
        subset = remote_path.split("/")[0]

        # Check if file exists
        if os.path.exists(local_path) and not overwrite:
            logger.debug(f"Skipping existing file: {local_path}")
            stats["skipped_files"] += 1
            if subset in stats["subsets"]:
                stats["subsets"][subset]["skipped"] += 1
            continue

        # Create directory
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        try:
            # Download file
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
                token=token,
                local_dir=data_dir,
                local_dir_use_symlinks=False,
            )

            # Get file size
            file_size = os.path.getsize(downloaded_path)
            stats["downloaded_files"] += 1
            stats["total_bytes"] += file_size

            if subset in stats["subsets"]:
                stats["subsets"][subset]["downloaded"] += 1
                stats["subsets"][subset]["bytes"] += file_size

            logger.debug(f"Downloaded: {remote_path} ({_format_size(file_size)})")

        except Exception as e:
            logger.error(f"Failed to download {remote_path}: {e}")
            errors.append(f"{remote_path}: {e}")
            stats["failed_files"] += 1

    # Build summary message
    lines = [
        f"Benchmark data download complete:",
        f"  Downloaded: {stats['downloaded_files']} files ({_format_size(stats['total_bytes'])})",
        f"  Skipped: {stats['skipped_files']} files (already exist)",
        f"  Failed: {stats['failed_files']} files",
    ]

    if stats["subsets"]:
        lines.append("  Subsets:")
        for subset, info in stats["subsets"].items():
            lines.append(
                f"    - {subset}: {info['downloaded']} downloaded, "
                f"{info['skipped']} skipped ({_format_size(info['bytes'])})"
            )

    if errors:
        lines.append("  Errors:")
        for err in errors[:5]:
            lines.append(f"    - {err}")
        if len(errors) > 5:
            lines.append(f"    ... and {len(errors) - 5} more errors")

    success = stats["failed_files"] == 0 or stats["downloaded_files"] > 0
    return success, "\n".join(lines), stats


def init_arena(
    arena_dir: str = "./arena",
    data_dir: str = "./data",
    subsets: Optional[list[str]] = None,
    benchmark_repo: str = DEFAULT_BENCHMARK_REPO,
    arena_repo: str = DEFAULT_ARENA_REPO,
    revision: str = "main",
    overwrite: bool = False,
    init_git: bool = False,
    data_only: bool = False,
    arena_only: bool = False,
    show_progress: bool = True,
) -> tuple[bool, str]:
    """
    One-click arena initialization.

    This function:
    1. Downloads benchmark Parquet data from HuggingFace (unless arena_only)
    2. Downloads arena data (model outputs + logs) from HuggingFace (unless data_only)
    3. Initializes Git repository in arena_dir (if init_git)

    Args:
        arena_dir: Path to arena directory
        data_dir: Path to benchmark data directory
        subsets: List of subsets to download (None = all available)
        benchmark_repo: HuggingFace repo for benchmark data
        arena_repo: HuggingFace repo for arena data
        revision: HuggingFace revision/branch
        overwrite: If True, overwrite existing files
        init_git: If True, initialize Git repository in arena_dir
        data_only: If True, only download benchmark data
        arena_only: If True, only download arena data
        show_progress: If True, show progress information

    Returns:
        Tuple of (success, summary_message)
    """
    from genarena.sync.hf_ops import pull_arena_data, get_hf_token
    from genarena.sync.git_ops import git_init, is_git_initialized

    lines: list[str] = []
    all_success = True
    benchmark_stats: dict = {}
    arena_stats: dict = {}

    # Resolve absolute paths
    arena_dir = os.path.abspath(arena_dir)
    data_dir = os.path.abspath(data_dir)

    # Step 1: Download benchmark data
    if not arena_only:
        step_num = 1
        total_steps = 2 if not data_only else 1
        if init_git:
            total_steps += 1

        print(f"[Step {step_num}/{total_steps}] Downloading benchmark data from {benchmark_repo}...")
        print(f"  Target directory: {data_dir}")
        if subsets:
            print(f"  Subsets: {', '.join(subsets)}")
        print()

        success, msg, benchmark_stats = download_benchmark_data(
            data_dir=data_dir,
            repo_id=benchmark_repo,
            subsets=subsets,
            revision=revision,
            overwrite=overwrite,
            show_progress=show_progress,
        )

        print(f"  {msg.replace(chr(10), chr(10) + '  ')}")
        print()

        if not success:
            all_success = False
            lines.append(f"Benchmark data download failed")
        else:
            lines.append(
                f"Benchmark data: {benchmark_stats.get('downloaded_files', 0)} files "
                f"({_format_size(benchmark_stats.get('total_bytes', 0))})"
            )

    # Step 2: Download arena data
    if not data_only:
        step_num = 1 if arena_only else 2
        total_steps = 1 if arena_only else 2
        if init_git:
            total_steps += 1

        print(f"[Step {step_num}/{total_steps}] Downloading arena data from {arena_repo}...")
        print(f"  Target directory: {arena_dir}")
        if subsets:
            print(f"  Subsets: {', '.join(subsets)}")
        print()

        # Create arena directory
        os.makedirs(arena_dir, exist_ok=True)

        success, msg = pull_arena_data(
            arena_dir=arena_dir,
            repo_id=arena_repo,
            subsets=subsets,
            revision=revision,
            overwrite=overwrite,
            show_progress=show_progress,
        )

        print(f"  {msg.replace(chr(10), chr(10) + '  ')}")
        print()

        if not success:
            all_success = False
            lines.append(f"Arena data download failed: {msg}")
        else:
            lines.append(f"Arena data: downloaded to {arena_dir}")

    # Step 3: Initialize Git
    if init_git and not data_only:
        step_num = total_steps
        print(f"[Step {step_num}/{total_steps}] Initializing Git repository...")

        if is_git_initialized(arena_dir):
            print(f"  Git repository already initialized at {arena_dir}")
            lines.append("Git: already initialized")
        else:
            success, msg = git_init(arena_dir)
            print(f"  {msg}")
            if success:
                lines.append("Git: initialized")
            else:
                lines.append(f"Git: initialization failed - {msg}")
        print()

    # Build final summary
    summary_lines = [
        "=== Summary ===",
    ]

    if not arena_only:
        summary_lines.append(f"Data directory:  {data_dir}")
    if not data_only:
        summary_lines.append(f"Arena directory: {arena_dir}")

    if subsets:
        summary_lines.append(f"Subsets:         {', '.join(subsets)}")
    elif benchmark_stats.get("subsets"):
        summary_lines.append(f"Subsets:         {', '.join(benchmark_stats['subsets'].keys())}")

    for line in lines:
        summary_lines.append(f"  {line}")

    # Add next steps
    summary_lines.append("")
    summary_lines.append("Next steps:")

    if not data_only:
        summary_lines.append(f"  # View current status")
        summary_lines.append(f"  genarena status --arena_dir {arena_dir} --data_dir {data_dir}")
        summary_lines.append("")
        summary_lines.append(f"  # Run evaluation battles")
        example_subset = subsets[0] if subsets else "basic"
        summary_lines.append(
            f"  genarena run --arena_dir {arena_dir} --data_dir {data_dir} --subset {example_subset}"
        )
        summary_lines.append("")
        summary_lines.append(f"  # View leaderboard")
        summary_lines.append(f"  genarena leaderboard --arena_dir {arena_dir} --subset {example_subset}")
    else:
        summary_lines.append(f"  # Initialize arena directory")
        summary_lines.append(f"  genarena init --arena_dir <path> --arena-only")

    return all_success, "\n".join(summary_lines)
