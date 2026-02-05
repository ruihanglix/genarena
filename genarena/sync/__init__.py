"""
Sync module for GenArena.

This module provides Git version control and Huggingface synchronization
capabilities for arena data.
"""

from genarena.sync.git_ops import (
    is_git_initialized,
    git_init,
    ensure_gitignore,
    git_add_all,
    git_commit,
    has_uncommitted_changes,
    git_remote_add,
    git_remote_get_url,
    git_push,
    git_sync,
)

from genarena.sync.auto_commit import (
    auto_commit_and_push,
    with_auto_commit,
)

from genarena.sync.hf_ops import (
    get_hf_token,
    require_hf_token,
    validate_dataset_repo,
    list_repo_files,
    get_repo_file_info,
    upload_file,
    upload_files_batch,
    download_file,
    check_file_exists,
    upload_arena_data,
    pull_arena_data,
    list_repo_contents,
)

from genarena.sync.packer import (
    pack_model_dir,
    pack_exp_dir,
    unpack_zip,
    collect_upload_tasks,
    collect_download_tasks,
    TempPackingContext,
    TaskType,
    PackTask,
    UnpackTask,
)

from genarena.sync.init_ops import (
    DEFAULT_BENCHMARK_REPO,
    DEFAULT_ARENA_REPO,
    discover_repo_subsets,
    download_benchmark_data,
    init_arena,
)

__all__ = [
    # Git operations
    "is_git_initialized",
    "git_init",
    "ensure_gitignore",
    "git_add_all",
    "git_commit",
    "has_uncommitted_changes",
    "git_remote_add",
    "git_remote_get_url",
    "git_push",
    "git_sync",
    # Auto commit
    "auto_commit_and_push",
    "with_auto_commit",
    # Huggingface operations
    "get_hf_token",
    "require_hf_token",
    "validate_dataset_repo",
    "list_repo_files",
    "get_repo_file_info",
    "upload_file",
    "upload_files_batch",
    "download_file",
    "check_file_exists",
    "upload_arena_data",
    "pull_arena_data",
    "list_repo_contents",
    # Packer utilities
    "pack_model_dir",
    "pack_exp_dir",
    "unpack_zip",
    "collect_upload_tasks",
    "collect_download_tasks",
    "TempPackingContext",
    "TaskType",
    "PackTask",
    "UnpackTask",
    # Init operations
    "DEFAULT_BENCHMARK_REPO",
    "DEFAULT_ARENA_REPO",
    "discover_repo_subsets",
    "download_benchmark_data",
    "init_arena",
]
