# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
ZIP packing utilities for GenArena.

This module provides functionality for packing and unpacking arena data
for Huggingface upload/download operations.
"""

import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Supported image file extensions for model directories
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".svg"}


class TaskType(Enum):
    """Type of upload/download task."""
    MODEL_ZIP = "model_zip"        # ZIP file for experiment-scoped model images
    EXP_ZIP = "exp_zip"            # ZIP file for experiment logs
    SMALL_FILE = "small_file"      # Small file (state.json, README.md)


@dataclass
class PackTask:
    """Represents a file packing/upload task."""
    task_type: TaskType
    local_path: str           # Local path (directory for ZIP, file for small files)
    remote_path: str          # Remote path in the HF repo
    subset: str               # Subset name
    name: str                 # Model name or experiment name or file name


@dataclass
class UnpackTask:
    """Represents a file unpacking/download task."""
    task_type: TaskType
    remote_path: str          # Remote path in the HF repo
    local_path: str           # Local target path
    subset: str               # Subset name
    name: str                 # Model name or experiment name or file name


def pack_directory(
    source_dir: str,
    output_zip: str,
    file_extensions: Optional[set] = None,
    max_depth: Optional[int] = None,
) -> tuple[bool, str]:
    """
    Pack a directory into a ZIP file.

    The directory name is preserved as the root folder inside the ZIP.
    Symbolic links are followed and the actual file contents are packed.

    Args:
        source_dir: Path to the directory to pack
        output_zip: Path to the output ZIP file
        file_extensions: Optional set of file extensions to include (e.g., {".png", ".jpg"}).
                        If None, all files are included. Extensions should be lowercase with dot.
        max_depth: Optional maximum directory depth to traverse. None means unlimited.
                   0 = only files directly in source_dir
                   1 = files in source_dir and its immediate subdirectories
                   etc.

    Returns:
        Tuple of (success, message)
    """
    if not os.path.isdir(source_dir):
        return False, f"Source directory does not exist: {source_dir}"

    # Resolve symlink if source_dir itself is a symlink
    resolved_source = os.path.realpath(source_dir)
    if not os.path.isdir(resolved_source):
        return False, f"Source directory symlink target does not exist: {resolved_source}"

    # Get the directory name to use as root in ZIP (use original name, not resolved)
    dir_name = os.path.basename(source_dir.rstrip(os.sep))

    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_zip), exist_ok=True)

        file_count = 0
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            # followlinks=True to traverse symlinked directories
            for root, dirs, files in os.walk(resolved_source, followlinks=True):
                # Calculate current depth relative to source
                if max_depth is not None:
                    rel_root = os.path.relpath(root, resolved_source)
                    if rel_root == ".":
                        current_depth = 0
                    else:
                        current_depth = len(rel_root.split(os.sep))

                    # Skip directories beyond max_depth
                    if current_depth > max_depth:
                        dirs[:] = []  # Prevent further recursion
                        continue

                    # Stop recursion at max_depth
                    if current_depth == max_depth:
                        dirs[:] = []

                for file in files:
                    # Filter by extension if specified
                    if file_extensions is not None:
                        ext = os.path.splitext(file)[1].lower()
                        if ext not in file_extensions:
                            continue

                    file_path = os.path.join(root, file)

                    # Skip broken symlinks
                    if os.path.islink(file_path) and not os.path.exists(file_path):
                        logger.warning(f"Skipping broken symlink: {file_path}")
                        continue

                    # Calculate archive name: use original dir_name as root
                    rel_to_resolved = os.path.relpath(file_path, resolved_source)
                    archive_name = os.path.join(dir_name, rel_to_resolved)
                    zf.write(file_path, archive_name)
                    file_count += 1

        if file_count == 0:
            # Remove empty ZIP file
            os.remove(output_zip)
            return False, f"No files to pack in {source_dir}"

        return True, f"Packed {source_dir} -> {output_zip} ({file_count} files)"
    except Exception as e:
        return False, f"Failed to pack directory: {e}"


def pack_model_dir(model_dir: str, output_zip: str) -> tuple[bool, str]:
    """
    Pack a single model directory (containing images) into a ZIP file.

    Only image files (png, jpg, jpeg, gif, webp, bmp, tiff, svg) are packed.
    Only files directly under the model directory are included;
    nested subdirectories (e.g., fail/) are excluded.

    Args:
        model_dir: Path to the model directory (e.g., arena_dir/basic/models/exp_001/model_a/)
        output_zip: Path to the output ZIP file

    Returns:
        Tuple of (success, message)
    """
    return pack_directory(model_dir, output_zip, file_extensions=IMAGE_EXTENSIONS, max_depth=0)


def pack_exp_dir(exp_dir: str, output_zip: str) -> tuple[bool, str]:
    """
    Pack an experiment directory (containing battle logs) into a ZIP file.

    Args:
        exp_dir: Path to the experiment directory (e.g., arena_dir/basic/pk_logs/exp_001/)
        output_zip: Path to the output ZIP file

    Returns:
        Tuple of (success, message)
    """
    return pack_directory(exp_dir, output_zip)


def unpack_zip(
    zip_path: str,
    target_dir: str,
    overwrite: bool = False,
) -> tuple[bool, str]:
    """
    Unpack a ZIP file to a target directory.

    Args:
        zip_path: Path to the ZIP file
        target_dir: Target directory to extract to
        overwrite: If True, overwrite existing files

    Returns:
        Tuple of (success, message)
    """
    if not os.path.isfile(zip_path):
        return False, f"ZIP file does not exist: {zip_path}"

    try:
        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                target_path = os.path.join(target_dir, member)

                # Check if file exists and skip if not overwriting
                if os.path.exists(target_path) and not overwrite:
                    logger.debug(f"Skipping existing file: {target_path}")
                    continue

                # Extract file
                zf.extract(member, target_dir)

        return True, f"Unpacked {zip_path} -> {target_dir}"
    except Exception as e:
        return False, f"Failed to unpack ZIP: {e}"


def discover_subsets(arena_dir: str) -> list[str]:
    """
    Discover all subset directories in the arena directory.

    A valid subset directory contains at least one of: models/, pk_logs/, arena/

    Args:
        arena_dir: Path to the arena directory

    Returns:
        List of subset names
    """
    subsets = []

    if not os.path.isdir(arena_dir):
        return subsets

    for name in os.listdir(arena_dir):
        subset_path = os.path.join(arena_dir, name)
        if not os.path.isdir(subset_path):
            continue

        # Check if it looks like a subset directory
        has_models = os.path.isdir(os.path.join(subset_path, "models"))
        has_pk_logs = os.path.isdir(os.path.join(subset_path, "pk_logs"))
        has_arena = os.path.isdir(os.path.join(subset_path, "arena"))

        if has_models or has_pk_logs or has_arena:
            subsets.append(name)

    return sorted(subsets)


def discover_models(arena_dir: str, subset: str) -> list[str]:
    """
    Discover all model names in a subset (v2 layout).

    Args:
        arena_dir: Path to the arena directory
        subset: Subset name

    Returns:
        List of model names (globally unique across experiments)
    """
    from genarena.models import GlobalModelOutputManager

    models_root = os.path.join(arena_dir, subset, "models")
    if not os.path.isdir(models_root):
        return []
    try:
        mgr = GlobalModelOutputManager(models_root)
        return mgr.models
    except Exception:
        # For packer utilities, be conservative: return empty on scan failure.
        return []


def discover_model_experiments(arena_dir: str, subset: str) -> list[str]:
    """
    Discover experiment directories under a subset's models (v2 layout).

    In v2, model outputs live under:
        models/<exp_name>/<model_name>/...
    This function returns exp_name directories that contain at least one model with images.
    """
    from genarena.models import GlobalModelOutputManager

    models_root = os.path.join(arena_dir, subset, "models")
    if not os.path.isdir(models_root):
        return []
    try:
        mgr = GlobalModelOutputManager(models_root)
        return mgr.experiments
    except Exception:
        return []


def discover_experiments(arena_dir: str, subset: str) -> list[str]:
    """
    Discover all experiment directories in a subset's pk_logs.

    Excludes .pk_logs_rm (deleted/orphaned logs).

    Args:
        arena_dir: Path to the arena directory
        subset: Subset name

    Returns:
        List of experiment names
    """
    pk_logs_dir = os.path.join(arena_dir, subset, "pk_logs")
    experiments = []

    if not os.path.isdir(pk_logs_dir):
        return experiments

    for name in os.listdir(pk_logs_dir):
        # Skip hidden directories and .pk_logs_rm
        if name.startswith("."):
            continue

        exp_path = os.path.join(pk_logs_dir, name)
        if os.path.isdir(exp_path):
            experiments.append(name)

    return sorted(experiments)


def collect_upload_tasks(
    arena_dir: str,
    subsets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    experiments: Optional[list[str]] = None,
) -> list[PackTask]:
    """
    Collect all files/directories that need to be uploaded.

    Args:
        arena_dir: Path to the arena directory
        subsets: List of subsets to include (None = all)
        models: List of models to include (None = all)

    Returns:
        List of PackTask objects
    """
    tasks = []

    # Discover subsets if not specified
    all_subsets = discover_subsets(arena_dir)
    target_subsets = subsets if subsets else all_subsets

    for subset in target_subsets:
        if subset not in all_subsets:
            logger.warning(f"Subset '{subset}' not found in arena directory")
            continue

        subset_path = os.path.join(arena_dir, subset)

        # Collect model directories (v2 layout: models/<exp_name>/<model_name>/):
        # Each model is packed as a separate ZIP file.
        # - Default: upload all models
        # - If experiments filter is provided: only models under those exp_name
        # - If models filter is provided: only those specific models
        models_root = os.path.join(subset_path, "models")
        all_model_exps = discover_model_experiments(arena_dir, subset)

        target_model_exps: list[str]
        if experiments:
            target_model_exps = [e for e in experiments if e in all_model_exps]
        else:
            target_model_exps = all_model_exps

        # Collect individual model directories
        for exp in target_model_exps:
            exp_model_path = os.path.join(subset_path, "models", exp)
            if not os.path.isdir(exp_model_path):
                continue

            # List all model directories under this experiment
            for model_name in os.listdir(exp_model_path):
                model_path = os.path.join(exp_model_path, model_name)
                if not os.path.isdir(model_path):
                    continue

                # Apply models filter if specified
                if models and model_name not in models:
                    continue

                remote_path = f"{subset}/models/{exp}/{model_name}.zip"

                tasks.append(PackTask(
                    task_type=TaskType.MODEL_ZIP,
                    local_path=model_path,
                    remote_path=remote_path,
                    subset=subset,
                    name=f"{exp}/{model_name}",
                ))

        # Collect experiment directories (only if no model filter, or always)
        # Note: pk_logs are always uploaded regardless of model filter
        pk_experiments = discover_experiments(arena_dir, subset)
        if experiments:
            pk_experiments = [e for e in pk_experiments if e in set(experiments)]
        for exp in pk_experiments:
            exp_path = os.path.join(subset_path, "pk_logs", exp)
            remote_path = f"{subset}/pk_logs/{exp}.zip"

            tasks.append(PackTask(
                task_type=TaskType.EXP_ZIP,
                local_path=exp_path,
                remote_path=remote_path,
                subset=subset,
                name=exp,
            ))

        # Collect small files
        # state.json
        state_path = os.path.join(subset_path, "arena", "state.json")
        if os.path.isfile(state_path):
            tasks.append(PackTask(
                task_type=TaskType.SMALL_FILE,
                local_path=state_path,
                remote_path=f"{subset}/arena/state.json",
                subset=subset,
                name="state.json",
            ))

        # README.md
        readme_path = os.path.join(subset_path, "README.md")
        if os.path.isfile(readme_path):
            tasks.append(PackTask(
                task_type=TaskType.SMALL_FILE,
                local_path=readme_path,
                remote_path=f"{subset}/README.md",
                subset=subset,
                name="README.md",
            ))

    return tasks


def collect_download_tasks(
    repo_files: list[str],
    arena_dir: str,
    subsets: Optional[list[str]] = None,
    models: Optional[list[str]] = None,
    experiments: Optional[list[str]] = None,
) -> list[UnpackTask]:
    """
    Collect files to download based on repo contents and filters.

    Args:
        repo_files: List of file paths in the HF repo
        arena_dir: Local arena directory path
        subsets: List of subsets to download (None = all)
        models: List of models to download (None = all)
        experiments: List of experiments to download (None = all)

    Returns:
        List of UnpackTask objects
    """
    tasks = []

    for remote_path in repo_files:
        # Parse the remote path to determine type
        parts = remote_path.split("/")
        if len(parts) < 2:
            continue

        subset = parts[0]

        # Apply subset filter
        if subsets and subset not in subsets:
            continue

        # Determine task type and apply filters
        # New format: models/<exp_name>/<model_name>.zip
        if len(parts) >= 4 and parts[1] == "models" and parts[3].endswith(".zip"):
            exp_name = parts[2]
            model_name = parts[3][:-4]  # Remove .zip

            # Apply experiments filter
            if experiments and exp_name not in experiments:
                continue

            # Apply models filter
            if models and model_name not in models:
                continue

            local_path = os.path.join(arena_dir, subset, "models", exp_name)
            tasks.append(UnpackTask(
                task_type=TaskType.MODEL_ZIP,
                remote_path=remote_path,
                local_path=local_path,
                subset=subset,
                name=f"{exp_name}/{model_name}",
            ))

        # Legacy format: models/<exp_name>.zip (for backward compatibility)
        elif len(parts) == 3 and parts[1] == "models" and parts[2].endswith(".zip"):
            exp_name = parts[2][:-4]  # Remove .zip

            # Apply experiments filter (legacy: models filter acts as exp filter)
            exp_filter = experiments if experiments is not None else models
            if exp_filter and exp_name not in exp_filter:
                continue

            local_path = os.path.join(arena_dir, subset, "models")
            tasks.append(UnpackTask(
                task_type=TaskType.MODEL_ZIP,
                remote_path=remote_path,
                local_path=local_path,
                subset=subset,
                name=exp_name,
            ))

        elif len(parts) >= 3 and parts[1] == "pk_logs" and parts[2].endswith(".zip"):
            # Experiment ZIP file
            exp_name = parts[2][:-4]  # Remove .zip

            if experiments and exp_name not in experiments:
                continue

            local_path = os.path.join(arena_dir, subset, "pk_logs")
            tasks.append(UnpackTask(
                task_type=TaskType.EXP_ZIP,
                remote_path=remote_path,
                local_path=local_path,
                subset=subset,
                name=exp_name,
            ))

        elif len(parts) >= 3 and parts[1] == "arena" and parts[2] == "state.json":
            # state.json
            local_path = os.path.join(arena_dir, subset, "arena", "state.json")
            tasks.append(UnpackTask(
                task_type=TaskType.SMALL_FILE,
                remote_path=remote_path,
                local_path=local_path,
                subset=subset,
                name="state.json",
            ))

        elif len(parts) >= 2 and parts[1] == "README.md":
            # README.md
            local_path = os.path.join(arena_dir, subset, "README.md")
            tasks.append(UnpackTask(
                task_type=TaskType.SMALL_FILE,
                remote_path=remote_path,
                local_path=local_path,
                subset=subset,
                name="README.md",
            ))

    return tasks


class TempPackingContext:
    """
    Context manager for temporary packing operations.

    Creates a temporary directory for ZIP files and cleans up on exit.
    """

    def __init__(self):
        self.temp_dir: Optional[str] = None

    def __enter__(self) -> "TempPackingContext":
        self.temp_dir = tempfile.mkdtemp(prefix="genarena_pack_")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def get_temp_zip_path(self, remote_path: str) -> str:
        """
        Get a temporary path for a ZIP file.

        Args:
            remote_path: The remote path (used to generate unique local path)

        Returns:
            Temporary file path
        """
        if not self.temp_dir:
            raise RuntimeError("TempPackingContext not entered")

        # Use the remote path structure for the temp file
        temp_path = os.path.join(self.temp_dir, remote_path)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        return temp_path
