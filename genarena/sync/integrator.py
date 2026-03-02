# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Integration logic for GenArena submissions.

Downloads submission data from user's HuggingFace repo, uploads to official
repos (battlefield + leaderboard-data), rebuilds arena state, and updates
the official models list.

Used by the GitHub Actions integration workflow after a submission PR is merged.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BATTLEFIELD_REPO = "rhli/genarena-battlefield"
DEFAULT_LEADERBOARD_REPO = "genarena/leaderboard-data"


@dataclass
class IntegrationResult:
    """Result of a submission integration."""

    success: bool = True
    submission_id: str = ""
    exp_name: str = ""
    subset: str = ""
    new_models: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add_message(self, msg: str) -> None:
        self.messages.append(msg)
        logger.info(msg)

    def add_error(self, err: str) -> None:
        self.errors.append(err)
        self.success = False
        logger.error(err)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "submission_id": self.submission_id,
            "exp_name": self.exp_name,
            "subset": self.subset,
            "new_models": self.new_models,
            "messages": self.messages,
            "errors": self.errors,
        }


def _sha256(path: str) -> str:
    """Compute SHA256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_submission_data(
    submission: dict[str, Any],
    download_dir: str,
) -> tuple[Optional[str], Optional[str], list[str]]:
    """
    Download and verify submission data from submitter's HuggingFace repo.

    Args:
        submission: Submission metadata dictionary
        download_dir: Directory to download files to

    Returns:
        Tuple of (models_zip_path, pk_logs_zip_path, errors)
    """
    from huggingface_hub import hf_hub_download

    errors: list[str] = []
    data_loc = submission.get("data_location", {})
    hf_repo = data_loc.get("hf_repo_id", "")
    hf_revision = data_loc.get("hf_revision", "main")
    files = data_loc.get("files", {})

    models_info = files.get("models_zip", {})
    pk_logs_info = files.get("pk_logs_zip", {})

    models_zip_path = None
    pk_logs_zip_path = None

    for label, info in [("models", models_info), ("pk_logs", pk_logs_info)]:
        try:
            local_path = hf_hub_download(
                repo_id=hf_repo,
                filename=info["path"],
                repo_type="dataset",
                revision=hf_revision,
                local_dir=download_dir,
            )
            actual_sha = _sha256(local_path)
            expected_sha = info.get("sha256", "")
            if actual_sha != expected_sha:
                errors.append(
                    f"{label} ZIP SHA256 mismatch: "
                    f"expected {expected_sha[:16]}..., got {actual_sha[:16]}..."
                )
            if label == "models":
                models_zip_path = local_path
            else:
                pk_logs_zip_path = local_path
        except Exception as e:
            errors.append(f"Failed to download {label} ZIP: {e}")

    return models_zip_path, pk_logs_zip_path, errors


def upload_to_battlefield(
    subset: str,
    exp_name: str,
    models_zip_path: str,
    pk_logs_zip_path: str,
    battlefield_repo: str,
    token: Optional[str] = None,
) -> list[str]:
    """
    Upload submission data to the battlefield repo.

    Battlefield format:
      {subset}/models/{exp_name}/{model_name}.zip  (per-model ZIPs)
      {subset}/pk_logs/{exp_name}.zip               (per-experiment ZIP)

    The submission's models ZIP bundles all models in a single ZIP with
    internal structure: {exp_name}/{model_name}/images...
    This function re-packs into per-model ZIPs before uploading.
    """
    from huggingface_hub import HfApi

    errors: list[str] = []
    api = HfApi(token=token)

    # Upload pk_logs ZIP directly (format is already compatible)
    try:
        pk_logs_remote = f"{subset}/pk_logs/{exp_name}.zip"
        api.upload_file(
            path_or_fileobj=pk_logs_zip_path,
            path_in_repo=pk_logs_remote,
            repo_id=battlefield_repo,
            repo_type="dataset",
            commit_message=f"[integration] Add pk_logs for {exp_name}",
        )
        logger.info(f"Uploaded pk_logs to {battlefield_repo}/{pk_logs_remote}")
    except Exception as e:
        errors.append(f"Failed to upload pk_logs ZIP: {e}")

    # Unzip models ZIP and re-pack as per-model ZIPs
    with tempfile.TemporaryDirectory() as tmpdir:
        extract_dir = os.path.join(tmpdir, "models_extracted")
        try:
            with zipfile.ZipFile(models_zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except Exception as e:
            errors.append(f"Failed to extract models ZIP: {e}")
            return errors

        # The ZIP contains {exp_name}/{model_name}/images...
        # Find the exp_name root directory
        exp_root = os.path.join(extract_dir, exp_name)
        if not os.path.isdir(exp_root):
            # Fallback: try the only top-level directory
            top_dirs = [
                d for d in os.listdir(extract_dir)
                if os.path.isdir(os.path.join(extract_dir, d))
            ]
            if len(top_dirs) == 1:
                exp_root = os.path.join(extract_dir, top_dirs[0])
            else:
                errors.append(
                    f"Unexpected models ZIP structure: expected {exp_name}/ "
                    f"root, found {top_dirs}"
                )
                return errors

        # Create per-model ZIPs and upload
        for model_name in os.listdir(exp_root):
            model_dir = os.path.join(exp_root, model_name)
            if not os.path.isdir(model_dir):
                continue

            model_zip_path = os.path.join(tmpdir, f"{model_name}.zip")
            try:
                with zipfile.ZipFile(model_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for root, _dirs, files in os.walk(model_dir):
                        for f in files:
                            filepath = os.path.join(root, f)
                            arcname = os.path.join(
                                model_name,
                                os.path.relpath(filepath, model_dir),
                            )
                            zf.write(filepath, arcname)

                remote_path = f"{subset}/models/{exp_name}/{model_name}.zip"
                api.upload_file(
                    path_or_fileobj=model_zip_path,
                    path_in_repo=remote_path,
                    repo_id=battlefield_repo,
                    repo_type="dataset",
                    commit_message=f"[integration] Add model {model_name} for {exp_name}",
                )
                logger.info(f"Uploaded {model_name} to {battlefield_repo}/{remote_path}")
            except Exception as e:
                errors.append(f"Failed to upload model {model_name}: {e}")

    return errors


def upload_to_leaderboard(
    subset: str,
    exp_name: str,
    models_zip_path: str,
    pk_logs_zip_path: str,
    leaderboard_repo: str,
    token: Optional[str] = None,
) -> list[str]:
    """
    Upload submission data to leaderboard-data repo (CDN format).

    Individual files are uploaded for direct HTTP access:
      {subset}/models/{exp_name}/{model_name}/{image}.png
      {subset}/pk_logs/{exp_name}/*.jsonl
    """
    from huggingface_hub import HfApi, CommitOperationAdd

    errors: list[str] = []
    api = HfApi(token=token)

    valid_model_exts = {".png", ".jpg", ".jpeg", ".webp"}
    valid_log_exts = {".jsonl", ".json", ".md"}

    with tempfile.TemporaryDirectory() as tmpdir:
        models_dir = os.path.join(tmpdir, "models")
        logs_dir = os.path.join(tmpdir, "logs")

        try:
            with zipfile.ZipFile(models_zip_path, "r") as zf:
                zf.extractall(models_dir)
        except Exception as e:
            errors.append(f"Failed to extract models ZIP: {e}")
            return errors

        try:
            with zipfile.ZipFile(pk_logs_zip_path, "r") as zf:
                zf.extractall(logs_dir)
        except Exception as e:
            errors.append(f"Failed to extract pk_logs ZIP: {e}")
            return errors

        operations: list[CommitOperationAdd] = []

        # Collect model files
        # ZIP internal structure: {exp_name}/{model_name}/image.png
        # CDN path: {subset}/models/{exp_name}/{model_name}/image.png
        for root, dirs, files in os.walk(models_dir):
            dirs[:] = [d for d in dirs if d != "raw_outputs"]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in valid_model_exts:
                    continue
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, models_dir).replace("\\", "/")
                # rel_path is: {exp_name}/{model_name}/image.png
                # Enforce depth = 3 (exp/model/file)
                if len(rel_path.split("/")) != 3:
                    continue
                remote_path = f"{subset}/models/{rel_path}"
                operations.append(
                    CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=local_path)
                )

        # Collect log files
        # ZIP internal structure: {exp_name}/file.jsonl
        # CDN path: {subset}/pk_logs/{exp_name}/file.jsonl
        for root, dirs, files in os.walk(logs_dir):
            dirs[:] = [d for d in dirs if d != "raw_outputs"]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in valid_log_exts:
                    continue
                local_path = os.path.join(root, f)
                rel_path = os.path.relpath(local_path, logs_dir).replace("\\", "/")
                remote_path = f"{subset}/pk_logs/{rel_path}"
                operations.append(
                    CommitOperationAdd(path_in_repo=remote_path, path_or_fileobj=local_path)
                )

        if not operations:
            errors.append("No files to upload to leaderboard repo")
            return errors

        # Batch upload (HF recommends batches <= 500 operations)
        batch_size = 500
        for i in range(0, len(operations), batch_size):
            batch = operations[i : i + batch_size]
            try:
                api.create_commit(
                    repo_id=leaderboard_repo,
                    repo_type="dataset",
                    operations=batch,
                    commit_message=(
                        f"[integration] Add {exp_name} data "
                        f"(batch {i // batch_size + 1}/{(len(operations) - 1) // batch_size + 1})"
                    ),
                )
                logger.info(
                    f"Uploaded batch {i // batch_size + 1} "
                    f"({len(batch)} files) to {leaderboard_repo}"
                )
            except Exception as e:
                errors.append(f"Failed to upload batch to leaderboard: {e}")

    return errors


def rebuild_and_upload_state(
    subset: str,
    battlefield_repo: str,
    leaderboard_repo: str,
    token: Optional[str] = None,
) -> list[str]:
    """
    Download all pk_logs for a subset, rebuild ELO state, upload to both repos.

    This is pure computation (Bradley-Terry model fitting) with no VLM API calls.
    """
    from huggingface_hub import HfApi, hf_hub_download

    from genarena.state import rebuild_state_from_logs, save_state

    errors: list[str] = []
    api = HfApi(token=token)

    with tempfile.TemporaryDirectory() as tmpdir:
        pk_logs_dir = os.path.join(tmpdir, "pk_logs")
        os.makedirs(pk_logs_dir)

        # List all pk_logs ZIPs in the battlefield repo for this subset
        try:
            all_files = api.list_repo_files(
                repo_id=battlefield_repo, repo_type="dataset"
            )
            pk_logs_zips = [
                f
                for f in all_files
                if f.startswith(f"{subset}/pk_logs/") and f.endswith(".zip")
            ]
        except Exception as e:
            errors.append(f"Failed to list files in {battlefield_repo}: {e}")
            return errors

        if not pk_logs_zips:
            errors.append(f"No pk_logs found for subset {subset}")
            return errors

        logger.info(f"Found {len(pk_logs_zips)} pk_logs ZIPs for subset {subset}")

        # Download and extract each ZIP directly into pk_logs_dir.
        # Each ZIP has internal root {exp_name}/, so extracting to pk_logs_dir
        # gives us pk_logs_dir/{exp_name}/files... which is the expected layout.
        for zip_file in pk_logs_zips:
            try:
                local_path = hf_hub_download(
                    repo_id=battlefield_repo,
                    filename=zip_file,
                    repo_type="dataset",
                    local_dir=os.path.join(tmpdir, "downloads"),
                )
                with zipfile.ZipFile(local_path, "r") as zf:
                    zf.extractall(pk_logs_dir)
                logger.info(f"Extracted {zip_file}")
            except Exception as e:
                errors.append(f"Failed to download/extract {zip_file}: {e}")

        # Rebuild state from all logs
        try:
            state = rebuild_state_from_logs(pk_logs_dir)
            state_path = os.path.join(tmpdir, "state.json")
            save_state(state, state_path)
            logger.info(
                f"Rebuilt state for {subset}: "
                f"{len(state.models)} models, {state.total_battles} battles"
            )
        except Exception as e:
            errors.append(f"Failed to rebuild state: {e}")
            return errors

        # Upload state.json to both repos
        remote_path = f"{subset}/arena/state.json"
        for repo_id in [battlefield_repo, leaderboard_repo]:
            try:
                api.upload_file(
                    path_or_fileobj=state_path,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type="dataset",
                    commit_message=f"[integration] Update state.json for {subset}",
                )
                logger.info(f"Uploaded state.json to {repo_id}")
            except Exception as e:
                errors.append(f"Failed to upload state.json to {repo_id}: {e}")

    return errors


def update_official_models_file(
    models_json_path: str,
    subset: str,
    new_models: list[str],
    total_battles: Optional[int] = None,
) -> list[str]:
    """
    Update official_models.json with newly integrated models.

    Args:
        models_json_path: Path to official_models.json
        subset: Subset name
        new_models: List of new model names to add
        total_battles: Updated total battle count (optional)

    Returns:
        List of errors (empty on success)
    """
    errors: list[str] = []

    try:
        with open(models_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        errors.append(f"Failed to read {models_json_path}: {e}")
        return errors

    if "subsets" not in data:
        data["subsets"] = {}
    if subset not in data["subsets"]:
        data["subsets"][subset] = {"models": [], "model_count": 0, "total_battles": 0}

    subset_data = data["subsets"][subset]
    existing = set(subset_data.get("models", []))
    for model in new_models:
        existing.add(model)

    subset_data["models"] = sorted(existing)
    subset_data["model_count"] = len(subset_data["models"])
    if total_battles is not None:
        subset_data["total_battles"] = total_battles

    data["last_updated"] = datetime.now(timezone.utc).isoformat()

    try:
        with open(models_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except IOError as e:
        errors.append(f"Failed to write {models_json_path}: {e}")

    return errors


def integrate_submission(
    submission_path: str,
    battlefield_repo: str = DEFAULT_BATTLEFIELD_REPO,
    leaderboard_repo: str = DEFAULT_LEADERBOARD_REPO,
    official_models_path: Optional[str] = None,
    token: Optional[str] = None,
) -> IntegrationResult:
    """
    Integrate an approved submission into official repositories.

    Orchestrates the full integration pipeline:
    1. Download submission data from submitter's HF repo
    2. Verify SHA256 checksums
    3. Upload to battlefield repo (per-model ZIPs)
    4. Upload to leaderboard-data repo (individual files for CDN)
    5. Rebuild ELO state from all pk_logs
    6. Update official_models.json

    Args:
        submission_path: Path to submission JSON file
        battlefield_repo: Official battlefield HF repo
        leaderboard_repo: Official leaderboard data HF repo
        official_models_path: Path to official_models.json (optional)
        token: HuggingFace API token

    Returns:
        IntegrationResult with success status and details
    """
    result = IntegrationResult()

    # Load submission
    try:
        with open(submission_path, "r", encoding="utf-8") as f:
            submission = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        result.add_error(f"Failed to load submission: {e}")
        return result

    result.submission_id = submission.get("submission_id", "")
    exp = submission.get("experiment", {})
    result.exp_name = exp.get("exp_name", "")
    result.subset = exp.get("subset", "")
    result.new_models = exp.get("new_models", [])

    result.add_message(
        f"Integrating {result.submission_id}: "
        f"exp={result.exp_name}, subset={result.subset}, "
        f"new_models={result.new_models}"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Download and verify
        result.add_message("Step 1/5: Downloading submission data...")
        models_zip, pk_logs_zip, dl_errors = download_submission_data(
            submission, tmpdir
        )
        for err in dl_errors:
            result.add_error(err)

        if not models_zip or not pk_logs_zip or not result.success:
            return result

        result.add_message("Download and verification complete")

        # Step 2: Upload to battlefield repo
        result.add_message(f"Step 2/5: Uploading to {battlefield_repo}...")
        bf_errors = upload_to_battlefield(
            subset=result.subset,
            exp_name=result.exp_name,
            models_zip_path=models_zip,
            pk_logs_zip_path=pk_logs_zip,
            battlefield_repo=battlefield_repo,
            token=token,
        )
        for err in bf_errors:
            result.add_error(err)
        if bf_errors:
            return result
        result.add_message("Battlefield upload complete")

        # Step 3: Upload to leaderboard-data repo
        result.add_message(f"Step 3/5: Uploading to {leaderboard_repo}...")
        lb_errors = upload_to_leaderboard(
            subset=result.subset,
            exp_name=result.exp_name,
            models_zip_path=models_zip,
            pk_logs_zip_path=pk_logs_zip,
            leaderboard_repo=leaderboard_repo,
            token=token,
        )
        for err in lb_errors:
            result.add_error(err)
        if lb_errors:
            return result
        result.add_message("Leaderboard upload complete")

    # Step 4: Rebuild state
    result.add_message("Step 4/5: Rebuilding arena state...")
    state_errors = rebuild_and_upload_state(
        subset=result.subset,
        battlefield_repo=battlefield_repo,
        leaderboard_repo=leaderboard_repo,
        token=token,
    )
    for err in state_errors:
        result.add_error(err)
    if state_errors:
        return result
    result.add_message("State rebuild complete")

    # Step 5: Update official_models.json
    if official_models_path and os.path.isfile(official_models_path):
        result.add_message("Step 5/5: Updating official_models.json...")
        om_errors = update_official_models_file(
            official_models_path, result.subset, result.new_models
        )
        for err in om_errors:
            result.add_error(err)
        if not om_errors:
            result.add_message("official_models.json updated")
    else:
        result.add_message("Step 5/5: Skipped (no official_models.json path)")

    return result
