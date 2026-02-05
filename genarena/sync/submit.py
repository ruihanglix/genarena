# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Submission functionality for GenArena.

This module provides the ability for users to submit their evaluation results
to the official leaderboard via GitHub PR.

Workflow:
1. Validate local submission data
2. Upload data to user's HuggingFace repository
3. Create submission metadata JSON
4. Fork official repo and create PR via GitHub CLI
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from genarena import __version__
from genarena.experiments import is_valid_exp_name
from genarena.logs import load_battle_records
from genarena.sync.packer import (
    TempPackingContext,
    pack_exp_dir,
    pack_directory,
    IMAGE_EXTENSIONS,
)

logger = logging.getLogger(__name__)

# Default official submissions repository
DEFAULT_OFFICIAL_REPO = "genarena/submissions"

# URL to fetch official models list
OFFICIAL_MODELS_URL = (
    "https://raw.githubusercontent.com/genarena/submissions/main/official_models.json"
)


@dataclass
class ValidationResult:
    """Result of local submission validation."""

    valid: bool
    exp_name: str
    subset: str
    models: list[str] = field(default_factory=list)
    new_models: list[str] = field(default_factory=list)
    existing_models: list[str] = field(default_factory=list)
    total_battles: int = 0
    battles_per_pair: dict[str, int] = field(default_factory=dict)
    elo_ratings: dict[str, float] = field(default_factory=dict)
    elo_ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    evaluation_config: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class UploadResult:
    """Result of HuggingFace upload."""

    hf_repo: str
    hf_revision: str
    models_zip_path: str
    models_zip_sha256: str
    models_zip_size: int
    pk_logs_zip_path: str
    pk_logs_zip_sha256: str
    pk_logs_zip_size: int


def fetch_official_models(subset: str, timeout: int = 10) -> set[str]:
    """
    Fetch official models list from GitHub.

    Args:
        subset: Subset name to get models for
        timeout: Request timeout in seconds

    Returns:
        Set of official model names for the subset
    """
    import urllib.request
    import urllib.error

    try:
        with urllib.request.urlopen(OFFICIAL_MODELS_URL, timeout=timeout) as resp:
            data = json.load(resp)
            return set(data.get("subsets", {}).get(subset, {}).get("models", []))
    except urllib.error.URLError as e:
        logger.warning(f"Failed to fetch official models list: {e}")
        return set()
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse official models list: {e}")
        return set()
    except Exception as e:
        logger.warning(f"Unexpected error fetching official models: {e}")
        return set()


def _load_experiment_config(exp_dir: str) -> dict[str, Any]:
    """Load experiment configuration from config.json."""
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def validate_local_submission(
    arena_dir: str,
    subset: str,
    exp_name: str,
    skip_official_check: bool = False,
) -> ValidationResult:
    """
    Validate local submission data.

    Checks:
    1. exp_name format (_yyyymmdd suffix)
    2. pk_logs directory exists and has battle records
    3. models directory exists and has model outputs
    4. All models in battles have corresponding outputs
    5. At least one model is new (not in official leaderboard)

    Args:
        arena_dir: Arena directory path
        subset: Subset name
        exp_name: Experiment name
        skip_official_check: Skip checking against official models (for testing)

    Returns:
        ValidationResult with validation status and details
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Check exp_name format
    if not is_valid_exp_name(exp_name):
        errors.append(
            f"Invalid exp_name format: '{exp_name}' must end with _yyyymmdd"
        )

    # Check paths exist
    pk_logs_dir = os.path.join(arena_dir, subset, "pk_logs")
    exp_dir = os.path.join(pk_logs_dir, exp_name)
    models_root = os.path.join(arena_dir, subset, "models")
    exp_models_dir = os.path.join(models_root, exp_name)

    if not os.path.isdir(exp_dir):
        errors.append(f"pk_logs directory not found: {exp_dir}")

    if not os.path.isdir(exp_models_dir):
        errors.append(f"models directory not found: {exp_models_dir}")

    if errors:
        return ValidationResult(
            valid=False,
            exp_name=exp_name,
            subset=subset,
            errors=errors,
            warnings=warnings,
        )

    # Load battle records
    records = load_battle_records(pk_logs_dir, exp_name=exp_name)
    if not records:
        errors.append("No battle records found in pk_logs")
        return ValidationResult(
            valid=False,
            exp_name=exp_name,
            subset=subset,
            errors=errors,
            warnings=warnings,
        )

    # Extract models and battle statistics
    models: set[str] = set()
    battles_per_pair: dict[str, int] = {}

    for r in records:
        model_a = r.get("model_a", "")
        model_b = r.get("model_b", "")
        if model_a and model_b:
            models.add(model_a)
            models.add(model_b)
            # Ensure consistent pair key (sorted)
            pair_key = f"{min(model_a, model_b)}_vs_{max(model_a, model_b)}"
            battles_per_pair[pair_key] = battles_per_pair.get(pair_key, 0) + 1

    models_list = sorted(models)

    # Check model outputs exist
    for model in models_list:
        model_dir = os.path.join(exp_models_dir, model)
        if not os.path.isdir(model_dir):
            errors.append(f"Model output directory not found: {model_dir}")
        else:
            # Check if there are any images
            has_images = False
            for f in os.listdir(model_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    has_images = True
                    break
            if not has_images:
                errors.append(f"No image files found in model directory: {model_dir}")

    # Check against official models
    if not skip_official_check:
        official_models = fetch_official_models(subset)
        new_models = [m for m in models_list if m not in official_models]
        existing_models = [m for m in models_list if m in official_models]

        if not new_models:
            errors.append(
                "No new models found. All models already exist in official leaderboard. "
                "Submissions must include at least one new model."
            )
    else:
        new_models = models_list
        existing_models = []
        warnings.append("Skipped official models check (--skip-official-check)")

    # Calculate ELO (only if no critical errors so far)
    elo_ratings: dict[str, float] = {}
    elo_ci: dict[str, tuple[float, float]] = {}

    if not errors:
        try:
            from genarena.bt_elo import compute_bootstrap_bt_elo

            battles = [
                (r["model_a"], r["model_b"], r["final_winner"])
                for r in records
                if r.get("model_a") and r.get("model_b") and r.get("final_winner")
            ]

            if battles:
                bt_result = compute_bootstrap_bt_elo(battles, num_bootstrap=100)
                elo_ratings = bt_result.ratings
                for model in models_list:
                    if model in bt_result.ci_lower and model in bt_result.ci_upper:
                        elo_ci[model] = (
                            bt_result.ci_lower[model],
                            bt_result.ci_upper[model],
                        )
        except Exception as e:
            warnings.append(f"Failed to calculate ELO: {e}")

    # Load evaluation config
    evaluation_config = _load_experiment_config(exp_dir)

    return ValidationResult(
        valid=len(errors) == 0,
        exp_name=exp_name,
        subset=subset,
        models=models_list,
        new_models=new_models,
        existing_models=existing_models,
        total_battles=len(records),
        battles_per_pair=battles_per_pair,
        elo_ratings=elo_ratings,
        elo_ci=elo_ci,
        evaluation_config=evaluation_config,
        errors=errors,
        warnings=warnings,
    )


def upload_submission_data(
    arena_dir: str,
    subset: str,
    exp_name: str,
    hf_repo: str,
    hf_revision: str = "main",
    show_progress: bool = True,
) -> UploadResult:
    """
    Pack and upload submission data to HuggingFace.

    Args:
        arena_dir: Arena directory path
        subset: Subset name
        exp_name: Experiment name
        hf_repo: HuggingFace repository ID (e.g., "username/repo-name")
        hf_revision: Repository revision/branch (default: "main")
        show_progress: Show upload progress

    Returns:
        UploadResult with upload details

    Raises:
        RuntimeError: If upload fails
    """
    from huggingface_hub import HfApi

    api = HfApi()

    # Paths
    exp_models_dir = os.path.join(arena_dir, subset, "models", exp_name)
    exp_dir = os.path.join(arena_dir, subset, "pk_logs", exp_name)

    with TempPackingContext() as ctx:
        # Pack models
        models_zip_path = ctx.get_temp_zip_path(f"{subset}/models/{exp_name}.zip")
        success, msg = pack_directory(
            exp_models_dir, models_zip_path, file_extensions=IMAGE_EXTENSIONS
        )
        if not success:
            raise RuntimeError(f"Failed to pack models: {msg}")

        # Calculate SHA256 for models
        with open(models_zip_path, "rb") as f:
            models_sha256 = hashlib.sha256(f.read()).hexdigest()
        models_size = os.path.getsize(models_zip_path)

        # Pack pk_logs
        logs_zip_path = ctx.get_temp_zip_path(f"{subset}/pk_logs/{exp_name}.zip")
        success, msg = pack_exp_dir(exp_dir, logs_zip_path)
        if not success:
            raise RuntimeError(f"Failed to pack pk_logs: {msg}")

        # Calculate SHA256 for logs
        with open(logs_zip_path, "rb") as f:
            logs_sha256 = hashlib.sha256(f.read()).hexdigest()
        logs_size = os.path.getsize(logs_zip_path)

        # Upload to HF
        hf_models_path = f"{subset}/models/{exp_name}.zip"
        hf_logs_path = f"{subset}/pk_logs/{exp_name}.zip"

        logger.info(f"Uploading models ZIP ({models_size / 1024 / 1024:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=models_zip_path,
            path_in_repo=hf_models_path,
            repo_id=hf_repo,
            repo_type="dataset",
            revision=hf_revision,
        )

        logger.info(f"Uploading pk_logs ZIP ({logs_size / 1024 / 1024:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=logs_zip_path,
            path_in_repo=hf_logs_path,
            repo_id=hf_repo,
            repo_type="dataset",
            revision=hf_revision,
        )

    return UploadResult(
        hf_repo=hf_repo,
        hf_revision=hf_revision,
        models_zip_path=hf_models_path,
        models_zip_sha256=models_sha256,
        models_zip_size=models_size,
        pk_logs_zip_path=hf_logs_path,
        pk_logs_zip_sha256=logs_sha256,
        pk_logs_zip_size=logs_size,
    )


def create_submission_metadata(
    validation: ValidationResult,
    upload: UploadResult,
    github_username: str,
    title: str = "",
    description: str = "",
    contact: str = "",
) -> dict[str, Any]:
    """
    Create submission metadata JSON.

    Args:
        validation: ValidationResult from validate_local_submission
        upload: UploadResult from upload_submission_data
        github_username: GitHub username of submitter
        title: Submission title
        description: Submission description
        contact: Optional contact email

    Returns:
        Submission metadata dictionary
    """
    # Generate submission ID
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    hash_input = f"{timestamp}{validation.exp_name}{github_username}"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:8]
    submission_id = f"sub_{timestamp}_{short_hash}"

    # Build submitter info
    submitter: dict[str, str] = {"github_username": github_username}
    if contact:
        submitter["contact"] = contact

    # Build evaluation config (extract key fields)
    eval_config = validation.evaluation_config
    evaluation_config_summary = {
        "judge_model": eval_config.get("judge_model", "unknown"),
        "prompt_module": eval_config.get("prompt", "unknown"),
        "temperature": eval_config.get("temperature", 0.0),
        "position_debiasing": True,  # Always true in genarena
    }

    # Build model pairs list
    model_pairs = [
        [min(p.split("_vs_")[0], p.split("_vs_")[1]),
         max(p.split("_vs_")[0], p.split("_vs_")[1])]
        for p in validation.battles_per_pair.keys()
    ]

    return {
        "schema_version": "1.0",
        "submission_id": submission_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "submitter": submitter,
        "experiment": {
            "exp_name": validation.exp_name,
            "subset": validation.subset,
            "models": validation.models,
            "new_models": validation.new_models,
            "existing_models": validation.existing_models,
            "model_pairs": model_pairs,
            "total_battles": validation.total_battles,
            "battles_per_pair": validation.battles_per_pair,
        },
        "data_location": {
            "hf_repo_id": upload.hf_repo,
            "hf_revision": upload.hf_revision,
            "files": {
                "models_zip": {
                    "path": upload.models_zip_path,
                    "sha256": upload.models_zip_sha256,
                    "size_bytes": upload.models_zip_size,
                },
                "pk_logs_zip": {
                    "path": upload.pk_logs_zip_path,
                    "sha256": upload.pk_logs_zip_sha256,
                    "size_bytes": upload.pk_logs_zip_size,
                },
            },
        },
        "elo_preview": {
            "ratings": validation.elo_ratings,
            "ci_95": {m: list(ci) for m, ci in validation.elo_ci.items()},
        },
        "evaluation_config": evaluation_config_summary,
        "title": title or f"Submit {validation.exp_name}",
        "description": description,
        "verification": {
            "local_validation_passed": validation.valid,
            "genarena_version": __version__,
        },
    }


def _get_github_username() -> Optional[str]:
    """Get GitHub username from gh CLI."""
    try:
        result = subprocess.run(
            ["gh", "api", "user", "-q", ".login"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _check_gh_cli() -> tuple[bool, str]:
    """Check if GitHub CLI is available and authenticated."""
    try:
        # Check if gh is installed
        result = subprocess.run(
            ["gh", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "GitHub CLI (gh) is not installed"

        # Check if authenticated
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return False, "GitHub CLI is not authenticated. Run 'gh auth login' first."

        return True, "GitHub CLI is ready"
    except FileNotFoundError:
        return False, "GitHub CLI (gh) is not installed. Install it from https://cli.github.com"
    except subprocess.TimeoutExpired:
        return False, "GitHub CLI timed out"


def _generate_pr_body(submission: dict[str, Any]) -> str:
    """Generate PR description body."""
    exp = submission["experiment"]
    elo = submission["elo_preview"]["ratings"]
    eval_config = submission["evaluation_config"]

    body = f"""## Submission Details

**Experiment:** `{exp['exp_name']}`
**Subset:** `{exp['subset']}`
**New Models:** {', '.join(f'`{m}`' for m in exp['new_models']) or 'None'}
**Total Battles:** {exp['total_battles']:,}
**Model Pairs:** {len(exp['model_pairs'])}

### Evaluation Configuration

| Setting | Value |
|---------|-------|
| Judge Model | `{eval_config.get('judge_model', 'N/A')}` |
| Prompt Module | `{eval_config.get('prompt_module', 'N/A')}` |
| Temperature | {eval_config.get('temperature', 'N/A')} |
| Position Debiasing | {'Yes' if eval_config.get('position_debiasing') else 'No'} |

### ELO Preview

| Model | ELO | 95% CI |
|-------|-----|--------|
"""
    ci_data = submission["elo_preview"].get("ci_95", {})
    for model in sorted(elo.keys(), key=lambda m: -elo[m]):
        ci = ci_data.get(model, [None, None])
        ci_str = f"[{ci[0]:.1f}, {ci[1]:.1f}]" if ci[0] is not None else "N/A"
        body += f"| {model} | {elo[model]:.1f} | {ci_str} |\n"

    body += f"""
### Data Location

- **HuggingFace Repo:** `{submission['data_location']['hf_repo_id']}`
- **Models ZIP:** `{submission['data_location']['files']['models_zip']['path']}`
  - SHA256: `{submission['data_location']['files']['models_zip']['sha256'][:16]}...`
  - Size: {submission['data_location']['files']['models_zip']['size_bytes'] / 1024 / 1024:.1f} MB
- **Logs ZIP:** `{submission['data_location']['files']['pk_logs_zip']['path']}`
  - SHA256: `{submission['data_location']['files']['pk_logs_zip']['sha256'][:16]}...`
  - Size: {submission['data_location']['files']['pk_logs_zip']['size_bytes'] / 1024:.1f} KB

### Description

{submission.get('description') or submission.get('title', 'No description provided.')}

---
*Submitted via genarena v{submission['verification']['genarena_version']}*
"""
    return body


def create_submission_pr(
    submission: dict[str, Any],
    official_repo: str = DEFAULT_OFFICIAL_REPO,
    title: Optional[str] = None,
) -> str:
    """
    Fork official repo and create PR with submission.

    Args:
        submission: Submission metadata dictionary
        official_repo: Official submissions repository (default: genarena/submissions)
        title: PR title (optional, auto-generated if not provided)

    Returns:
        PR URL

    Raises:
        RuntimeError: If PR creation fails
    """
    submission_id = submission["submission_id"]
    filename = f"{submission_id}.json"

    # Get GitHub username
    gh_username = _get_github_username()
    if not gh_username:
        raise RuntimeError("Failed to get GitHub username. Ensure gh CLI is authenticated.")

    # Fork the repo (idempotent - won't fail if already forked)
    logger.info(f"Forking {official_repo}...")
    result = subprocess.run(
        ["gh", "repo", "fork", official_repo, "--clone=false"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Note: fork may "fail" if already forked, but that's OK

    # Clone forked repo to temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        fork_repo = f"{gh_username}/submissions"
        logger.info(f"Cloning {fork_repo}...")

        result = subprocess.run(
            ["gh", "repo", "clone", fork_repo, tmpdir],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to clone fork: {result.stderr}")

        # Sync with upstream
        logger.info("Syncing with upstream...")
        subprocess.run(
            ["gh", "repo", "sync", fork_repo, "--source", official_repo],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Pull latest changes
        subprocess.run(
            ["git", "pull", "origin", "main"],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Create branch
        branch_name = f"submit/{submission_id}"
        logger.info(f"Creating branch {branch_name}...")
        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to create branch: {result.stderr}")

        # Write submission file
        submissions_dir = os.path.join(tmpdir, "submissions", "pending")
        os.makedirs(submissions_dir, exist_ok=True)
        submission_path = os.path.join(submissions_dir, filename)

        with open(submission_path, "w", encoding="utf-8") as f:
            json.dump(submission, f, indent=2, ensure_ascii=False)

        # Commit
        logger.info("Committing submission...")
        subprocess.run(["git", "add", "."], cwd=tmpdir, check=True)

        commit_msg = title or f"Submit {submission['experiment']['exp_name']}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=tmpdir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to commit: {result.stderr}")

        # Push
        logger.info("Pushing to fork...")
        result = subprocess.run(
            ["git", "push", "-u", "origin", branch_name],
            cwd=tmpdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to push: {result.stderr}")

        # Create PR
        logger.info("Creating PR...")
        pr_title = title or f"[Submission] {submission['experiment']['exp_name']}"
        pr_body = _generate_pr_body(submission)

        result = subprocess.run(
            [
                "gh", "pr", "create",
                "--repo", official_repo,
                "--head", f"{gh_username}:{branch_name}",
                "--title", pr_title,
                "--body", pr_body,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create PR: {result.stderr}")

        pr_url = result.stdout.strip()
        return pr_url


def print_validation_summary(validation: ValidationResult) -> None:
    """Print validation summary to console."""
    print("\nValidation Results:")
    print("-" * 40)

    if validation.valid:
        print("Status: PASSED")
    else:
        print("Status: FAILED")

    print(f"\nExperiment: {validation.exp_name}")
    print(f"Subset: {validation.subset}")
    print(f"Models: {len(validation.models)}")
    print(f"  New models: {', '.join(validation.new_models) or 'None'}")
    print(f"  Existing models: {', '.join(validation.existing_models) or 'None'}")
    print(f"Total battles: {validation.total_battles:,}")
    print(f"Model pairs: {len(validation.battles_per_pair)}")

    if validation.elo_ratings:
        print("\nELO Preview:")
        sorted_models = sorted(
            validation.elo_ratings.keys(),
            key=lambda m: -validation.elo_ratings[m]
        )
        for model in sorted_models:
            elo = validation.elo_ratings[model]
            ci = validation.elo_ci.get(model)
            ci_str = f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
            new_marker = " (new)" if model in validation.new_models else ""
            print(f"  {model}: {elo:.1f}{ci_str}{new_marker}")

    if validation.evaluation_config:
        config = validation.evaluation_config
        print("\nEvaluation Config:")
        print(f"  Judge model: {config.get('judge_model', 'N/A')}")
        print(f"  Prompt: {config.get('prompt', 'N/A')}")
        print(f"  Temperature: {config.get('temperature', 'N/A')}")

    if validation.warnings:
        print("\nWarnings:")
        for w in validation.warnings:
            print(f"  - {w}")

    if validation.errors:
        print("\nErrors:")
        for e in validation.errors:
            print(f"  - {e}")

    print()


def generate_official_models_json(
    arena_dir: str,
    output_path: Optional[str] = None,
) -> dict[str, Any]:
    """
    Generate official_models.json from arena state files.

    This function scans all subsets in the arena directory and extracts
    the list of models from each subset's state.json file.

    Args:
        arena_dir: Path to the official arena directory
        output_path: Optional path to write the JSON file

    Returns:
        The official_models.json content as a dictionary
    """
    from genarena.sync.packer import discover_subsets
    from genarena.state import load_state

    result: dict[str, Any] = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "description": "List of models currently on the official GenArena leaderboard",
        "subsets": {},
    }

    # Discover all subsets
    subsets = discover_subsets(arena_dir)

    for subset in subsets:
        state_path = os.path.join(arena_dir, subset, "arena", "state.json")
        if not os.path.isfile(state_path):
            continue

        state = load_state(state_path)
        if not state.models:
            continue

        # Get sorted list of model names
        models = sorted(state.models.keys())

        result["subsets"][subset] = {
            "models": models,
            "model_count": len(models),
            "total_battles": state.total_battles,
        }

    # Write to file if output_path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote official_models.json to {output_path}")

    return result


def print_official_models_summary(data: dict[str, Any]) -> None:
    """Print summary of official models."""
    print("\n=== Official Models ===\n")
    print(f"Last Updated: {data.get('last_updated', 'N/A')}")
    print()

    subsets = data.get("subsets", {})
    if not subsets:
        print("No subsets found.")
        return

    for subset, info in sorted(subsets.items()):
        models = info.get("models", [])
        print(f"Subset: {subset}")
        print(f"  Models ({len(models)}):")
        for model in models:
            print(f"    - {model}")
        print(f"  Total Battles: {info.get('total_battles', 0):,}")
        print()
