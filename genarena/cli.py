# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""CLI entry point for genarena."""

import argparse
import json
import logging
import os
import sys
from typing import Optional

from genarena import __version__
from genarena.arena import Arena, ArenaConfig, get_all_subsets_status
from genarena.data import discover_subsets
from genarena.leaderboard import print_leaderboard
from genarena.sampling import SamplingConfig
from genarena.state import load_state


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging.

    Args:
        verbose: If True, enable DEBUG level logging
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_models(models_str: Optional[str]) -> Optional[list[str]]:
    """
    Parse comma-separated model names.

    Args:
        models_str: Comma-separated model names or None

    Returns:
        List of model names or None
    """
    if not models_str:
        return None
    return [m.strip() for m in models_str.split(",") if m.strip()]


def cmd_run(args: argparse.Namespace) -> int:
    """
    Execute the 'run' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Discover subsets if not specified
    if args.subset:
        subsets = [args.subset]
    else:
        subsets = discover_subsets(args.data_dir)
        if not subsets:
            logger.error(f"No subsets found in {args.data_dir}")
            return 1
        logger.info(f"Discovered subsets: {subsets}")

    # Parse models
    models = parse_models(args.models)

    # Build sampling configuration
    if args.sampling_mode == "full":
        sampling_config = SamplingConfig.full_mode(sample_size=args.sample_size)
    else:
        sampling_config = SamplingConfig.adaptive_mode(
            target_ci_width=args.target_ci_width,
            min_samples=args.min_samples,
            max_samples=args.max_samples,
        )
        # Override milestone_min_samples if provided
        if args.milestone_min_samples:
            sampling_config.milestone_min_samples = args.milestone_min_samples

    # Run arena for each subset
    for subset in subsets:
        logger.info(f"Running arena for subset: {subset}")

        config = ArenaConfig(
            arena_dir=args.arena_dir,
            data_dir=args.data_dir,
            subset=subset,
            models=models,
            exp_name=args.exp_name,
            sample_size=args.sample_size,
            num_threads=args.num_threads,
            num_processes=args.num_processes,
            parallel_swap_calls=args.parallel_swap_calls,
            enable_progress_bar=args.enable_progress_bar,
            sampling=sampling_config,
            judge_model=args.judge_model,
            temperature=args.temperature,
            prompt=args.prompt,
            timeout=args.timeout,
            max_retries=args.max_retries,
            base_urls=args.base_urls,
            api_keys=args.api_keys,
            enable_audit_log=not args.no_audit_log,
            clean_orphaned_logs=not args.no_clean_orphaned_logs,
            verbose=args.verbose
        )

        try:
            arena = Arena(config)
            state = arena.run()
            arena.update_leaderboard()

            logger.info(
                f"Subset {subset} completed: "
                f"{state.total_battles} total battles, "
                f"{len(state.models)} models"
            )
        except Exception as e:
            logger.error(f"Error running arena for subset {subset}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

    # Auto commit and push if Git is initialized
    from genarena.sync.auto_commit import auto_commit_and_push
    auto_commit_and_push(args.arena_dir, "run")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """
    Execute the 'status' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)

    # Get status for all subsets
    statuses = get_all_subsets_status(args.arena_dir, args.data_dir)

    if not statuses:
        print("No subsets found.")
        return 0

    print("\n=== Arena Status ===\n")
    print(f"Arena Directory: {args.arena_dir}")
    print(f"Data Directory: {args.data_dir}")
    print()

    for status in statuses:
        print(f"Subset: {status['subset']}")
        print(f"  Models: {status['total_models']} ({', '.join(status['models'][:3])}{'...' if len(status['models']) > 3 else ''})")
        print(f"  Total Battles: {status['total_battles']}")
        print(f"  Last Updated: {status['last_updated'] or 'Never'}")
        print()

    return 0


def cmd_leaderboard(args: argparse.Namespace) -> int:
    """
    Execute the 'leaderboard' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)

    # Load state
    state_path = os.path.join(args.arena_dir, args.subset, "arena", "state.json")

    if not os.path.isfile(state_path):
        print(f"No arena state found for subset '{args.subset}'.")
        print("Run battles first with: genarena run --arena_dir <path> --data_dir <path>")
        return 1

    state = load_state(state_path)

    if not state.models:
        print(f"No battles recorded for subset '{args.subset}'.")
        return 0

    # Print leaderboard
    title = f"{args.subset.capitalize()} Leaderboard"
    print_leaderboard(state, title)

    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """
    Execute the 'serve' subcommand to start the visualization server.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    # Import here to avoid loading Flask when not needed
    from genarena.visualize.app import run_server

    # Validate directories
    if not os.path.isdir(args.arena_dir):
        print(f"Error: Arena directory does not exist: {args.arena_dir}")
        return 1

    if not os.path.isdir(args.data_dir):
        print(f"Error: Data directory does not exist: {args.data_dir}")
        return 1

    # Start server
    run_server(
        arena_dir=args.arena_dir,
        data_dir=args.data_dir,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    return 0


# === init command ===

def cmd_init(args: argparse.Namespace) -> int:
    """
    Execute the 'init' subcommand.

    Downloads benchmark data and arena data from official repositories.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    from genarena.sync.init_ops import init_arena

    # Validate mutually exclusive options
    data_only = getattr(args, "data_only", False)
    arena_only = getattr(args, "arena_only", False)

    if data_only and arena_only:
        print("Error: --data-only and --arena-only cannot be used together")
        return 1

    # Parse subsets
    subsets = parse_models(args.subsets) if args.subsets else None

    print("=== GenArena Init ===\n")

    success, msg = init_arena(
        arena_dir=args.arena_dir,
        data_dir=args.data_dir,
        subsets=subsets,
        benchmark_repo=args.benchmark_repo,
        arena_repo=args.arena_repo,
        revision=args.revision,
        overwrite=args.overwrite if hasattr(args, "overwrite") else False,
        init_git=args.git if hasattr(args, "git") else False,
        data_only=data_only,
        arena_only=arena_only,
        show_progress=True,
    )

    print(msg)
    return 0 if success else 1


# === Git commands ===

def cmd_git_init(args: argparse.Namespace) -> int:
    """
    Execute the 'git init' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.git_ops import git_init, is_git_initialized

    if is_git_initialized(args.arena_dir):
        print(f"Git repository already initialized in {args.arena_dir}")
        return 0

    success, msg = git_init(args.arena_dir)
    print(msg)
    return 0 if success else 1


def cmd_git_commit(args: argparse.Namespace) -> int:
    """
    Execute the 'git commit' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.git_ops import git_commit, is_git_initialized

    if not is_git_initialized(args.arena_dir):
        print(f"Error: Git repository not initialized in {args.arena_dir}")
        print("Run 'genarena git init --arena_dir <path>' first.")
        return 1

    message = args.message if hasattr(args, "message") and args.message else None
    success, msg = git_commit(args.arena_dir, message=message)
    print(msg)
    return 0 if success else 1


def cmd_git_remote(args: argparse.Namespace) -> int:
    """
    Execute the 'git remote' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.git_ops import git_remote_add, git_remote_get_url, is_git_initialized

    if not is_git_initialized(args.arena_dir):
        print(f"Error: Git repository not initialized in {args.arena_dir}")
        print("Run 'genarena git init --arena_dir <path>' first.")
        return 1

    # If no URL provided, just show current remote
    if not args.url:
        url = git_remote_get_url(args.arena_dir)
        if url:
            print(f"Remote 'origin': {url}")
        else:
            print("No remote configured.")
        return 0

    force = args.force if hasattr(args, "force") else False
    success, msg = git_remote_add(args.arena_dir, args.url, force=force)
    print(msg)
    return 0 if success else 1


def cmd_git_push(args: argparse.Namespace) -> int:
    """
    Execute the 'git push' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.git_ops import git_push, is_git_initialized

    if not is_git_initialized(args.arena_dir):
        print(f"Error: Git repository not initialized in {args.arena_dir}")
        print("Run 'genarena git init --arena_dir <path>' first.")
        return 1

    success, msg = git_push(args.arena_dir)
    print(msg)
    return 0 if success else 1


def cmd_git_sync(args: argparse.Namespace) -> int:
    """
    Execute the 'git sync' subcommand (commit + push).

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.git_ops import git_sync, is_git_initialized

    if not is_git_initialized(args.arena_dir):
        print(f"Error: Git repository not initialized in {args.arena_dir}")
        print("Run 'genarena git init --arena_dir <path>' first.")
        return 1

    success, msg = git_sync(args.arena_dir)
    print(msg)
    return 0 if success else 1


# === Huggingface commands ===

def cmd_hf_upload(args: argparse.Namespace) -> int:
    """
    Execute the 'hf upload' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    from genarena.sync.hf_ops import upload_arena_data

    # Parse filters
    subsets = parse_models(args.subsets) if hasattr(args, "subsets") and args.subsets else None
    models = parse_models(args.models) if hasattr(args, "models") and args.models else None
    experiments = parse_models(args.experiments) if hasattr(args, "experiments") and args.experiments else None
    overwrite = args.overwrite if hasattr(args, "overwrite") else False
    max_retries = getattr(args, "max_retries", 3)

    print(f"Uploading arena data to {args.repo_id}...")
    if subsets:
        print(f"  Subsets: {', '.join(subsets)}")
    if models:
        print(f"  Models: {', '.join(models)}")
    if experiments:
        print(f"  Experiments: {', '.join(experiments)}")
    print(f"  Max retries per file: {max_retries}")
    print()

    success, msg = upload_arena_data(
        arena_dir=args.arena_dir,
        repo_id=args.repo_id,
        subsets=subsets,
        models=models,
        experiments=experiments,
        overwrite=overwrite,
        show_progress=True,
        max_retries=max_retries,
    )

    print()
    print(msg)
    return 0 if success else 1


def cmd_hf_pull(args: argparse.Namespace) -> int:
    """
    Execute the 'hf pull' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    from genarena.sync.hf_ops import pull_arena_data

    # Parse filters
    subsets = parse_models(args.subsets) if hasattr(args, "subsets") and args.subsets else None
    models = parse_models(args.models) if hasattr(args, "models") and args.models else None
    experiments = parse_models(args.experiments) if hasattr(args, "experiments") and args.experiments else None
    overwrite = args.overwrite if hasattr(args, "overwrite") else False
    revision = args.revision if hasattr(args, "revision") and args.revision else "main"

    print(f"Pulling arena data from {args.repo_id} (revision: {revision})...")
    if subsets:
        print(f"  Subsets: {', '.join(subsets)}")
    if models:
        print(f"  Models: {', '.join(models)}")
    if experiments:
        print(f"  Experiments: {', '.join(experiments)}")
    print()

    success, msg = pull_arena_data(
        arena_dir=args.arena_dir,
        repo_id=args.repo_id,
        subsets=subsets,
        models=models,
        experiments=experiments,
        revision=revision,
        overwrite=overwrite,
        show_progress=True,
    )

    print()
    print(msg)
    return 0 if success else 1


def cmd_hf_list(args: argparse.Namespace) -> int:
    """
    Execute the 'hf list' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.hf_ops import list_repo_contents

    revision = args.revision if hasattr(args, "revision") and args.revision else "main"

    success, output = list_repo_contents(
        repo_id=args.repo_id,
        revision=revision,
    )

    print(output)
    return 0 if success else 1


# === submit command ===

def cmd_submit(args: argparse.Namespace) -> int:
    """
    Execute the 'submit' subcommand.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, "verbose") else False)
    logger = logging.getLogger(__name__)

    from genarena.sync.submit import (
        validate_local_submission,
        upload_submission_data,
        create_submission_metadata,
        create_submission_pr,
        print_validation_summary,
        _check_gh_cli,
        _get_github_username,
        DEFAULT_OFFICIAL_REPO,
    )

    # Check GitHub CLI (unless dry-run)
    if not args.dry_run:
        gh_ok, gh_msg = _check_gh_cli()
        if not gh_ok:
            print(f"Error: {gh_msg}")
            return 1

    print("=== GenArena Submission ===\n")

    # Step 1: Validate local data
    print("Validating local data...")
    validation = validate_local_submission(
        arena_dir=args.arena_dir,
        subset=args.subset,
        exp_name=args.exp_name,
        skip_official_check=args.skip_official_check if hasattr(args, "skip_official_check") else False,
    )

    print_validation_summary(validation)

    if not validation.valid:
        print("Validation failed. Please fix the errors above.")
        return 1

    # Dry run - stop here
    if args.dry_run:
        print("Dry run complete. No data was uploaded or PR created.")
        return 0

    # Confirm before proceeding
    if not args.yes:
        confirm = input("Proceed with submission? [y/N] ")
        if confirm.lower() != "y":
            print("Submission cancelled.")
            return 0

    # Step 2: Upload to HuggingFace
    print(f"\nUploading to HuggingFace ({args.hf_repo})...")
    try:
        upload = upload_submission_data(
            arena_dir=args.arena_dir,
            subset=args.subset,
            exp_name=args.exp_name,
            hf_repo=args.hf_repo,
        )
        print(f"  Models ZIP: {upload.models_zip_size / 1024 / 1024:.1f} MB uploaded")
        print(f"  Logs ZIP: {upload.pk_logs_zip_size / 1024:.1f} KB uploaded")
    except Exception as e:
        print(f"Error uploading to HuggingFace: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # Step 3: Create submission metadata
    print("\nCreating submission metadata...")
    gh_username = _get_github_username()
    if not gh_username:
        print("Error: Failed to get GitHub username")
        return 1

    submission = create_submission_metadata(
        validation=validation,
        upload=upload,
        github_username=gh_username,
        title=args.title if hasattr(args, "title") and args.title else "",
        description=args.description if hasattr(args, "description") and args.description else "",
    )
    print(f"  Submission ID: {submission['submission_id']}")

    # Step 4: Create PR
    print("\nCreating PR...")
    official_repo = args.official_repo if hasattr(args, "official_repo") and args.official_repo else DEFAULT_OFFICIAL_REPO
    try:
        pr_url = create_submission_pr(
            submission=submission,
            official_repo=official_repo,
            title=args.title if hasattr(args, "title") else None,
        )
        print(f"\nSubmission created successfully!")
        print(f"PR URL: {pr_url}")
        print("\nNext steps:")
        print("  1. The validation bot will automatically check your submission")
        print("  2. A maintainer will review and merge if approved")
        print("  3. Your models will appear on the official leaderboard after integration")
    except Exception as e:
        print(f"Error creating PR: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    return 0


def cmd_export_models(args: argparse.Namespace) -> int:
    """
    Execute the 'export-models' subcommand.

    Generates official_models.json from arena state.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    from genarena.sync.submit import (
        generate_official_models_json,
        print_official_models_summary,
    )

    # Generate official_models.json
    output_path = args.output if hasattr(args, "output") and args.output else None

    try:
        data = generate_official_models_json(
            arena_dir=args.arena_dir,
            output_path=output_path,
        )

        # Print summary
        print_official_models_summary(data)

        if output_path:
            print(f"Wrote to: {output_path}")
        else:
            # Print JSON to stdout if no output file specified
            print("\n--- JSON Output ---\n")
            print(json.dumps(data, indent=2))

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


# === deploy commands ===

def cmd_deploy_upload(args: argparse.Namespace) -> int:
    """
    Execute the 'deploy upload' subcommand.

    Uploads arena data to HuggingFace for Spaces deployment.
    Parquet benchmark data is downloaded from rhli/genarena during Docker build.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    setup_logging(args.verbose if hasattr(args, "verbose") else False)

    from genarena.sync.deploy_ops import upload_for_deploy, DEFAULT_NUM_WORKERS, DEFAULT_WORKER_TIMEOUT

    subsets = parse_models(args.subsets) if args.subsets else None
    max_retries = getattr(args, "max_retries", 3)
    num_workers = getattr(args, "num_workers", DEFAULT_NUM_WORKERS)
    worker_timeout = getattr(args, "timeout", DEFAULT_WORKER_TIMEOUT)

    print("=" * 60)
    print("  GenArena HuggingFace Spaces Deployment")
    print("=" * 60)
    print(f"  Arena dir:   {args.arena_dir}")
    print(f"  Arena repo:  {args.arena_repo}")
    print(f"  Space repo:  {args.space_repo}")
    if subsets:
        print(f"  Subsets:     {', '.join(subsets)}")
    print(f"  Mode:        {'overwrite' if args.overwrite else 'incremental'}")
    print(f"  Max retries: {max_retries}")
    print(f"  Workers:     {num_workers}")
    print(f"  Timeout:     {worker_timeout}s per worker")
    print()
    print("  Note: Parquet data is downloaded from rhli/genarena during build")
    print("=" * 60)
    print()

    success, msg = upload_for_deploy(
        arena_dir=args.arena_dir,
        arena_repo=args.arena_repo,
        space_repo=args.space_repo,
        subsets=subsets,
        overwrite=args.overwrite,
        show_progress=True,
        max_retries=max_retries,
        num_workers=num_workers,
        worker_timeout=worker_timeout,
    )

    print()
    print("=" * 60)
    if success:
        print("  Deployment upload completed successfully!")
    else:
        print("  Deployment upload failed!")
    print("=" * 60)
    print()
    print(msg)

    if success:
        print()
        print("Next steps:")
        print(f"  1. Go to https://huggingface.co/spaces/{args.space_repo}")
        print("  2. Set environment variable: HF_ARENA_REPO=" + args.arena_repo)
        print("  3. The Space should automatically rebuild and deploy")

    return 0 if success else 1


def cmd_deploy_info(args: argparse.Namespace) -> int:
    """
    Execute the 'deploy info' subcommand.

    Shows deployment configuration and instructions.

    Args:
        args: Parsed arguments

    Returns:
        Exit code (0 for success)
    """
    print("""
GenArena HuggingFace Spaces Deployment
======================================

OVERVIEW
--------
Deploy GenArena Explorer to HuggingFace Spaces for public access.

Data is split across repositories:
  - Space repo: Contains Dockerfile, app code, and genarena package
  - Arena repo: Contains battle logs and model output images (served via CDN)
  - Data repo (rhli/genarena): Parquet benchmark data (downloaded during build)

PREREQUISITES
-------------
1. Create a HuggingFace account and get an API token
2. Set the HF_TOKEN environment variable:
   export HF_TOKEN='your_token_here'

3. Create two repositories on HuggingFace:
   - A Dataset repository for arena data (e.g., 'your-org/genarena-arena')
   - A Space repository for deployment (e.g., 'your-org/genarena-explorer')
     Select "Docker" as the SDK when creating the Space

USAGE
-----
Upload all data for deployment:

  genarena deploy upload \\
    --arena_dir ./arena \\
    --arena_repo your-org/genarena-arena \\
    --space_repo your-org/genarena-explorer

Options:
  --subsets       Comma-separated list of subsets to upload (default: all)
  --overwrite     Overwrite existing files (default: incremental upload)
  --max-retries   Max retry attempts per file (default: 3)

INCREMENTAL UPLOADS
-------------------
By default, the upload is incremental - existing files are skipped.
This allows you to resume interrupted uploads or add new data.

Use --overwrite to force re-upload of all files.

ENVIRONMENT VARIABLES
---------------------
The Space uses these environment variables:
  - HF_ARENA_REPO: Arena data repository (required)
    Set this in the Space settings after deployment.
  - HF_DATA_REPO: Parquet data repository (default: rhli/genarena)
    Override if using a different benchmark dataset.

DIRECTORY STRUCTURE
-------------------
Space repo:
  ├── Dockerfile           # Docker build file (downloads rhli/genarena)
  ├── README.md            # Space configuration
  ├── genarena/            # Python package
  │   └── deploy/app.py    # Startup script
  └── pyproject.toml       # Package config

Arena repo (Dataset):
  └── <subset>/
      ├── pk_logs/         # Battle logs
      ├── models/          # Model output images
      └── arena/           # State files

Data repo (rhli/genarena - downloaded at build time):
  └── <subset>/
      └── data-*.parquet   # Benchmark prompts

For more information, see:
  https://github.com/genarena/genarena
""")
    return 0


def create_parser() -> argparse.ArgumentParser:
    """
    Create the argument parser.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="genarena",
        description="GenArena Arena Evaluation - VLM-based pairwise image generation evaluation"
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # === run command ===
    run_parser = subparsers.add_parser(
        "run",
        help="Run pairwise evaluation battles"
    )

    # Required arguments
    run_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    run_parser.add_argument(
        "--data_dir",
        required=True,
        help="Parquet dataset directory path"
    )

    # Optional arguments
    run_parser.add_argument(
        "--subset",
        default=None,
        help="Subset name (default: process all subsets)"
    )
    run_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to include (default: all)"
    )
    run_parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples per model pair (default: all)"
    )
    run_parser.add_argument(
        "--num_threads",
        type=int,
        default=8,
        help="Number of parallel threads (default: 8)"
    )
    run_parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1). When >1, work is sharded by parquet file."
    )
    run_parser.add_argument(
        "--parallel_swap_calls",
        action="store_true",
        help="Run original+swapped VLM calls in parallel within a battle (may increase 429 risk)."
    )
    run_parser.add_argument(
        "--enable_progress_bar",
        action="store_true",
        help="Show a progress bar by VLM API call count (total may be unknown and grow dynamically). "
             "When enabled, noisy httpx/httpcore request logs are silenced."
    )
    run_parser.add_argument(
        "--judge_model",
        default="Qwen/Qwen3-VL-32B-Instruct-FP8",
        help="VLM judge model name (default: Qwen/Qwen3-VL-32B-Instruct-FP8)"
    )
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="VLM temperature (default: 0 for greedy mode)"
    )
    run_parser.add_argument(
        "--prompt",
        default="mmrb2",
        help="Prompt module name (default: mmrb2)"
    )
    run_parser.add_argument(
        "--exp_name",
        default=None,
        help="Experiment name (must end with `_yyyymmdd`). "
             "If omitted, uses the latest `models/<exp_name>` by date suffix."
    )
    run_parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="API timeout in seconds (default: 120)"
    )
    run_parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)"
    )
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    run_parser.add_argument(
        "--no-audit-log",
        action="store_true",
        help="Disable audit logging (raw VLM outputs)"
    )
    run_parser.add_argument(
        "--base_urls",
        default=None,
        help="Comma-separated VLM API base URLs for multi-endpoint support (default: from OPENAI_BASE_URL(S) env)"
    )
    run_parser.add_argument(
        "--api_keys",
        default=None,
        help="Comma-separated API keys for multi-endpoint support (default: from OPENAI_API_KEY env)"
    )
    run_parser.add_argument(
        "--no-clean-orphaned-logs",
        action="store_true",
        help="Disable auto-deletion of battle logs for removed models (default: enabled)"
    )

    # Sampling configuration
    run_parser.add_argument(
        "--sampling_mode",
        choices=["adaptive", "full"],
        default="adaptive",
        help="Sampling mode: 'adaptive' (CI-based, default) or 'full' (all samples)"
    )
    run_parser.add_argument(
        "--target_ci_width",
        type=float,
        default=15.0,
        help="Target 95%% CI width for adaptive mode (default: 15.0, i.e., ±7.5 Elo)"
    )
    run_parser.add_argument(
        "--min_samples",
        type=int,
        default=100,
        help="Minimum samples per pair before checking CI in adaptive mode (default: 100)"
    )
    run_parser.add_argument(
        "--max_samples",
        type=int,
        default=1500,
        help="Maximum samples per pair in adaptive mode (default: 1500)"
    )
    run_parser.add_argument(
        "--milestone_min_samples",
        type=int,
        default=None,
        help="Minimum samples per pair for milestone experiments (default: 1000)"
    )

    run_parser.set_defaults(func=cmd_run)

    # === init command ===
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize arena and download data from official repositories"
    )

    init_parser.add_argument(
        "--arena_dir",
        default="./arena",
        help="Arena directory path (default: ./arena)"
    )
    init_parser.add_argument(
        "--data_dir",
        default="./data",
        help="Benchmark data directory path (default: ./data)"
    )
    init_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated list of subsets to download (default: all available)"
    )
    init_parser.add_argument(
        "--git",
        action="store_true",
        help="Initialize Git repository in arena_dir after downloading"
    )
    init_parser.add_argument(
        "--data-only",
        action="store_true",
        dest="data_only",
        help="Only download benchmark Parquet data (skip arena data)"
    )
    init_parser.add_argument(
        "--arena-only",
        action="store_true",
        dest="arena_only",
        help="Only download arena data (skip benchmark data)"
    )
    init_parser.add_argument(
        "--benchmark-repo",
        default="rhli/genarena",
        dest="benchmark_repo",
        help="HuggingFace repository for benchmark data (default: rhli/genarena)"
    )
    init_parser.add_argument(
        "--arena-repo",
        default="rhli/genarena-battlefield",
        dest="arena_repo",
        help="HuggingFace repository for arena data (default: rhli/genarena-battlefield)"
    )
    init_parser.add_argument(
        "--revision",
        default="main",
        help="HuggingFace revision/branch (default: main)"
    )
    init_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )
    init_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    init_parser.set_defaults(func=cmd_init)

    # === status command ===
    status_parser = subparsers.add_parser(
        "status",
        help="Show arena status summary"
    )

    status_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    status_parser.add_argument(
        "--data_dir",
        required=True,
        help="Parquet dataset directory path"
    )
    status_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    status_parser.set_defaults(func=cmd_status)

    # === leaderboard command ===
    lb_parser = subparsers.add_parser(
        "leaderboard",
        help="Display leaderboard for a subset"
    )

    lb_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    lb_parser.add_argument(
        "--subset",
        required=True,
        help="Subset name to display leaderboard for"
    )
    lb_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    lb_parser.set_defaults(func=cmd_leaderboard)

    # === serve command ===
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the battle visualization web server"
    )

    serve_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    serve_parser.add_argument(
        "--data_dir",
        required=True,
        help="Parquet dataset directory path"
    )
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    serve_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode"
    )

    serve_parser.set_defaults(func=cmd_serve)

    # === git command group ===
    git_parser = subparsers.add_parser(
        "git",
        help="Git version control commands"
    )
    git_subparsers = git_parser.add_subparsers(dest="git_command", help="Git subcommands")

    # git init
    git_init_parser = git_subparsers.add_parser(
        "init",
        help="Initialize Git repository for arena directory"
    )
    git_init_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    git_init_parser.set_defaults(func=cmd_git_init)

    # git commit
    git_commit_parser = git_subparsers.add_parser(
        "commit",
        help="Commit changes to local Git repository"
    )
    git_commit_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    git_commit_parser.add_argument(
        "--message", "-m",
        default=None,
        help="Custom commit message (default: auto-generated)"
    )
    git_commit_parser.set_defaults(func=cmd_git_commit)

    # git remote
    git_remote_parser = git_subparsers.add_parser(
        "remote",
        help="Configure remote repository"
    )
    git_remote_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    git_remote_parser.add_argument(
        "--url",
        default=None,
        help="Remote repository URL"
    )
    git_remote_parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force overwrite existing remote URL"
    )
    git_remote_parser.set_defaults(func=cmd_git_remote)

    # git push
    git_push_parser = git_subparsers.add_parser(
        "push",
        help="Push commits to remote repository"
    )
    git_push_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    git_push_parser.set_defaults(func=cmd_git_push)

    # git sync
    git_sync_parser = git_subparsers.add_parser(
        "sync",
        help="Commit all changes and push to remote (one-click sync)"
    )
    git_sync_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    git_sync_parser.set_defaults(func=cmd_git_sync)

    # === hf command group ===
    hf_parser = subparsers.add_parser(
        "hf",
        help="Huggingface Dataset repository commands"
    )
    hf_subparsers = hf_parser.add_subparsers(dest="hf_command", help="Huggingface subcommands")

    # hf upload
    hf_upload_parser = hf_subparsers.add_parser(
        "upload",
        help="Upload arena data to Huggingface Dataset repository"
    )
    hf_upload_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    hf_upload_parser.add_argument(
        "--repo_id",
        required=True,
        help="Huggingface repository ID (e.g., 'username/repo-name')"
    )
    hf_upload_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated list of subsets to upload (default: all)"
    )
    hf_upload_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to upload (default: all)"
    )
    hf_upload_parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated list of experiments (exp_name) to upload (default: all). "
             "In v2 layout, model outputs are uploaded as one ZIP per exp_name."
    )
    hf_upload_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the repository"
    )
    hf_upload_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    hf_upload_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per file on connection failure (default: 3)"
    )
    hf_upload_parser.set_defaults(func=cmd_hf_upload)

    # hf pull
    hf_pull_parser = hf_subparsers.add_parser(
        "pull",
        help="Pull arena data from Huggingface Dataset repository"
    )
    hf_pull_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path to save data to"
    )
    hf_pull_parser.add_argument(
        "--repo_id",
        required=True,
        help="Huggingface repository ID (e.g., 'username/repo-name')"
    )
    hf_pull_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated list of subsets to download (default: all)"
    )
    hf_pull_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of models to download (default: all)"
    )
    hf_pull_parser.add_argument(
        "--experiments",
        default=None,
        help="Comma-separated list of experiments (exp_name) to download (default: all). "
             "In v2 layout, model outputs are downloaded as one ZIP per exp_name."
    )
    hf_pull_parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision/branch to download from (default: main)"
    )
    hf_pull_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local files"
    )
    hf_pull_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    hf_pull_parser.set_defaults(func=cmd_hf_pull)

    # hf list
    hf_list_parser = hf_subparsers.add_parser(
        "list",
        help="List contents of a Huggingface Dataset repository"
    )
    hf_list_parser.add_argument(
        "--repo_id",
        required=True,
        help="Huggingface repository ID (e.g., 'username/repo-name')"
    )
    hf_list_parser.add_argument(
        "--revision",
        default="main",
        help="Repository revision/branch (default: main)"
    )
    hf_list_parser.set_defaults(func=cmd_hf_list)

    # === submit command ===
    submit_parser = subparsers.add_parser(
        "submit",
        help="Submit evaluation results to official leaderboard via GitHub PR"
    )

    submit_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    submit_parser.add_argument(
        "--subset",
        required=True,
        help="Subset name (e.g., 'basic')"
    )
    submit_parser.add_argument(
        "--exp_name",
        required=True,
        help="Experiment name (must end with _yyyymmdd)"
    )
    submit_parser.add_argument(
        "--hf_repo",
        required=True,
        help="Your HuggingFace Dataset repository ID (e.g., 'username/my-genarena-results')"
    )
    submit_parser.add_argument(
        "--official_repo",
        default=None,
        help="Official submissions repository (default: genarena/submissions)"
    )
    submit_parser.add_argument(
        "--title",
        default=None,
        help="PR title (default: auto-generated)"
    )
    submit_parser.add_argument(
        "--description",
        default=None,
        help="PR description"
    )
    submit_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    submit_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only, do not upload or create PR"
    )
    submit_parser.add_argument(
        "--skip-official-check",
        action="store_true",
        help="Skip checking against official models (for testing)"
    )
    submit_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    submit_parser.set_defaults(func=cmd_submit)

    # === export-models command ===
    export_models_parser = subparsers.add_parser(
        "export-models",
        help="Export official_models.json from arena state (for maintainers)"
    )

    export_models_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    export_models_parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output file path (default: print to stdout)"
    )

    export_models_parser.set_defaults(func=cmd_export_models)

    # === deploy command group ===
    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Deployment commands for HuggingFace Spaces"
    )
    deploy_subparsers = deploy_parser.add_subparsers(dest="deploy_command", help="Deploy subcommands")

    # deploy upload
    deploy_upload_parser = deploy_subparsers.add_parser(
        "upload",
        help="Upload all data for HuggingFace Spaces deployment (incremental)"
    )
    deploy_upload_parser.add_argument(
        "--arena_dir",
        required=True,
        help="Arena directory path"
    )
    deploy_upload_parser.add_argument(
        "--arena_repo",
        required=True,
        help="HuggingFace Dataset repo for arena data (e.g., 'your-org/leaderboard-data')"
    )
    deploy_upload_parser.add_argument(
        "--space_repo",
        required=True,
        help="HuggingFace Space repo for deployment (e.g., 'your-org/leaderboard')"
    )
    deploy_upload_parser.add_argument(
        "--subsets",
        default=None,
        help="Comma-separated list of subsets to upload (default: all)"
    )
    deploy_upload_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip existing for incremental upload)"
    )
    deploy_upload_parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts per file on connection failure (default: 3)"
    )
    deploy_upload_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    deploy_upload_parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of parallel workers for upload (default: 16)"
    )
    deploy_upload_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each worker (default: 300)"
    )
    deploy_upload_parser.set_defaults(func=cmd_deploy_upload)

    # deploy info
    deploy_info_parser = deploy_subparsers.add_parser(
        "info",
        help="Show deployment configuration and instructions"
    )
    deploy_info_parser.set_defaults(func=cmd_deploy_info)

    return parser


def main() -> int:
    """
    Main entry point.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Handle git subcommands
    if args.command == "git":
        if not hasattr(args, "git_command") or args.git_command is None:
            # Print git help
            parser.parse_args(["git", "--help"])
            return 0

    # Handle hf subcommands
    if args.command == "hf":
        if not hasattr(args, "hf_command") or args.hf_command is None:
            # Print hf help
            parser.parse_args(["hf", "--help"])
            return 0

    # Handle deploy subcommands
    if args.command == "deploy":
        if not hasattr(args, "deploy_command") or args.deploy_command is None:
            # Print deploy help
            parser.parse_args(["deploy", "--help"])
            return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
