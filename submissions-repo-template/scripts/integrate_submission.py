#!/usr/bin/env python3
"""
Integrate approved submissions into official GenArena repositories.

This script is run by GitHub Actions after a submission PR is merged.
It downloads data from the submitter's HuggingFace repo, uploads to
official repos, rebuilds arena state, and updates official_models.json.
"""

import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Integrate GenArena submissions")
    parser.add_argument(
        "--files",
        required=True,
        help="Space-separated list of submission files to integrate",
    )
    parser.add_argument(
        "--official-models",
        required=True,
        help="Path to official_models.json",
    )
    parser.add_argument(
        "--battlefield-repo",
        default="rhli/genarena-battlefield",
        help="Official battlefield HF repo",
    )
    parser.add_argument(
        "--leaderboard-repo",
        default="genarena/leaderboard-data",
        help="Official leaderboard data HF repo",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write integration result JSON",
    )
    args = parser.parse_args()

    files = args.files.split() if args.files else []

    if not files:
        result = {"status": "skipped", "message": "No files to integrate"}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print("No submission files to integrate.")
        return 0

    try:
        from genarena.sync.integrator import integrate_submission
    except ImportError as e:
        result = {"status": "failed", "errors": [f"Failed to import genarena: {e}"]}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        return 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        result = {"status": "failed", "errors": ["HF_TOKEN environment variable not set"]}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        return 1

    all_results = []
    overall_success = True

    for submission_file in files:
        if not os.path.isfile(submission_file):
            print(f"Skipping missing file: {submission_file}")
            continue

        print(f"\n{'=' * 60}")
        print(f"Integrating: {submission_file}")
        print(f"{'=' * 60}")

        integration = integrate_submission(
            submission_path=submission_file,
            battlefield_repo=args.battlefield_repo,
            leaderboard_repo=args.leaderboard_repo,
            official_models_path=args.official_models,
            token=token,
        )

        file_result = integration.to_dict()
        file_result["file"] = submission_file
        all_results.append(file_result)

        if integration.success:
            # Move from pending/ to approved/
            approved_path = submission_file.replace(
                "submissions/pending/", "submissions/approved/"
            )
            os.makedirs(os.path.dirname(approved_path), exist_ok=True)
            shutil.move(submission_file, approved_path)
            print(f"  Moved to {approved_path}")
        else:
            overall_success = False
            print("  Integration FAILED:")
            for err in integration.errors:
                print(f"    - {err}")

    result = {
        "status": "success" if overall_success else "failed",
        "integrations": all_results,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
