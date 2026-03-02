#!/usr/bin/env python3
"""
Validate submission files for GenArena.

This script is run by GitHub Actions to validate submission PRs.
It downloads the submission data from HuggingFace, verifies integrity,
and checks that ELO calculations match.
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Validate GenArena submissions")
    parser.add_argument(
        "--files",
        required=True,
        help="Space-separated list of submission files to validate",
    )
    parser.add_argument(
        "--official-models",
        required=True,
        help="Path to official_models.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write validation result JSON",
    )
    args = parser.parse_args()

    # Get list of files
    files = args.files.split() if args.files else []
    
    if not files:
        result = {
            "status": "failed",
            "checks": [{"name": "Find submission file", "passed": False, "error": "No files provided"}],
            "errors": ["No submission files found in PR"],
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        return 1

    # Import validation module
    try:
        from genarena.validation import validate_submission_file
    except ImportError as e:
        result = {
            "status": "failed",
            "checks": [{"name": "Import genarena", "passed": False, "error": str(e)}],
            "errors": [f"Failed to import genarena: {e}"],
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        return 1

    # Validate the first file (typically one submission per PR)
    submission_file = files[0]
    
    if not os.path.isfile(submission_file):
        result = {
            "status": "failed",
            "checks": [{"name": "File exists", "passed": False, "error": f"File not found: {submission_file}"}],
            "errors": [f"Submission file not found: {submission_file}"],
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        return 1

    # Run validation
    print(f"Validating submission: {submission_file}")
    report = validate_submission_file(
        submission_path=submission_file,
        official_models_path=args.official_models,
        download_data=True,
    )

    # Write result
    result = report.to_dict()
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Validation complete. Status: {result['status']}")
    
    if result["status"] == "success":
        print("All checks passed!")
        return 0
    else:
        print("Validation failed:")
        for error in result.get("errors", []):
            print(f"  - {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
