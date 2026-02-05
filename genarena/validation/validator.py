# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
Validator for GenArena submissions.

This module provides functions to validate submission files,
including downloading and verifying data from HuggingFace.
Used by the GitHub Actions bot for automated validation.
"""

import hashlib
import json
import logging
import os
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import Any, Optional

from genarena.validation.schema import validate_submission_schema

logger = logging.getLogger(__name__)


@dataclass
class ValidationCheck:
    """Single validation check result."""

    name: str
    passed: bool
    error: Optional[str] = None


@dataclass
class ValidationReport:
    """Complete validation report for a submission."""

    status: str  # "success" or "failed"
    submission_id: str = ""
    exp_name: str = ""
    subset: str = ""
    models: list[str] = field(default_factory=list)
    new_models: list[str] = field(default_factory=list)
    total_battles: int = 0
    checks: list[ValidationCheck] = field(default_factory=list)
    elo_comparison: dict[str, dict[str, float]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def add_check(self, name: str, passed: bool, error: Optional[str] = None) -> None:
        """Add a validation check result."""
        self.checks.append(ValidationCheck(name=name, passed=passed, error=error))
        if not passed:
            self.status = "failed"
            if error:
                self.errors.append(f"{name}: {error}")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status,
            "submission_id": self.submission_id,
            "exp_name": self.exp_name,
            "subset": self.subset,
            "models": self.models,
            "new_models": self.new_models,
            "total_battles": self.total_battles,
            "checks": [
                {"name": c.name, "passed": c.passed, "error": c.error}
                for c in self.checks
            ],
            "elo_comparison": self.elo_comparison,
            "errors": self.errors,
        }


def validate_submission_file(
    submission_path: str,
    official_models_path: Optional[str] = None,
    download_data: bool = True,
) -> ValidationReport:
    """
    Validate a submission JSON file.

    This is the main entry point for validating submissions,
    used by the GitHub Actions bot.

    Args:
        submission_path: Path to submission JSON file
        official_models_path: Path to official_models.json (optional)
        download_data: Whether to download and verify data from HF

    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport(status="success")

    # 1. Load and parse JSON
    try:
        with open(submission_path, "r", encoding="utf-8") as f:
            submission = json.load(f)
        report.add_check("JSON parse", True)
    except json.JSONDecodeError as e:
        report.add_check("JSON parse", False, str(e))
        return report
    except IOError as e:
        report.add_check("File read", False, str(e))
        return report

    # 2. Schema validation
    is_valid, schema_errors = validate_submission_schema(submission)
    if is_valid:
        report.add_check("Schema validation", True)
    else:
        for err in schema_errors:
            report.add_check("Schema validation", False, err)
        return report

    # Extract basic info
    report.submission_id = submission.get("submission_id", "")
    exp = submission.get("experiment", {})
    report.exp_name = exp.get("exp_name", "")
    report.subset = exp.get("subset", "")
    report.models = exp.get("models", [])
    report.new_models = exp.get("new_models", [])
    report.total_battles = exp.get("total_battles", 0)

    # 3. Check new models against official list
    if official_models_path and os.path.isfile(official_models_path):
        try:
            with open(official_models_path, "r", encoding="utf-8") as f:
                official_data = json.load(f)
            official_models = set(
                official_data.get("subsets", {})
                .get(report.subset, {})
                .get("models", [])
            )

            # Verify new_models are actually new
            for model in report.new_models:
                if model in official_models:
                    report.add_check(
                        f"Model '{model}' is new",
                        False,
                        "Model already exists in official leaderboard",
                    )
                else:
                    report.add_check(f"Model '{model}' is new", True)

        except Exception as e:
            report.add_check(
                "Check official models", False, f"Failed to load official models: {e}"
            )
    else:
        report.add_check(
            "Check official models",
            True,
            "Skipped (no official_models.json provided)",
        )

    # 4. Download and verify data from HuggingFace
    if download_data:
        data_report = validate_submission_data(submission)
        for check in data_report.checks:
            report.checks.append(check)
            if not check.passed:
                report.status = "failed"
                if check.error:
                    report.errors.append(f"{check.name}: {check.error}")
        report.elo_comparison = data_report.elo_comparison
    else:
        report.add_check("Data verification", True, "Skipped (download_data=False)")

    return report


def validate_submission_data(submission: dict[str, Any]) -> ValidationReport:
    """
    Download and validate submission data from HuggingFace.

    Downloads the pk_logs ZIP, verifies checksum, extracts battles,
    and recalculates ELO for comparison.

    Args:
        submission: Submission metadata dictionary

    Returns:
        ValidationReport with data validation results
    """
    report = ValidationReport(status="success")

    data_loc = submission.get("data_location", {})
    hf_repo = data_loc.get("hf_repo_id", "")
    hf_revision = data_loc.get("hf_revision", "main")
    files = data_loc.get("files", {})
    pk_logs_info = files.get("pk_logs_zip", {})

    if not hf_repo or not pk_logs_info:
        report.add_check("Data location", False, "Missing HF repo or file info")
        return report

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        report.add_check(
            "HuggingFace Hub",
            False,
            "huggingface_hub not installed",
        )
        return report

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download pk_logs ZIP
        try:
            pk_logs_path = hf_hub_download(
                repo_id=hf_repo,
                filename=pk_logs_info["path"],
                repo_type="dataset",
                revision=hf_revision,
                local_dir=tmpdir,
            )
            report.add_check("Download pk_logs", True)
        except Exception as e:
            report.add_check("Download pk_logs", False, str(e))
            return report

        # Verify SHA256
        expected_sha = pk_logs_info.get("sha256", "")
        try:
            with open(pk_logs_path, "rb") as f:
                actual_sha = hashlib.sha256(f.read()).hexdigest()

            if actual_sha == expected_sha:
                report.add_check("SHA256 checksum", True)
            else:
                report.add_check(
                    "SHA256 checksum",
                    False,
                    f"Expected {expected_sha[:16]}..., got {actual_sha[:16]}...",
                )
                return report
        except Exception as e:
            report.add_check("SHA256 checksum", False, str(e))
            return report

        # Extract ZIP
        extract_dir = os.path.join(tmpdir, "extracted")
        try:
            with zipfile.ZipFile(pk_logs_path, "r") as zf:
                zf.extractall(extract_dir)
            report.add_check("Extract ZIP", True)
        except Exception as e:
            report.add_check("Extract ZIP", False, str(e))
            return report

        # Find battle log files
        # The ZIP structure is: <exp_name>/*.jsonl
        battle_records = []
        try:
            for root, dirs, filenames in os.walk(extract_dir):
                for filename in filenames:
                    if filename.endswith(".jsonl") and "raw_outputs" not in root:
                        filepath = os.path.join(root, filename)
                        with open(filepath, "r", encoding="utf-8") as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        record = json.loads(line)
                                        battle_records.append(record)
                                    except json.JSONDecodeError:
                                        continue
            report.add_check("Parse battle logs", True)
        except Exception as e:
            report.add_check("Parse battle logs", False, str(e))
            return report

        # Verify battle count
        expected_battles = submission.get("experiment", {}).get("total_battles", 0)
        if len(battle_records) == expected_battles:
            report.add_check("Battle count", True)
        else:
            report.add_check(
                "Battle count",
                False,
                f"Expected {expected_battles}, got {len(battle_records)}",
            )

        # Recalculate ELO
        try:
            from genarena.bt_elo import compute_bt_elo_ratings

            battles = [
                (r["model_a"], r["model_b"], r["final_winner"])
                for r in battle_records
                if r.get("model_a") and r.get("model_b") and r.get("final_winner")
            ]

            if battles:
                recalc_elo = compute_bt_elo_ratings(battles)
                submitted_elo = submission.get("elo_preview", {}).get("ratings", {})

                all_match = True
                for model, submitted_rating in submitted_elo.items():
                    recalc_rating = recalc_elo.get(model, 0)
                    report.elo_comparison[model] = {
                        "submitted": submitted_rating,
                        "recalculated": recalc_rating,
                    }

                    # Allow small floating point differences (±1.0)
                    diff = abs(submitted_rating - recalc_rating)
                    if diff > 1.0:
                        report.add_check(
                            f"ELO '{model}'",
                            False,
                            f"Diff: {diff:.1f} (submitted: {submitted_rating:.1f}, "
                            f"recalc: {recalc_rating:.1f})",
                        )
                        all_match = False

                if all_match:
                    report.add_check("ELO verification", True)

        except Exception as e:
            report.add_check("ELO verification", False, str(e))

    return report
