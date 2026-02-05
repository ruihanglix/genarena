# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""
JSON Schema definition for GenArena submissions.

This schema defines the structure of submission metadata files
that are submitted via GitHub PR to the official leaderboard.
"""

from typing import Any

# JSON Schema for submission metadata
SUBMISSION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "GenArena Submission",
    "description": "Metadata for a GenArena evaluation submission",
    "type": "object",
    "required": [
        "schema_version",
        "submission_id",
        "created_at",
        "submitter",
        "experiment",
        "data_location",
        "elo_preview",
    ],
    "properties": {
        "schema_version": {
            "type": "string",
            "description": "Schema version (e.g., '1.0')",
            "pattern": "^\\d+\\.\\d+$",
        },
        "submission_id": {
            "type": "string",
            "description": "Unique submission identifier",
            "pattern": "^sub_\\d{8}T\\d{6}_[a-f0-9]{8}$",
        },
        "created_at": {
            "type": "string",
            "description": "ISO 8601 timestamp of submission creation",
            "format": "date-time",
        },
        "submitter": {
            "type": "object",
            "required": ["github_username"],
            "properties": {
                "github_username": {
                    "type": "string",
                    "description": "GitHub username of submitter",
                    "minLength": 1,
                },
                "contact": {
                    "type": "string",
                    "description": "Optional contact email",
                    "format": "email",
                },
            },
        },
        "experiment": {
            "type": "object",
            "required": [
                "exp_name",
                "subset",
                "models",
                "new_models",
                "total_battles",
            ],
            "properties": {
                "exp_name": {
                    "type": "string",
                    "description": "Experiment name (must end with _yyyymmdd)",
                    "pattern": "^.+_\\d{8}$",
                },
                "subset": {
                    "type": "string",
                    "description": "Subset name (e.g., 'basic')",
                    "minLength": 1,
                },
                "models": {
                    "type": "array",
                    "description": "List of all model names in the experiment",
                    "items": {"type": "string"},
                    "minItems": 2,
                },
                "new_models": {
                    "type": "array",
                    "description": "List of new model names (not in official leaderboard)",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "existing_models": {
                    "type": "array",
                    "description": "List of existing model names (already in official)",
                    "items": {"type": "string"},
                },
                "model_pairs": {
                    "type": "array",
                    "description": "List of model pairs evaluated",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "total_battles": {
                    "type": "integer",
                    "description": "Total number of battles",
                    "minimum": 1,
                },
                "battles_per_pair": {
                    "type": "object",
                    "description": "Battle count per model pair",
                    "additionalProperties": {"type": "integer"},
                },
            },
        },
        "data_location": {
            "type": "object",
            "required": ["hf_repo_id", "files"],
            "properties": {
                "hf_repo_id": {
                    "type": "string",
                    "description": "HuggingFace repository ID",
                    "pattern": "^[\\w.-]+/[\\w.-]+$",
                },
                "hf_revision": {
                    "type": "string",
                    "description": "HuggingFace revision/branch",
                    "default": "main",
                },
                "files": {
                    "type": "object",
                    "required": ["models_zip", "pk_logs_zip"],
                    "properties": {
                        "models_zip": {
                            "$ref": "#/$defs/file_info",
                        },
                        "pk_logs_zip": {
                            "$ref": "#/$defs/file_info",
                        },
                    },
                },
            },
        },
        "elo_preview": {
            "type": "object",
            "required": ["ratings"],
            "properties": {
                "ratings": {
                    "type": "object",
                    "description": "ELO ratings by model",
                    "additionalProperties": {"type": "number"},
                },
                "ci_95": {
                    "type": "object",
                    "description": "95% confidence intervals by model",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
            },
        },
        "evaluation_config": {
            "type": "object",
            "description": "Evaluation configuration used",
            "properties": {
                "judge_model": {
                    "type": "string",
                    "description": "VLM judge model name",
                },
                "prompt_module": {
                    "type": "string",
                    "description": "Prompt module name",
                },
                "temperature": {
                    "type": "number",
                    "description": "VLM temperature",
                    "minimum": 0,
                },
                "position_debiasing": {
                    "type": "boolean",
                    "description": "Whether position debiasing was used",
                },
            },
        },
        "title": {
            "type": "string",
            "description": "Submission title",
        },
        "description": {
            "type": "string",
            "description": "Submission description",
        },
        "verification": {
            "type": "object",
            "properties": {
                "local_validation_passed": {
                    "type": "boolean",
                    "description": "Whether local validation passed",
                },
                "genarena_version": {
                    "type": "string",
                    "description": "genarena version used for submission",
                },
            },
        },
    },
    "$defs": {
        "file_info": {
            "type": "object",
            "required": ["path", "sha256", "size_bytes"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path in HF repo",
                },
                "sha256": {
                    "type": "string",
                    "description": "SHA256 checksum",
                    "pattern": "^[a-f0-9]{64}$",
                },
                "size_bytes": {
                    "type": "integer",
                    "description": "File size in bytes",
                    "minimum": 1,
                },
            },
        },
    },
}


def validate_submission_schema(submission: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate submission against JSON schema.

    Args:
        submission: Submission metadata dictionary

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    try:
        import jsonschema
    except ImportError:
        # If jsonschema is not available, do basic validation
        return _basic_validation(submission)

    errors: list[str] = []

    try:
        jsonschema.validate(instance=submission, schema=SUBMISSION_SCHEMA)
        return True, []
    except jsonschema.ValidationError as e:
        errors.append(f"Schema validation error: {e.message}")
        if e.path:
            errors.append(f"  at path: {'.'.join(str(p) for p in e.path)}")
        return False, errors
    except jsonschema.SchemaError as e:
        errors.append(f"Schema error: {e.message}")
        return False, errors


def _basic_validation(submission: dict[str, Any]) -> tuple[bool, list[str]]:
    """Basic validation without jsonschema library."""
    errors: list[str] = []

    required_fields = [
        "schema_version",
        "submission_id",
        "created_at",
        "submitter",
        "experiment",
        "data_location",
        "elo_preview",
    ]

    for field in required_fields:
        if field not in submission:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # Check submitter
    if "github_username" not in submission.get("submitter", {}):
        errors.append("Missing submitter.github_username")

    # Check experiment
    exp = submission.get("experiment", {})
    exp_required = ["exp_name", "subset", "models", "new_models", "total_battles"]
    for field in exp_required:
        if field not in exp:
            errors.append(f"Missing experiment.{field}")

    # Check new_models is not empty
    if not exp.get("new_models"):
        errors.append("experiment.new_models must have at least one model")

    # Check data_location
    data_loc = submission.get("data_location", {})
    if "hf_repo_id" not in data_loc:
        errors.append("Missing data_location.hf_repo_id")
    if "files" not in data_loc:
        errors.append("Missing data_location.files")
    else:
        files = data_loc.get("files", {})
        for zip_type in ["models_zip", "pk_logs_zip"]:
            if zip_type not in files:
                errors.append(f"Missing data_location.files.{zip_type}")
            else:
                file_info = files[zip_type]
                for field in ["path", "sha256", "size_bytes"]:
                    if field not in file_info:
                        errors.append(f"Missing data_location.files.{zip_type}.{field}")

    # Check elo_preview
    if "ratings" not in submission.get("elo_preview", {}):
        errors.append("Missing elo_preview.ratings")

    return len(errors) == 0, errors
