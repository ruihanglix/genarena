# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Experiment name utilities for GenArena.

This module centralizes experiment naming rules and milestone detection.

Key rules:
- exp_name must end with a date suffix: `_yyyymmdd` (8 digits)
- milestone experiments are identified by a fixed prefix constant
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Optional, Tuple

# Fixed milestone prefix (must be constant across the codebase)
MILESTONE_PREFIX: str = "GenArena_"

# exp_name must end with "_yyyymmdd"
_EXP_DATE_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_(?P<date>\d{8})$")


def parse_exp_date_suffix(exp_name: str) -> Optional[date]:
    """Parse the `_yyyymmdd` suffix from exp_name.

    Returns None if exp_name does not match the required format or if the date is invalid.
    """
    m = _EXP_DATE_SUFFIX_RE.match(exp_name or "")
    if not m:
        return None
    datestr = m.group("date")
    try:
        return datetime.strptime(datestr, "%Y%m%d").date()
    except ValueError:
        return None


def is_valid_exp_name(exp_name: str) -> bool:
    """Return True iff exp_name ends with a valid `_yyyymmdd` date suffix."""
    return parse_exp_date_suffix(exp_name) is not None


def validate_exp_name(exp_name: str) -> Tuple[bool, str]:
    """Validate exp_name. Returns (ok, error_message)."""
    if not exp_name:
        return False, "exp_name is empty"
    if not _EXP_DATE_SUFFIX_RE.match(exp_name):
        return False, "exp_name must end with `_yyyymmdd` (e.g., `MyExp_20260128`)"
    if parse_exp_date_suffix(exp_name) is None:
        return False, "exp_name has an invalid date suffix; expected a real calendar date in `_yyyymmdd`"
    return True, ""


def require_valid_exp_name(exp_name: str) -> None:
    """Raise ValueError if exp_name is invalid."""
    ok, msg = validate_exp_name(exp_name)
    if not ok:
        raise ValueError(msg)


def is_milestone_exp(exp_name: str) -> bool:
    """Return True if exp_name is a milestone experiment (fixed prefix)."""
    return (exp_name or "").startswith(MILESTONE_PREFIX)


@dataclass(frozen=True)
class ExperimentInfo:
    """Parsed experiment directory info."""

    name: str
    date: date


def discover_experiments(models_root: str) -> List[ExperimentInfo]:
    """Discover valid experiment directories under a models root directory.

    Args:
        models_root: `arena_dir/<subset>/models`

    Returns:
        List of ExperimentInfo for directories whose name matches `_yyyymmdd`.
    """
    infos: List[ExperimentInfo] = []
    if not os.path.isdir(models_root):
        return infos

    for name in os.listdir(models_root):
        if name.startswith("."):
            continue
        path = os.path.join(models_root, name)
        if not os.path.isdir(path):
            continue
        d = parse_exp_date_suffix(name)
        if d is None:
            continue
        infos.append(ExperimentInfo(name=name, date=d))

    # Sort by (date, name) for deterministic ordering
    infos.sort(key=lambda x: (x.date, x.name))
    return infos


def pick_latest_experiment_name(models_root: str) -> str:
    """Pick the latest experiment name under models_root by `_yyyymmdd` suffix.

    Raises ValueError if no valid experiment directories exist.
    """
    infos = discover_experiments(models_root)
    if not infos:
        raise ValueError(
            f"No valid experiments found under models dir: {models_root}. "
            f"Expected subdirectories named like `<something>_yyyymmdd`."
        )
    return infos[-1].name

