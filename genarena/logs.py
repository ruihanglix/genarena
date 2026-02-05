# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Logging module for battle results and audit trails."""

import fcntl
import json
import os
from typing import Any, Optional

from genarena.battle import BattleResult
from genarena.utils import ensure_dir, get_sorted_model_pair, iso_timestamp, sanitize_name


class BattleLogger:
    """
    Logger for battle results (slim format for ELO calculation).

    Stores minimal information needed for ELO scoring:
    - model_a, model_b
    - sample_index
    - final_winner
    - is_consistent
    - timestamp

    Thread-safe file writes using file locking.
    """

    def __init__(self, exp_dir: str):
        """
        Initialize the battle logger.

        Args:
            exp_dir: Experiment directory path (e.g., pk_logs/<exp_name>/)
        """
        self.exp_dir = exp_dir
        ensure_dir(exp_dir)

    def _get_log_path(self, model_a: str, model_b: str) -> str:
        """
        Get the log file path for a model pair.

        Models are sorted alphabetically to ensure consistent file naming.

        Args:
            model_a: First model name
            model_b: Second model name

        Returns:
            Path to the jsonl log file
        """
        first, second, _ = get_sorted_model_pair(model_a, model_b)
        filename = f"{sanitize_name(first)}_vs_{sanitize_name(second)}.jsonl"
        return os.path.join(self.exp_dir, filename)

    def log(
        self,
        model_a: str,
        model_b: str,
        sample_index: int,
        final_winner: str,
        is_consistent: bool
    ) -> None:
        """
        Log a battle result.

        Thread-safe append to the model pair's log file.

        Args:
            model_a: First model name
            model_b: Second model name
            sample_index: Data sample index
            final_winner: Winner ("model_a", "model_b", or "tie")
            is_consistent: Whether both VLM calls agreed
        """
        log_path = self._get_log_path(model_a, model_b)

        # Ensure winner uses consistent naming based on sorted order
        first, second, swapped = get_sorted_model_pair(model_a, model_b)

        # Convert winner to sorted model names
        if final_winner == "model_a":
            sorted_winner = second if swapped else first
        elif final_winner == "model_b":
            sorted_winner = first if swapped else second
        else:
            sorted_winner = "tie"

        record = {
            "model_a": first,
            "model_b": second,
            "sample_index": sample_index,
            "final_winner": sorted_winner,
            "is_consistent": is_consistent,
            "timestamp": iso_timestamp()
        }

        # Thread-safe write with file locking
        with open(log_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log_battle_result(self, result: BattleResult) -> None:
        """
        Log a BattleResult object.

        Args:
            result: BattleResult from battle execution
        """
        self.log(
            model_a=result.model_a,
            model_b=result.model_b,
            sample_index=result.sample_index,
            final_winner=result.final_winner,
            is_consistent=result.is_consistent
        )


class AuditLogger:
    """
    Audit logger for detailed VLM outputs.

    Stores complete information for debugging and verification:
    - Raw VLM responses
    - Parsed results
    - Parse success/error status
    - Final outcome

    Thread-safe file writes using file locking.
    """

    def __init__(self, exp_dir: str):
        """
        Initialize the audit logger.

        Args:
            exp_dir: Experiment directory path (e.g., pk_logs/<exp_name>/)
        """
        self.raw_outputs_dir = os.path.join(exp_dir, "raw_outputs")
        ensure_dir(self.raw_outputs_dir)

    def _get_log_path(self, model_a: str, model_b: str) -> str:
        """
        Get the audit log file path for a model pair.

        Args:
            model_a: First model name
            model_b: Second model name

        Returns:
            Path to the jsonl audit log file
        """
        first, second, _ = get_sorted_model_pair(model_a, model_b)
        filename = f"{sanitize_name(first)}_vs_{sanitize_name(second)}.jsonl"
        return os.path.join(self.raw_outputs_dir, filename)

    def log(
        self,
        model_a: str,
        model_b: str,
        sample_index: int,
        original_call: dict[str, Any],
        swapped_call: dict[str, Any],
        final_winner: str,
        is_consistent: bool
    ) -> None:
        """
        Log detailed audit information.

        Args:
            model_a: First model name
            model_b: Second model name
            sample_index: Data sample index
            original_call: Dict with raw_response, parsed_result, parse_success, parse_error
            swapped_call: Dict with raw_response, parsed_result, parse_success, parse_error
            final_winner: Winner ("model_a", "model_b", or "tie")
            is_consistent: Whether both VLM calls agreed
        """
        log_path = self._get_log_path(model_a, model_b)

        # Use sorted model names for consistency
        first, second, swapped = get_sorted_model_pair(model_a, model_b)

        # Convert winner to sorted model names
        if final_winner == "model_a":
            sorted_winner = second if swapped else first
        elif final_winner == "model_b":
            sorted_winner = first if swapped else second
        else:
            sorted_winner = "tie"

        record = {
            "model_a": first,
            "model_b": second,
            "sample_index": sample_index,
            "timestamp": iso_timestamp(),
            "original_call": original_call,
            "swapped_call": swapped_call,
            "final_winner": sorted_winner,
            "is_consistent": is_consistent
        }

        # Thread-safe write with file locking
        with open(log_path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def log_battle_result(self, result: BattleResult) -> None:
        """
        Log a BattleResult object with full audit details.

        Args:
            result: BattleResult from battle execution
        """
        # Convert CallResult to dict
        original_call = {
            "raw_response": result.original_call.raw_response,
            "parsed_result": result.original_call.parsed_result,
            "parse_success": result.original_call.parse_success,
            "parse_error": result.original_call.parse_error
        }

        swapped_call = {
            "raw_response": result.swapped_call.raw_response,
            "parsed_result": result.swapped_call.parsed_result,
            "parse_success": result.swapped_call.parse_success,
            "parse_error": result.swapped_call.parse_error
        }

        self.log(
            model_a=result.model_a,
            model_b=result.model_b,
            sample_index=result.sample_index,
            original_call=original_call,
            swapped_call=swapped_call,
            final_winner=result.final_winner,
            is_consistent=result.is_consistent
        )


def load_battle_history(pk_logs_dir: str) -> set[tuple[str, str, int]]:
    """
    Load completed battle records from all experiment directories.

    Used for checkpoint/resume functionality to skip already-completed battles.

    Args:
        pk_logs_dir: Path to pk_logs directory

    Returns:
        Set of (model_a, model_b, sample_index) tuples representing completed battles
        (model names are in sorted order)
    """
    completed: set[tuple[str, str, int]] = set()

    if not os.path.isdir(pk_logs_dir):
        return completed

    # Iterate over all experiment directories
    for exp_name in os.listdir(pk_logs_dir):
        exp_dir = os.path.join(pk_logs_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue

        # Iterate over all jsonl files (excluding raw_outputs subdirectory)
        for filename in os.listdir(exp_dir):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(exp_dir, filename)
            if not os.path.isfile(filepath):
                continue

            # Read and parse each line
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            model_a = record.get("model_a", "")
                            model_b = record.get("model_b", "")
                            sample_index = record.get("sample_index", -1)

                            if model_a and model_b and sample_index >= 0:
                                # Models should already be sorted in the log
                                completed.add((model_a, model_b, sample_index))
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

    return completed


def load_battle_records(pk_logs_dir: str, exp_name: Optional[str] = None):
    """
    Load battle records from log files.

    Args:
        pk_logs_dir: Path to pk_logs directory
        exp_name: Specific experiment name (optional, loads all if None)

    Returns:
        List of battle record dicts
    """
    records = []

    if not os.path.isdir(pk_logs_dir):
        return records

    # Determine which directories to scan
    if exp_name:
        exp_dirs = [os.path.join(pk_logs_dir, exp_name)]
    else:
        exp_dirs = [
            os.path.join(pk_logs_dir, name)
            for name in os.listdir(pk_logs_dir)
            if os.path.isdir(os.path.join(pk_logs_dir, name))
        ]

    for exp_dir in exp_dirs:
        if not os.path.isdir(exp_dir):
            continue

        for filename in os.listdir(exp_dir):
            if not filename.endswith(".jsonl"):
                continue

            filepath = os.path.join(exp_dir, filename)
            if not os.path.isfile(filepath):
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            records.append(record)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

    return records


def count_battles_per_pair(pk_logs_dir: str) -> dict[tuple[str, str], int]:
    """
    Count the number of completed battles for each model pair.
    
    Args:
        pk_logs_dir: Path to pk_logs directory
    
    Returns:
        Dict mapping (model_a, model_b) tuples (sorted) to battle count
    """
    counts: dict[tuple[str, str], int] = {}
    
    if not os.path.isdir(pk_logs_dir):
        return counts
    
    # Iterate over all experiment directories
    for exp_name in os.listdir(pk_logs_dir):
        exp_dir = os.path.join(pk_logs_dir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        
        for filename in os.listdir(exp_dir):
            if not filename.endswith(".jsonl"):
                continue
            
            filepath = os.path.join(exp_dir, filename)
            if not os.path.isfile(filepath):
                continue
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            model_a = record.get("model_a", "")
                            model_b = record.get("model_b", "")
                            
                            if model_a and model_b:
                                # Ensure sorted order
                                key = (min(model_a, model_b), max(model_a, model_b))
                                counts[key] = counts.get(key, 0) + 1
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue
    
    return counts
