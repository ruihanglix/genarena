# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Arena core coordinator module."""

import itertools
import json
import logging
import os
import random
import threading
import queue as thread_queue
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from genarena.battle import BattleResult, execute_battle
from genarena.bt_elo import compute_bootstrap_bt_elo, BattleTuple
from genarena.data import ParquetDataset, discover_subsets
from genarena.experiments import pick_latest_experiment_name, require_valid_exp_name, is_milestone_exp, parse_exp_date_suffix
from genarena.leaderboard import save_leaderboard
from genarena.logs import AuditLogger, BattleLogger, load_battle_history, count_battles_per_pair, load_battle_records
from genarena.models import GlobalModelOutputManager, ModelOutputManager
from genarena.prompts import load_prompt
from genarena.sampling import SamplingConfig, AdaptiveSamplingScheduler
from genarena.state import ArenaState, load_state, rebuild_state_from_logs, save_state, update_stats
from genarena.utils import ensure_dir, get_sorted_model_pair, iso_timestamp
from genarena.vlm import VLMJudge


logger = logging.getLogger(__name__)


@dataclass
class BattlePair:
    """A pair of models and sample for a battle."""

    model_a: str
    model_b: str
    sample_index: int


@dataclass
class ArenaConfig:
    """Configuration for an arena run."""

    # Required paths
    arena_dir: str
    data_dir: str
    subset: str

    # Model configuration
    models: Optional[list[str]] = None  # None = all models

    # Experiment configuration
    exp_name: Optional[str] = None  # None = timestamp
    sample_size: Optional[int] = None  # None = all samples (used in full mode)
    num_threads: int = 8
    num_processes: int = 1
    parallel_swap_calls: bool = False
    enable_progress_bar: bool = False
    
    # Sampling configuration
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    # VLM configuration
    judge_model: str = "Qwen/Qwen3-VL-32B-Instruct-FP8"
    temperature: float = 0.0
    prompt: str = "mmrb2"
    timeout: int = 120
    max_retries: int = 3

    # Multi-endpoint configuration
    base_urls: Optional[Union[str, list[str]]] = None  # Comma-separated or list
    api_keys: Optional[Union[str, list[str]]] = None   # Comma-separated or list

    # Logging configuration
    enable_audit_log: bool = True
    verbose: bool = False

    # Model removal behavior
    clean_orphaned_logs: bool = True  # Delete battle logs involving removed models

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        # Parse base_urls for logging
        base_urls_list = []
        if self.base_urls:
            if isinstance(self.base_urls, str):
                base_urls_list = [u.strip() for u in self.base_urls.split(",") if u.strip()]
            else:
                base_urls_list = list(self.base_urls)

        # Count api_keys for logging (don't expose actual keys)
        num_api_keys = 0
        if self.api_keys:
            if isinstance(self.api_keys, str):
                num_api_keys = len([k for k in self.api_keys.split(",") if k.strip()])
            else:
                num_api_keys = len(self.api_keys)

        return {
            "arena_dir": self.arena_dir,
            "data_dir": self.data_dir,
            "subset": self.subset,
            "models": self.models,
            "exp_name": self.exp_name,
            "sample_size": self.sample_size,
            "num_threads": self.num_threads,
            "num_processes": self.num_processes,
            "parallel_swap_calls": self.parallel_swap_calls,
            "enable_progress_bar": self.enable_progress_bar,
            "sampling": self.sampling.to_dict(),
            "judge_model": self.judge_model,
            "temperature": self.temperature,
            "prompt": self.prompt,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "base_urls": base_urls_list,
            "num_api_keys": num_api_keys,
            "enable_audit_log": self.enable_audit_log,
            "clean_orphaned_logs": self.clean_orphaned_logs,
            "timestamp": iso_timestamp()
        }


def _run_parquet_bucket_worker(
    *,
    arena_dir: str,
    data_dir: str,
    subset: str,
    exp_name: str,
    parquet_work: list[tuple[str, list[int]]],
    models: list[str],
    new_models: list[str],
    num_threads: int,
    judge_model: str,
    temperature: float,
    prompt: str,
    timeout: int,
    max_retries: int,
    base_urls: Optional[Union[str, list[str]]],
    api_keys: Optional[Union[str, list[str]]],
    enable_audit_log: bool,
    parallel_swap_calls: bool,
    progress_queue: Any = None,
) -> dict[str, int]:
    """
    Worker entry point for multiprocessing: execute battles for a bucket of parquet files.

    Notes:
    - Each process initializes its own VLM client/endpoint manager.
    - Results are persisted via jsonl logs (with fcntl locks), so the parent process
      only needs counts for progress reporting.
    """
    # Local imports are avoided here because the module is already imported in workers,
    # but keep this function at module-level so it's picklable by ProcessPoolExecutor.
    subset_dir = os.path.join(arena_dir, subset)
    models_dir = os.path.join(subset_dir, "models")
    pk_logs_dir = os.path.join(subset_dir, "pk_logs")
    exp_dir = os.path.join(pk_logs_dir, exp_name)

    ensure_dir(exp_dir)

    prompt_module = load_prompt(prompt)
    # In v2 layout, models are stored under models/<exp_name>/<model>/...
    # and model names are globally unique across experiments.
    model_manager = GlobalModelOutputManager(models_dir)

    vlm = VLMJudge(
        model=judge_model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        base_urls=base_urls,
        api_keys=api_keys,
        progress=progress_queue,
    )

    battle_logger = BattleLogger(exp_dir)
    audit_logger = AuditLogger(exp_dir) if enable_audit_log else None

    completed_set = load_battle_history(pk_logs_dir)

    class _ProgressBuffer:
        """Batch progress updates to reduce cross-process queue overhead."""

        def __init__(self, q: Any, flush_every: int = 20):
            self._q = q
            self._flush_every = flush_every
            self._buf = 0

        def put(self, n: int) -> None:
            if self._q is None:
                return
            self._buf += int(n)
            if self._buf >= self._flush_every:
                try:
                    self._q.put(self._buf)
                finally:
                    self._buf = 0

        def total(self, n: int) -> None:
            """Increase progress bar total by n (best-effort)."""
            if self._q is None:
                return
            try:
                n_int = int(n)
            except Exception:
                return
            if n_int <= 0:
                return
            try:
                self._q.put(("total", n_int))
            except Exception:
                pass

        def flush(self) -> None:
            if self._q is None:
                return
            if self._buf > 0:
                try:
                    self._q.put(self._buf)
                finally:
                    self._buf = 0

    progress = _ProgressBuffer(progress_queue) if progress_queue is not None else None

    def _execute_one(dataset: ParquetDataset, model_a: str, model_b: str, sample_index: int) -> bool:
        # Skip if already completed (sorted key)
        first, second, _ = get_sorted_model_pair(model_a, model_b)
        if (first, second, sample_index) in completed_set:
            return False

        sample = dataset.get_by_index(sample_index)
        if sample is None:
            return False

        output_a = model_manager.get_output_path(model_a, sample_index)
        output_b = model_manager.get_output_path(model_b, sample_index)
        if output_a is None or output_b is None:
            return False

        result = execute_battle(
            vlm=vlm,
            prompt_module=prompt_module,
            sample=sample,
            model_a_output=output_a,
            model_b_output=output_b,
            model_a=model_a,
            model_b=model_b,
            parallel_swap_calls=parallel_swap_calls,
            progress=progress,
        )

        battle_logger.log_battle_result(result)
        if audit_logger:
            audit_logger.log_battle_result(result)

        return True

    # Build tasks lazily and keep inflight bounded to reduce overhead for large runs.
    completed = 0
    total_attempted = 0
    total_indices = 0

    selected_models = set(models)
    new_models_filtered = [m for m in new_models if m in selected_models]
    if not new_models_filtered:
        return {"completed": 0, "attempted": 0, "indices": 0}

    pair_set: set[tuple[str, str]] = set()
    for m in new_models_filtered:
        for other in selected_models:
            if other == m:
                continue
            a, b, _ = get_sorted_model_pair(m, other)
            pair_set.add((a, b))

    model_pairs = sorted(pair_set)

    if num_threads <= 1:
        for pf, indices in parquet_work:
            if not indices:
                continue
            total_indices += len(indices)
            dataset = ParquetDataset(data_dir, subset, parquet_files=[pf])
            for model_a, model_b in model_pairs:
                valid_indices = model_manager.validate_coverage(model_a, model_b, indices)
                first, second, _ = get_sorted_model_pair(model_a, model_b)
                pending_indices = [idx for idx in valid_indices if (first, second, idx) not in completed_set]
                if progress is not None:
                    # Each battle always makes 2 API calls (original + swapped).
                    progress.total(2 * len(pending_indices))
                for idx in pending_indices:
                    total_attempted += 1
                    if _execute_one(dataset, model_a, model_b, idx):
                        completed += 1
    else:
        max_inflight = max(1, num_threads * 4)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            inflight = set()

            def _drain_one() -> None:
                nonlocal completed
                done_future = next(as_completed(inflight))
                inflight.remove(done_future)
                try:
                    ok = done_future.result()
                    if ok:
                        completed += 1
                except Exception:
                    # Worker-level robustness: ignore individual battle failures.
                    pass

            for pf, indices in parquet_work:
                if not indices:
                    continue
                total_indices += len(indices)
                dataset = ParquetDataset(data_dir, subset, parquet_files=[pf])
                for model_a, model_b in model_pairs:
                    valid_indices = model_manager.validate_coverage(model_a, model_b, indices)
                    first, second, _ = get_sorted_model_pair(model_a, model_b)
                    pending_indices = [idx for idx in valid_indices if (first, second, idx) not in completed_set]
                    if progress is not None:
                        progress.total(2 * len(pending_indices))
                    for idx in pending_indices:
                        total_attempted += 1
                        inflight.add(executor.submit(_execute_one, dataset, model_a, model_b, idx))
                        if len(inflight) >= max_inflight:
                            _drain_one()

            while inflight:
                _drain_one()

    if progress is not None:
        progress.flush()

    return {
        "completed": completed,
        "attempted": total_attempted,
        "indices": total_indices,
    }


def _start_calls_progress_consumer(
    *,
    enabled: bool,
    total: Optional[int] = None,
) -> tuple[Any, Optional[threading.Thread], Any]:
    """
    Start a progress consumer thread that reads integer increments from a queue.

    Returns:
        (progress_queue, thread, close_fn)
    """
    if not enabled:
        return None, None, lambda: None

    try:
        from tqdm import tqdm  # type: ignore
    except Exception:
        logger.warning("tqdm is not available; progress bar disabled")
        return None, None, lambda: None

    q: thread_queue.Queue[Any] = thread_queue.Queue()
    stop_sentinel = object()
    bar = tqdm(total=total, unit="call", desc="API Calls", dynamic_ncols=True)
    recent: deque[str] = deque(maxlen=10)

    def _run() -> None:
        while True:
            item = q.get()
            if item is stop_sentinel:
                break
            if isinstance(item, (int, float)):
                try:
                    bar.update(int(item))
                except Exception:
                    pass
            elif isinstance(item, tuple) and len(item) == 2 and item[0] == "log":
                try:
                    recent.append(str(item[1]))
                    bar.set_postfix_str(" | ".join(recent))
                except Exception:
                    pass
            elif isinstance(item, tuple) and len(item) == 2 and item[0] == "total":
                try:
                    delta = int(item[1])
                    if delta > 0:
                        bar.total = (bar.total or 0) + delta
                        bar.refresh()
                except Exception:
                    pass
            else:
                # Unknown item type, ignore.
                pass

    t = threading.Thread(target=_run, name="calls-progress-consumer", daemon=True)
    t.start()

    def _close() -> None:
        try:
            q.put(stop_sentinel)
        except Exception:
            pass
        if t is not None:
            t.join(timeout=5)
        try:
            bar.close()
        except Exception:
            pass

    return q, t, _close


class Arena:
    """
    Arena coordinator for running pairwise model evaluations.

    Manages:
    - Subset directory structure
    - Model discovery and output management
    - Battle pair generation
    - Checkpoint/resume functionality
    - Parallel battle execution
    - ELO state management
    - Leaderboard generation
    """

    def __init__(self, config: ArenaConfig):
        """
        Initialize the arena.

        Args:
            config: ArenaConfig with all settings
        """
        self.config = config

        # Set up paths
        self.subset_dir = os.path.join(config.arena_dir, config.subset)
        self.models_root_dir = os.path.join(self.subset_dir, "models")
        self.pk_logs_dir = os.path.join(self.subset_dir, "pk_logs")
        # Resolve experiment name (infer from models/ if not provided)
        if config.exp_name is not None:
            require_valid_exp_name(config.exp_name)
        else:
            config.exp_name = pick_latest_experiment_name(self.models_root_dir)

        # In v2 layout, per-experiment model outputs live under: models/<exp_name>/<model>/...
        self.models_dir = os.path.join(self.models_root_dir, config.exp_name)
        if not os.path.isdir(self.models_dir):
            raise ValueError(
                f"Experiment models directory does not exist: {self.models_dir}. "
                f"Expected `models/{config.exp_name}/<model_name>/...`."
            )
        self.exp_dir = os.path.join(self.pk_logs_dir, config.exp_name)
        self.arena_state_dir = os.path.join(self.subset_dir, "arena")
        self.state_path = os.path.join(self.arena_state_dir, "state.json")
        self.leaderboard_path = os.path.join(self.subset_dir, "README.md")

        # Initialize directories
        self._init_directories()

        # Load components
        self.prompt_module = load_prompt(config.prompt)
        # In multiprocessing mode, we only need fast index scanning in the parent
        # process (full data is loaded per-parquet inside workers).
        load_mode = "index_only" if config.num_processes > 1 else "full"
        self.dataset = ParquetDataset(config.data_dir, config.subset, load_mode=load_mode)
        # Global model registry (v2 layout): models/<exp_name>/<model>/...
        self.model_manager = GlobalModelOutputManager(self.models_root_dir)

        # Models that are newly introduced in this experiment (directory listing)
        self.new_models = self.model_manager.get_experiment_models(config.exp_name)

        # Parse experiment date for filtering eligible opponents
        self.exp_date = parse_exp_date_suffix(config.exp_name)

        # Resolve selected model universe for this run
        # When running an old experiment, only consider models from experiments
        # with date <= this experiment's date (to avoid battling "future" models).
        if config.models:
            self.models = [m for m in config.models if self.model_manager.has_model(m)]
        elif self.exp_date is not None:
            # Filter to models from experiments up to this experiment's date
            self.models = self.model_manager.get_models_up_to_date(self.exp_date)
        else:
            self.models = self.model_manager.models

        # Canonical "current models on disk" (used for state/log cleanup even when --models is used)
        self.all_models = self.model_manager.models

        # Initialize loggers
        self.battle_logger = BattleLogger(self.exp_dir)
        self.audit_logger = AuditLogger(self.exp_dir) if config.enable_audit_log else None

        # Initialize VLM judge with multi-endpoint support
        self.vlm = VLMJudge(
            model=config.judge_model,
            temperature=config.temperature,
            timeout=config.timeout,
            max_retries=config.max_retries,
            base_urls=config.base_urls,
            api_keys=config.api_keys,
        )

        # Save experiment config
        self._save_config()
        self._progress_queue = None

    def _init_directories(self) -> None:
        """Create necessary directory structure."""
        ensure_dir(self.subset_dir)
        ensure_dir(self.models_root_dir)
        ensure_dir(self.pk_logs_dir)
        ensure_dir(self.exp_dir)
        ensure_dir(self.arena_state_dir)

        if self.config.enable_audit_log:
            ensure_dir(os.path.join(self.exp_dir, "raw_outputs"))

    def _save_config(self) -> None:
        """Save experiment configuration."""
        config_path = os.path.join(self.exp_dir, "config.json")
        history_path = os.path.join(self.exp_dir, "config_history.json")

        config_dict = self.config.to_dict()
        config_dict["models_actual"] = self.models

        # If config exists, append to history
        if os.path.isfile(config_path):
            # Read existing config and append to history
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)

                # Load or create history
                history = []
                if os.path.isfile(history_path):
                    with open(history_path, "r", encoding="utf-8") as f:
                        history = json.load(f)

                history.append(existing)

                with open(history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        # Write current config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def _sync_state_with_models(self) -> bool:
        """
        Synchronize arena state with current available models.

        If models have been removed from the models directory, this method will:
        1. Detect removed models (from both state and pk_logs)
        2. Move battle logs involving removed models to .pk_logs_rm/ (if clean_orphaned_logs=True)
        3. Rebuild ELO state from remaining battle logs
        4. Save the updated state

        Returns:
            True if state was rebuilt due to model changes, False otherwise
        """
        state = load_state(self.state_path)
        # Use the canonical on-disk model set (do NOT treat --models filter as removals)
        current_models = set(self.all_models)

        # Get models that exist in state but not in current model list
        state_models = set(state.models.keys())
        removed_from_state = state_models - current_models

        # Also scan pk_logs to find models that exist in logs but not in models/
        logs_models = self._scan_models_from_logs()
        removed_from_logs = logs_models - current_models

        # Combine both sources of removed models
        removed_models = removed_from_state | removed_from_logs

        if not removed_models:
            return False

        logger.info(
            f"Detected removed models: {removed_models}. "
            f"Rebuilding ELO state from battle logs..."
        )

        # Clean up orphaned battle logs if enabled
        if self.config.clean_orphaned_logs:
            self._delete_orphaned_logs(removed_models)

        # Rebuild state from logs, only including current models
        new_state = rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)

        # Save the rebuilt state
        save_state(new_state, self.state_path)

        logger.info(
            f"State rebuilt: {new_state.total_battles} battles, "
            f"{len(new_state.models)} models"
        )

        return True

    def _scan_models_from_logs(self) -> set[str]:
        """
        Scan all battle log files to extract model names.

        This method reads the actual content of jsonl files to get the original
        model names, which is more reliable than parsing sanitized filenames.

        Returns:
            Set of all model names found in battle logs
        """
        models_found: set[str] = set()

        if not os.path.isdir(self.pk_logs_dir):
            return models_found

        for exp_name in os.listdir(self.pk_logs_dir):
            exp_dir = os.path.join(self.pk_logs_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            for filename in os.listdir(exp_dir):
                if not filename.endswith(".jsonl"):
                    continue

                filepath = os.path.join(exp_dir, filename)
                if not os.path.isfile(filepath):
                    continue

                # Read first line to extract model names (all lines in a file
                # should have the same model pair)
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
                                if model_a:
                                    models_found.add(model_a)
                                if model_b:
                                    models_found.add(model_b)
                                # Only need first valid line per file
                                break
                            except json.JSONDecodeError:
                                continue
                except Exception:
                    pass

        return models_found

    def _delete_orphaned_logs(self, removed_models: set[str]) -> None:
        """
        Move battle log files that involve removed models to .pk_logs_rm/ directory.

        This method reads the actual content of each jsonl file to extract the
        original model names, which is more reliable than parsing sanitized
        filenames. Instead of deleting, files are moved to a backup directory
        (.pk_logs_rm/) at the same level as pk_logs/.

        Args:
            removed_models: Set of model names that have been removed
        """
        import shutil

        if not os.path.isdir(self.pk_logs_dir):
            return

        # Create backup directory at the same level as pk_logs
        pk_logs_rm_dir = os.path.join(self.subset_dir, ".pk_logs_rm")

        moved_count = 0

        def _file_involves_removed_model(filepath: str) -> bool:
            """
            Check if a jsonl file involves any removed model by reading its content.

            Returns True if any record in the file has model_a or model_b in removed_models.
            """
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
                            if model_a in removed_models or model_b in removed_models:
                                return True
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
            return False

        def _move_to_backup(filepath: str, relative_path: str) -> bool:
            """
            Move a file to the backup directory, preserving relative path structure.

            Args:
                filepath: Absolute path to the source file
                relative_path: Relative path from pk_logs_dir (e.g., "exp_name/file.jsonl")

            Returns:
                True if moved successfully, False otherwise
            """
            dest_path = os.path.join(pk_logs_rm_dir, relative_path)
            dest_dir = os.path.dirname(dest_path)

            try:
                ensure_dir(dest_dir)
                shutil.move(filepath, dest_path)
                return True
            except Exception as e:
                logger.warning(f"Failed to move {filepath} to {dest_path}: {e}")
                return False

        # Iterate over all experiment directories
        for exp_name in os.listdir(self.pk_logs_dir):
            exp_dir = os.path.join(self.pk_logs_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            # Check battle log files (format: model_a_vs_model_b.jsonl)
            for filename in os.listdir(exp_dir):
                if not filename.endswith(".jsonl"):
                    continue

                filepath = os.path.join(exp_dir, filename)
                if not os.path.isfile(filepath):
                    continue

                # Check file content to determine if it involves removed models
                if _file_involves_removed_model(filepath):
                    relative_path = os.path.join(exp_name, filename)
                    if _move_to_backup(filepath, relative_path):
                        moved_count += 1
                        logger.debug(f"Moved orphaned log to backup: {filepath}")

            # Also check raw_outputs subdirectory
            raw_outputs_dir = os.path.join(exp_dir, "raw_outputs")
            if os.path.isdir(raw_outputs_dir):
                for filename in os.listdir(raw_outputs_dir):
                    if not filename.endswith(".jsonl"):
                        continue

                    filepath = os.path.join(raw_outputs_dir, filename)
                    if not os.path.isfile(filepath):
                        continue

                    if _file_involves_removed_model(filepath):
                        relative_path = os.path.join(exp_name, "raw_outputs", filename)
                        if _move_to_backup(filepath, relative_path):
                            moved_count += 1
                            logger.debug(f"Moved orphaned audit log to backup: {filepath}")

        if moved_count > 0:
            logger.info(f"Moved {moved_count} orphaned battle log files to {pk_logs_rm_dir}")

    def _generate_battle_pairs(self) -> list[BattlePair]:
        """
        Generate all battle pairs to execute.
        
        In full mode: generates all possible pairs up to sample_size.
        In adaptive mode: generates pairs based on sampling config, respecting
        min_samples and max_samples per model pair.

        Returns:
            List of BattlePair objects
        """
        pairs = []

        # Get all dataset indices
        all_indices = self.dataset.get_all_indices()

        # In full mode, apply global sample_size limit
        # In adaptive mode, we apply per-pair limits later
        if self.config.sampling.mode == "full":
            if self.config.sample_size and self.config.sample_size < len(all_indices):
                indices = random.sample(all_indices, self.config.sample_size)
            else:
                indices = all_indices
        else:
            # Adaptive mode: use all indices, will limit per-pair
            indices = all_indices

        # Generate model pairs to run for this exp:
        # - only include pairs where at least one side is a "new model" in this exp
        # - but respect the user-provided --models filter (self.models)
        selected_models = set(self.models)
        new_models = [m for m in self.new_models if m in selected_models]

        if not new_models:
            return []

        # Build unique pair set (sorted) for: new-vs-all + new-vs-new
        pair_set: set[tuple[str, str]] = set()
        for m in new_models:
            for other in selected_models:
                if other == m:
                    continue
                a, b, _ = get_sorted_model_pair(m, other)
                pair_set.add((a, b))

        model_pairs = sorted(pair_set)

        # Load existing battle counts for adaptive mode
        if self.config.sampling.mode == "adaptive":
            existing_counts = count_battles_per_pair(self.pk_logs_dir)
            # Determine target samples per pair based on experiment type
            if is_milestone_exp(self.config.exp_name or ""):
                target_samples = self.config.sampling.milestone_min_samples
            else:
                target_samples = self.config.sampling.min_samples
        else:
            existing_counts = {}
            target_samples = None

        # Generate battle pairs for each model pair and sample
        for model_a, model_b in model_pairs:
            # Validate coverage
            valid_indices = self.model_manager.validate_coverage(
                model_a, model_b, indices
            )

            # In adaptive mode, limit samples per pair
            if self.config.sampling.mode == "adaptive" and target_samples is not None:
                key = (min(model_a, model_b), max(model_a, model_b))
                existing = existing_counts.get(key, 0)
                needed = max(0, target_samples - existing)
                
                if needed == 0:
                    continue  # This pair already has enough samples
                
                # Limit to needed samples (randomly select if more available)
                if len(valid_indices) > needed:
                    valid_indices = random.sample(valid_indices, needed)

            for idx in valid_indices:
                pairs.append(BattlePair(
                    model_a=model_a,
                    model_b=model_b,
                    sample_index=idx
                ))

        return pairs

    def _skip_completed(
        self,
        pairs: list[BattlePair]
    ) -> list[BattlePair]:
        """
        Filter out already completed battles.

        Only considers battles where both models still exist in the current
        model list. Battles involving removed models are ignored.

        Args:
            pairs: List of battle pairs

        Returns:
            Filtered list excluding completed battles
        """
        all_completed = load_battle_history(self.pk_logs_dir)

        # Filter completed battles to only include those with current on-disk models.
        # This avoids treating --models filters as removals.
        current_models = set(self.all_models)
        completed = {
            (m_a, m_b, idx)
            for m_a, m_b, idx in all_completed
            if m_a in current_models and m_b in current_models
        }

        remaining = []
        for pair in pairs:
            # Get sorted model names for lookup
            first, second, _ = get_sorted_model_pair(pair.model_a, pair.model_b)
            key = (first, second, pair.sample_index)

            if key not in completed:
                remaining.append(pair)

        skipped = len(pairs) - len(remaining)
        if skipped > 0:
            logger.info(f"Skipping {skipped} already completed battles")

        # Log if there are orphaned battles from removed models
        orphaned = len(all_completed) - len(completed)
        if orphaned > 0:
            logger.info(
                f"Ignoring {orphaned} battle records involving removed models"
            )

        return remaining

    def _execute_single_battle(
        self,
        pair: BattlePair
    ) -> Optional[BattleResult]:
        """
        Execute a single battle.

        Args:
            pair: BattlePair to execute

        Returns:
            BattleResult or None if failed
        """
        try:
            # Get sample data
            sample = self.dataset.get_by_index(pair.sample_index)
            if sample is None:
                logger.warning(
                    f"Sample {pair.sample_index} not found in dataset"
                )
                return None

            # Get model outputs
            output_a = self.model_manager.get_output_path(
                pair.model_a, pair.sample_index
            )
            output_b = self.model_manager.get_output_path(
                pair.model_b, pair.sample_index
            )

            if output_a is None or output_b is None:
                logger.warning(
                    f"Missing output for battle {pair.model_a} vs {pair.model_b} "
                    f"at index {pair.sample_index}"
                )
                return None

            # Execute battle
            result = execute_battle(
                vlm=self.vlm,
                prompt_module=self.prompt_module,
                sample=sample,
                model_a_output=output_a,
                model_b_output=output_b,
                model_a=pair.model_a,
                model_b=pair.model_b,
                parallel_swap_calls=self.config.parallel_swap_calls,
                progress=self._progress_queue,
            )

            return result

        except Exception as e:
            logger.error(
                f"Error executing battle {pair.model_a} vs {pair.model_b} "
                f"at index {pair.sample_index}: {e}"
            )
            return None

    def _process_result(
        self,
        result: BattleResult,
        state: ArenaState
    ) -> ArenaState:
        """
        Process a battle result: log and update state.

        Args:
            result: BattleResult from battle execution
            state: Current arena state

        Returns:
            Updated arena state
        """
        # Log battle result (slim)
        self.battle_logger.log_battle_result(result)

        # Log audit trail (detailed)
        if self.audit_logger:
            self.audit_logger.log_battle_result(result)

        # Update W/L/T stats only. Elo is recomputed via Bradley-Terry fitting
        # from accumulated battle logs (order-independent).
        state = update_stats(state, result.model_a, result.model_b, result.final_winner)

        return state

    def _get_battles_from_logs(self) -> list[BattleTuple]:
        """
        Load battle records from logs and convert to BattleTuple format.
        
        Returns:
            List of (model_a, model_b, winner) tuples for BT-Elo computation.
        """
        records = load_battle_records(self.pk_logs_dir)
        battles: list[BattleTuple] = []
        
        current_models = set(self.all_models)
        
        for record in records:
            model_a = record.get("model_a", "")
            model_b = record.get("model_b", "")
            final_winner = record.get("final_winner", "")
            
            # Skip records involving removed models
            if model_a not in current_models or model_b not in current_models:
                continue
            
            # Convert winner to standard format
            if final_winner == model_a:
                winner = "model_a"
            elif final_winner == model_b:
                winner = "model_b"
            elif final_winner == "tie":
                winner = "tie"
            else:
                continue  # Skip invalid records
            
            battles.append((model_a, model_b, winner))
        
        return battles

    def _load_anchor_elo(self) -> dict[str, float]:
        """
        Load anchor ELO ratings from the latest milestone snapshot.
        
        Returns:
            Dict mapping model name to ELO rating for milestone models,
            or empty dict if no milestone exists.
        """
        # Discover milestone experiments
        exp_keys: list[tuple[tuple, str]] = []
        if not os.path.isdir(self.pk_logs_dir):
            return {}
        
        for name in os.listdir(self.pk_logs_dir):
            if name.startswith("."):
                continue
            exp_dir = os.path.join(self.pk_logs_dir, name)
            if not os.path.isdir(exp_dir):
                continue
            d = parse_exp_date_suffix(name)
            if d is None:
                continue
            exp_keys.append(((d, name), name))
        
        exp_keys.sort(key=lambda x: x[0])
        
        # Find milestones
        milestones = [name for (key, name) in exp_keys if is_milestone_exp(name)]
        if not milestones:
            return {}
        
        # Load from latest milestone snapshot
        latest_milestone = milestones[-1]
        snapshot_path = os.path.join(self.pk_logs_dir, latest_milestone, "elo_snapshot.json")
        
        if not os.path.isfile(snapshot_path):
            return {}
        
        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {}
        
        if not isinstance(data, dict):
            return {}
        
        # Accept either: {"elo": {...}} or a direct {model: elo} mapping
        raw = data.get("elo") if isinstance(data.get("elo"), dict) else data
        if not isinstance(raw, dict):
            return {}
        
        # Filter to only include models that exist in current model set
        current_models = set(self.all_models)
        anchor_elo: dict[str, float] = {}
        for k, v in raw.items():
            if str(k) in current_models:
                try:
                    anchor_elo[str(k)] = float(v)
                except Exception:
                    continue
        
        return anchor_elo

    def _run_adaptive_with_ci_checking(self) -> ArenaState:
        """
        Run arena evaluation with adaptive CI-based sampling.
        
        This method implements the iterative loop:
        1. Run initial batch (min_samples per pair)
        2. Compute bootstrap CI
        3. If max CI width > target, add batch_size more samples to unconverged pairs
        4. Repeat until all pairs converge or reach max_samples
        
        Returns:
            Final ArenaState after all battles
        """
        sampling_config = self.config.sampling
        is_milestone = is_milestone_exp(self.config.exp_name or "")
        
        # Determine target samples per pair for initial batch
        if is_milestone:
            target_samples = sampling_config.milestone_min_samples
            logger.info(f"Milestone experiment: targeting {target_samples} samples/pair initially")
        else:
            target_samples = sampling_config.min_samples
            logger.info(f"Incremental experiment: targeting {target_samples} samples/pair initially")
        
        # Get all dataset indices
        all_indices = self.dataset.get_all_indices()
        
        # Build model pairs (new models vs all selected models)
        selected_models = set(self.models)
        new_models = [m for m in self.new_models if m in selected_models]
        
        if not new_models:
            logger.info("No new models to evaluate")
            return rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)
        
        # Build unique pair set
        pair_set: set[tuple[str, str]] = set()
        for m in new_models:
            for other in selected_models:
                if other == m:
                    continue
                a, b, _ = get_sorted_model_pair(m, other)
                pair_set.add((a, b))
        
        model_pairs = sorted(pair_set)
        logger.info(f"Evaluating {len(model_pairs)} model pairs with adaptive sampling")
        
        # Initialize scheduler
        scheduler = AdaptiveSamplingScheduler(config=sampling_config)
        
        # Load existing battle counts
        existing_counts = count_battles_per_pair(self.pk_logs_dir)
        for pair in model_pairs:
            count = existing_counts.get(pair, 0)
            scheduler.update_state(pair[0], pair[1], current_samples=count)
        
        # Load existing state
        state = load_state(self.state_path)
        
        # Progress tracking
        progress_queue, _progress_thread, progress_close = _start_calls_progress_consumer(
            enabled=self.config.enable_progress_bar,
            total=None,  # Dynamic total
        )
        self._progress_queue = progress_queue
        if self._progress_queue is not None:
            try:
                self.vlm.set_progress(self._progress_queue)
            except Exception:
                pass
        
        iteration = 0
        total_completed = 0
        
        while True:
            iteration += 1
            
            # Determine which pairs need more samples
            pairs_to_run: list[tuple[str, str]] = []
            samples_per_pair: dict[tuple[str, str], int] = {}
            
            for pair in model_pairs:
                pair_state = scheduler.get_or_create_state(pair[0], pair[1])
                samples_to_run = pair_state.get_samples_to_run(sampling_config, len(all_indices))
                
                if samples_to_run > 0:
                    pairs_to_run.append(pair)
                    samples_per_pair[pair] = samples_to_run
            
            if not pairs_to_run:
                logger.info("All pairs have converged or reached max_samples")
                break
            
            total_samples_this_iter = sum(samples_per_pair.values())
            logger.info(
                f"Iteration {iteration}: running {total_samples_this_iter} battles "
                f"across {len(pairs_to_run)} pairs"
            )
            
            # Generate battle pairs for this iteration
            completed_set = load_battle_history(self.pk_logs_dir)
            battle_pairs: list[BattlePair] = []
            
            for pair in pairs_to_run:
                model_a, model_b = pair
                needed = samples_per_pair[pair]
                
                # Get valid indices for this pair
                valid_indices = self.model_manager.validate_coverage(model_a, model_b, all_indices)
                
                # Filter out already completed
                pending_indices = [
                    idx for idx in valid_indices
                    if (model_a, model_b, idx) not in completed_set
                ]
                
                # Select up to 'needed' samples
                if len(pending_indices) > needed:
                    selected = random.sample(pending_indices, needed)
                else:
                    selected = pending_indices
                
                for idx in selected:
                    battle_pairs.append(BattlePair(
                        model_a=model_a,
                        model_b=model_b,
                        sample_index=idx
                    ))
            
            if not battle_pairs:
                logger.info("No more battles to execute")
                break
            
            # Update progress bar total
            if self._progress_queue is not None:
                try:
                    self._progress_queue.put(("total", 2 * len(battle_pairs)))
                except Exception:
                    pass
            
            # Execute battles
            iter_completed = 0
            
            if self.config.num_threads <= 1:
                # Sequential execution
                for pair in battle_pairs:
                    result = self._execute_single_battle(pair)
                    if result:
                        state = self._process_result(result, state)
                        iter_completed += 1
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
                    future_to_pair = {
                        executor.submit(self._execute_single_battle, pair): pair
                        for pair in battle_pairs
                    }
                    
                    for future in as_completed(future_to_pair):
                        try:
                            result = future.result()
                            if result:
                                state = self._process_result(result, state)
                                iter_completed += 1
                        except Exception as e:
                            pair = future_to_pair[future]
                            logger.error(f"Battle {pair.model_a} vs {pair.model_b} failed: {e}")
            
            total_completed += iter_completed
            logger.info(f"Iteration {iteration} completed: {iter_completed} battles")
            
            # Save intermediate state
            save_state(state, self.state_path)
            
            # Update scheduler with new counts
            new_counts = count_battles_per_pair(self.pk_logs_dir)
            for pair in model_pairs:
                count = new_counts.get(pair, 0)
                scheduler.update_state(pair[0], pair[1], current_samples=count)
            
            # Compute bootstrap CI to check convergence
            battles = self._get_battles_from_logs()
            if battles:
                # Load anchor ELO from latest milestone snapshot
                # Milestone models have fixed ELO, so we only check CI for new models
                anchor_elo = self._load_anchor_elo()
                
                bootstrap_result = compute_bootstrap_bt_elo(
                    battles,
                    models=self.all_models,
                    fixed_ratings=anchor_elo if anchor_elo else None,
                    num_bootstrap=sampling_config.num_bootstrap,
                )
                
                # Only check CI for new models (non-anchor models)
                # Anchor models have CI width = 0 since their ELO is fixed
                new_models_set = set(new_models)
                new_model_ci_widths = [
                    bootstrap_result.ci_width.get(m, 0.0)
                    for m in new_models_set
                    if m in bootstrap_result.ci_width
                ]
                
                if new_model_ci_widths:
                    max_ci_width = max(new_model_ci_widths)
                    mean_ci_width = sum(new_model_ci_widths) / len(new_model_ci_widths)
                else:
                    max_ci_width = bootstrap_result.get_max_ci_width()
                    mean_ci_width = bootstrap_result.get_mean_ci_width()
                
                logger.info(
                    f"CI check (new models only): max_width={max_ci_width:.2f}, "
                    f"mean_width={mean_ci_width:.2f}, target={sampling_config.target_ci_width:.2f}"
                )
                
                # Check if all new models have converged
                if max_ci_width <= sampling_config.target_ci_width:
                    logger.info(f"CI target reached! Max CI width for new models: {max_ci_width:.2f}")
                    # Mark all pairs as converged
                    for pair in model_pairs:
                        pair_state = scheduler.get_or_create_state(pair[0], pair[1])
                        pair_state.converged = True
                    break
            
            # Check if all pairs have reached max_samples
            all_maxed = True
            for pair in model_pairs:
                pair_state = scheduler.get_or_create_state(pair[0], pair[1])
                if pair_state.current_samples < sampling_config.max_samples:
                    all_maxed = False
                    break
            
            if all_maxed:
                logger.info("All pairs reached max_samples limit")
                break
        
        progress_close()
        
        # Final summary
        summary = scheduler.get_summary()
        logger.info(
            f"Adaptive sampling complete: "
            f"{summary['total_pairs']} pairs, "
            f"{summary['converged_pairs']} converged, "
            f"{summary['maxed_pairs']} reached max_samples, "
            f"{summary['total_samples']} total samples"
        )
        
        # Final Elo recompute (Bradley-Terry) and state save
        final_state = rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)
        save_state(final_state, self.state_path)
        
        logger.info(f"Arena completed: {total_completed} battles executed in {iteration} iterations")
        
        return final_state

    def run(self) -> ArenaState:
        """
        Run the arena evaluation.

        If models have been removed from the arena directory, the ELO state
        will be automatically rebuilt from battle logs (excluding removed models).

        Returns:
            Final ArenaState after all battles
        """
        # Sync state with current models (rebuild if models were removed).
        # This rebuild uses Bradley-Terry Elo scoring from logs.
        self._sync_state_with_models()
        
        # Use adaptive CI-checking mode if enabled (and not multiprocessing)
        if (self.config.sampling.mode == "adaptive" and 
            self.config.num_processes <= 1):
            return self._run_adaptive_with_ci_checking()

        # Generate and filter battle pairs
        # If we can shard by parquet file, we can avoid constructing the full pair list
        # in the parent process (and avoid pickling huge lists).
        all_indices = self.dataset.get_all_indices()

        # Apply sample size limit
        if self.config.sample_size and self.config.sample_size < len(all_indices):
            indices = random.sample(all_indices, self.config.sample_size)
        else:
            indices = all_indices

        # If num_processes <= 1, fall back to the original thread-based implementation.
        if self.config.num_processes <= 1:
            all_pairs = self._generate_battle_pairs()
            pairs = self._skip_completed(all_pairs)

            if not pairs:
                logger.info("No battles to execute")
                # Ensure state is up-to-date and order-independent
                state = rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)
                save_state(state, self.state_path)
                return state

            logger.info(f"Starting arena with {len(pairs)} battles to execute")
            logger.info(f"Models: {self.models}")
            logger.info(f"Experiment: {self.config.exp_name}")
            logger.info(f"Sampling mode: full")

            # Load existing state
            state = load_state(self.state_path)

            # Progress tracking
            completed = 0
            total = len(pairs)

            progress_queue, _progress_thread, progress_close = _start_calls_progress_consumer(
                enabled=self.config.enable_progress_bar,
                total=(2 * len(pairs)) if self.config.enable_progress_bar else None,
            )
            self._progress_queue = progress_queue
            if self._progress_queue is not None:
                try:
                    self.vlm.set_progress(self._progress_queue)
                except Exception:
                    pass

            if self.config.num_threads <= 1:
                # Sequential execution
                for pair in pairs:
                    result = self._execute_single_battle(pair)

                    if result:
                        state = self._process_result(result, state)
                        completed += 1

                        # Progress logging every 10 battles
                        if completed % 10 == 0:
                            logger.info(f"Progress: {completed}/{total} battles")
                            # Save intermediate state
                            save_state(state, self.state_path)
            else:
                # Parallel execution
                with ThreadPoolExecutor(max_workers=self.config.num_threads) as executor:
                    # Submit all battles
                    future_to_pair = {
                        executor.submit(self._execute_single_battle, pair): pair
                        for pair in pairs
                    }

                    # Process completed futures
                    for future in as_completed(future_to_pair):
                        pair = future_to_pair[future]

                        try:
                            result = future.result()

                            if result:
                                state = self._process_result(result, state)
                                completed += 1

                                # Progress logging every 10 battles
                                if completed % 10 == 0:
                                    logger.info(f"Progress: {completed}/{total} battles")
                                    # Save intermediate state
                                    save_state(state, self.state_path)
                        except Exception as e:
                            logger.error(
                                f"Battle {pair.model_a} vs {pair.model_b} failed: {e}"
                            )

            progress_close()

            # Final Elo recompute (Bradley-Terry) and state save
            final_state = rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)
            save_state(final_state, self.state_path)

            logger.info(f"Arena completed: {completed}/{total} battles executed")

            return final_state

        # === Multiprocessing path (per-parquet sharding) ===
        grouped = self.dataset.group_indices_by_parquet(indices)
        if "" in grouped:
            logger.warning(
                "Parquet source mapping is incomplete (missing index->parquet mapping). "
                "Falling back to single-process execution."
            )
            self.config.num_processes = 1
            # Re-load dataset in full mode for single-process execution.
            self.dataset = ParquetDataset(self.config.data_dir, self.config.subset, load_mode="full")
            return self.run()

        total_concurrency = max(1, int(self.config.num_processes)) * max(1, int(self.config.num_threads))
        logger.info(
            f"Starting arena with multiprocessing: num_processes={self.config.num_processes}, "
            f"num_threads={self.config.num_threads}, total_concurrency~{total_concurrency}"
        )
        logger.info(f"Models: {self.models}")
        logger.info(f"Experiment: {self.config.exp_name}")

        completed = 0
        attempted = 0
        parquet_tasks = [(pf, idxs) for pf, idxs in grouped.items() if idxs]
        parquet_tasks.sort(key=lambda x: x[0])

        # Assign parquet files to processes up-front (avoid per-parquet re-init overhead in a worker).
        # Simple greedy bin-packing by number of indices for load balancing.
        num_workers = max(1, int(self.config.num_processes))
        buckets: list[list[tuple[str, list[int]]]] = [[] for _ in range(num_workers)]
        bucket_sizes = [0 for _ in range(num_workers)]
        for pf, idxs in sorted(parquet_tasks, key=lambda x: len(x[1]), reverse=True):
            k = bucket_sizes.index(min(bucket_sizes))
            buckets[k].append((pf, idxs))
            bucket_sizes[k] += len(idxs)

        # Progress consumer (optional). Use a process-safe Manager queue and batch updates in workers.
        manager = None
        mp_progress_queue = None
        progress_close = lambda: None
        if self.config.enable_progress_bar:
            try:
                import multiprocessing
                manager = multiprocessing.Manager()
                mp_progress_queue = manager.Queue()
                # Capture the queue reference for the closure (type narrowing)
                _queue = mp_progress_queue
                # Reuse same tqdm consumer code by wrapping manager queue into a local consumer thread.
                try:
                    from tqdm import tqdm  # type: ignore
                    bar = tqdm(total=None, unit="call", desc="API Calls", dynamic_ncols=True)
                    # Must be picklable across processes.
                    stop_sentinel = ("stop", None)
                    recent: deque[str] = deque(maxlen=10)

                    def _mp_consumer() -> None:
                        while True:
                            item = _queue.get()
                            if item == stop_sentinel:
                                break
                            if isinstance(item, (int, float)):
                                try:
                                    bar.update(int(item))
                                except Exception:
                                    pass
                            elif isinstance(item, tuple) and len(item) == 2 and item[0] == "log":
                                try:
                                    recent.append(str(item[1]))
                                    bar.set_postfix_str(" | ".join(recent))
                                except Exception:
                                    pass
                            elif isinstance(item, tuple) and len(item) == 2 and item[0] == "total":
                                try:
                                    delta = int(item[1])
                                    if delta > 0:
                                        bar.total = (bar.total or 0) + delta
                                        bar.refresh()
                                except Exception:
                                    pass
                            else:
                                pass

                    t = threading.Thread(target=_mp_consumer, name="mp-calls-progress-consumer", daemon=True)
                    t.start()

                    def progress_close() -> None:
                        try:
                            mp_progress_queue.put(stop_sentinel)
                        except Exception:
                            pass
                        t.join(timeout=5)
                        try:
                            bar.close()
                        except Exception:
                            pass

                except Exception:
                    logger.warning("tqdm is not available; progress bar disabled")
                    mp_progress_queue = None
            except Exception:
                logger.warning("Failed to initialize multiprocessing progress queue; progress bar disabled")
                mp_progress_queue = None

        with ProcessPoolExecutor(max_workers=self.config.num_processes) as executor:
            futures = []
            for work in buckets:
                if not work:
                    continue
                futures.append(executor.submit(
                    _run_parquet_bucket_worker,
                    arena_dir=self.config.arena_dir,
                    data_dir=self.config.data_dir,
                    subset=self.config.subset,
                    exp_name=self.config.exp_name or "",  # exp_name is guaranteed to be set in __init__
                    parquet_work=work,
                    models=self.models,
                    new_models=self.new_models,
                    num_threads=self.config.num_threads,
                    judge_model=self.config.judge_model,
                    temperature=self.config.temperature,
                    prompt=self.config.prompt,
                    timeout=self.config.timeout,
                    max_retries=self.config.max_retries,
                    base_urls=self.config.base_urls,
                    api_keys=self.config.api_keys,
                    enable_audit_log=self.config.enable_audit_log,
                    parallel_swap_calls=self.config.parallel_swap_calls,
                    progress_queue=mp_progress_queue,
                ))

            for fut in as_completed(futures):
                try:
                    res = fut.result()
                    completed += int(res.get("completed", 0))
                    attempted += int(res.get("attempted", 0))
                    if completed > 0 and completed % 50 == 0:
                        logger.info(f"Progress: completed={completed} attempted={attempted}")
                except Exception as e:
                    logger.error(f"Worker failed: {e}")

        progress_close()
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass

        # Final Elo recompute (Bradley-Terry) and state save
        final_state = rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)
        save_state(final_state, self.state_path)

        logger.info(f"Arena completed (multiprocessing): completed={completed} attempted={attempted}")

        return final_state

    def update_leaderboard(self) -> None:
        """Update the leaderboard README.md file."""
        # Always rebuild state from logs to ensure BT Elo is consistent and up-to-date.
        state = rebuild_state_from_logs(self.pk_logs_dir, models=self.all_models)
        save_state(state, self.state_path)

        title = f"{self.config.subset.capitalize()} Leaderboard"
        save_leaderboard(state, self.leaderboard_path, title)

        logger.info(f"Leaderboard saved to {self.leaderboard_path}")

    def get_status(self) -> dict[str, Any]:
        """
        Get arena status summary.

        Returns:
            Dict with status information
        """
        state = load_state(self.state_path)

        return {
            "subset": self.config.subset,
            "models": self.models,
            "total_models": len(self.models),
            "total_battles": state.total_battles,
            "last_updated": state.last_updated,
            "dataset_size": len(self.dataset),
            "arena_dir": self.config.arena_dir
        }


def get_all_subsets_status(arena_dir: str, data_dir: str) -> list[dict[str, Any]]:
    """
    Get status for all subsets in an arena directory.

    Args:
        arena_dir: Arena directory path
        data_dir: Data directory path

    Returns:
        List of status dicts for each subset
    """
    subsets = discover_subsets(data_dir)
    statuses = []

    for subset in subsets:
        state_path = os.path.join(arena_dir, subset, "arena", "state.json")
        state = load_state(state_path)

        models_dir = os.path.join(arena_dir, subset, "models")
        model_manager = GlobalModelOutputManager(models_dir)

        statuses.append({
            "subset": subset,
            "models": model_manager.models,
            "total_models": len(model_manager.models),
            "total_battles": state.total_battles,
            "last_updated": state.last_updated
        })

    return statuses
