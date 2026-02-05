# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Data loader for arena visualization with preloading support."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from genarena.data import DataSample, ParquetDataset, discover_subsets
from genarena.models import GlobalModelOutputManager
from genarena.state import ArenaState, load_state


logger = logging.getLogger(__name__)


@dataclass
class BattleRecord:
    """A single battle record with all relevant information."""

    # Battle identification
    subset: str
    exp_name: str
    sample_index: int
    model_a: str
    model_b: str

    # Battle result
    final_winner: str  # model name or "tie"
    is_consistent: bool
    timestamp: str = ""

    # Raw VLM outputs (from audit logs, optional)
    original_call: Optional[dict[str, Any]] = None
    swapped_call: Optional[dict[str, Any]] = None

    # Sample data (loaded on demand)
    instruction: str = ""
    task_type: str = ""
    input_image_count: int = 1
    prompt_source: Optional[str] = None
    original_metadata: Optional[dict[str, Any]] = None

    @property
    def id(self) -> str:
        """Unique identifier for this battle."""
        return f"{self.subset}:{self.exp_name}:{self.model_a}_vs_{self.model_b}:{self.sample_index}"

    @property
    def winner_display(self) -> str:
        """Display-friendly winner string."""
        if self.final_winner == "tie":
            return "Tie"
        return self.final_winner

    @property
    def models(self) -> set[str]:
        """Set of models involved in this battle."""
        return {self.model_a, self.model_b}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "subset": self.subset,
            "exp_name": self.exp_name,
            "sample_index": self.sample_index,
            "model_a": self.model_a,
            "model_b": self.model_b,
            "final_winner": self.final_winner,
            "winner_display": self.winner_display,
            "is_consistent": self.is_consistent,
            "timestamp": self.timestamp,
            "instruction": self.instruction,
            "task_type": self.task_type,
            "input_image_count": self.input_image_count,
            "prompt_source": self.prompt_source,
            "original_metadata": self.original_metadata,
            "has_audit": self.original_call is not None,
        }

    def to_detail_dict(self) -> dict[str, Any]:
        """Convert to detailed dictionary including VLM outputs."""
        d = self.to_dict()
        d["original_call"] = self.original_call
        d["swapped_call"] = self.swapped_call
        return d


@dataclass
class SubsetInfo:
    """Information about a subset."""

    name: str
    models: list[str]
    experiments: list[str]
    total_battles: int
    state: Optional[ArenaState] = None
    min_input_images: int = 1
    max_input_images: int = 1
    prompt_sources: list[str] = field(default_factory=list)


class ArenaDataLoader:
    """
    Data loader for arena visualization.

    Manages loading and querying battle records across multiple subsets.
    Supports preloading for better performance with large datasets.
    """

    def __init__(self, arena_dir: str, data_dir: str, preload: bool = True):
        """
        Initialize the data loader.

        Args:
            arena_dir: Path to arena directory containing subset folders
            data_dir: Path to data directory containing parquet files
            preload: If True, preload all data at initialization
        """
        self.arena_dir = arena_dir
        self.data_dir = data_dir

        # Cached data
        self._subsets: Optional[list[str]] = None
        self._subset_info_cache: dict[str, SubsetInfo] = {}
        self._dataset_cache: dict[str, ParquetDataset] = {}
        self._model_manager_cache: dict[str, GlobalModelOutputManager] = {}

        # Battle records cache: (subset, exp_name) -> List[BattleRecord]
        self._battle_cache: dict[tuple[str, str], list[BattleRecord]] = {}

        # Index for faster lookups: (subset, exp_name) -> {model -> [record_indices]}
        self._model_index: dict[tuple[str, str], dict[str, list[int]]] = {}

        # Sample data cache: (subset, sample_index) -> SampleMetadata dict
        self._sample_cache: dict[tuple[str, int], dict[str, Any]] = {}

        # Sample to parquet file mapping: (subset, sample_index) -> parquet_file_path
        self._sample_file_map: dict[tuple[str, int], str] = {}

        # Input image count range per subset: subset -> (min_count, max_count)
        self._image_count_range: dict[str, tuple[int, int]] = {}

        # Prompt sources per subset: subset -> list of unique prompt_source values
        self._prompt_sources: dict[str, list[str]] = {}

        # Audit logs cache: (subset, exp_name, model_a, model_b, sample_index) -> audit data
        self._audit_cache: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}

        # Cross-subset ELO cache: (sorted_subsets_tuple, exp_name, model_scope) -> result dict
        self._cross_subset_elo_cache: dict[tuple[tuple[str, ...], str, str], dict[str, Any]] = {}

        if preload:
            self._preload_all()

    def _preload_all(self) -> None:
        """Preload all data at initialization for better performance."""
        logger.info("Preloading arena data...")

        subsets = self.discover_subsets()
        logger.info(f"Found {len(subsets)} subsets: {subsets}")

        for subset in subsets:
            logger.info(f"Loading subset: {subset}")

            # Preload parquet dataset
            self._preload_dataset(subset)

            # Load subset info (models, experiments)
            info = self.get_subset_info(subset)
            if info:
                logger.info(f"  - {len(info.models)} models, {len(info.experiments)} experiments")

                # Preload battle logs for each experiment
                for exp_name in info.experiments:
                    records = self._load_battle_logs(subset, exp_name)
                    logger.info(f"  - Experiment '{exp_name}': {len(records)} battles")

        logger.info("Preloading complete!")

    def _preload_dataset(self, subset: str) -> None:
        """
        Preload sample text data (instruction, task_type) using pyarrow directly.

        This is much faster than using HuggingFace datasets because we skip
        decoding image columns. Images are loaded on-demand when requested.
        """
        import pyarrow.parquet as pq

        subset_path = os.path.join(self.data_dir, subset)
        if not os.path.isdir(subset_path):
            return

        # Find parquet files
        parquet_files = sorted([
            os.path.join(subset_path, f)
            for f in os.listdir(subset_path)
            if f.startswith("data-") and f.endswith(".parquet")
        ])

        if not parquet_files:
            return

        logger.info(f"  - Loading metadata from parquet (fast mode)...")

        # Read all metadata columns + input_images (only to count, not decode)
        columns_to_read = ["index", "instruction", "task_type", "input_images", "prompt_source", "original_metadata"]

        total_rows = 0
        min_img_count = float('inf')
        max_img_count = 0
        prompt_sources_set: set[str] = set()

        for pf in parquet_files:
            try:
                # Get available columns in this file
                import pyarrow.parquet as pq_schema
                schema = pq.read_schema(pf)
                available_columns = [c for c in columns_to_read if c in schema.names]

                # Read the columns we need
                table = pq.read_table(pf, columns=available_columns)

                # Extract columns with defaults
                def get_column(name, default=None):
                    if name in table.column_names:
                        return table.column(name).to_pylist()
                    return [default] * table.num_rows

                indices = get_column("index", 0)
                instructions = get_column("instruction", "")
                task_types = get_column("task_type", "")
                prompt_sources = get_column("prompt_source", None)
                original_metadatas = get_column("original_metadata", None)

                # Handle input_images separately for counting
                has_input_images = "input_images" in table.column_names
                input_images_col = table.column("input_images") if has_input_images else None

                for i, idx in enumerate(indices):
                    idx = int(idx) if idx is not None else i

                    # Count input images without decoding
                    img_count = 0
                    if input_images_col is not None:
                        img_list = input_images_col[i].as_py()
                        img_count = len(img_list) if img_list else 0

                    min_img_count = min(min_img_count, img_count) if img_count > 0 else min_img_count
                    max_img_count = max(max_img_count, img_count)

                    # Track prompt sources
                    ps = prompt_sources[i] if prompt_sources[i] else None
                    if ps:
                        prompt_sources_set.add(str(ps))

                    # Build metadata dict
                    metadata = {
                        "instruction": str(instructions[i]) if instructions[i] else "",
                        "task_type": str(task_types[i]) if task_types[i] else "",
                        "input_image_count": img_count,
                        "prompt_source": ps,
                        "original_metadata": original_metadatas[i] if original_metadatas[i] else None,
                    }

                    self._sample_cache[(subset, idx)] = metadata
                    self._sample_file_map[(subset, idx)] = pf
                    total_rows += 1

            except Exception as e:
                logger.warning(f"Failed to read {pf}: {e}")
                continue

        # Store image count range for this subset
        if total_rows > 0:
            self._image_count_range[subset] = (
                min_img_count if min_img_count != float('inf') else 1,
                max_img_count if max_img_count > 0 else 1
            )

        # Store prompt sources for this subset
        self._prompt_sources[subset] = sorted(prompt_sources_set)

        logger.info(f"  - Cached {total_rows} samples (input images: {self._image_count_range.get(subset, (1,1))}, sources: {len(prompt_sources_set)})")

    def discover_subsets(self) -> list[str]:
        """
        Discover all available subsets.

        A valid subset must exist in both arena_dir (with pk_logs) and data_dir.

        Returns:
            List of subset names
        """
        if self._subsets is not None:
            return self._subsets

        # Get subsets from data_dir (have parquet files)
        data_subsets = set(discover_subsets(self.data_dir))

        # Get subsets from arena_dir (have pk_logs)
        arena_subsets = set()
        if os.path.isdir(self.arena_dir):
            for name in os.listdir(self.arena_dir):
                subset_path = os.path.join(self.arena_dir, name)
                pk_logs_path = os.path.join(subset_path, "pk_logs")
                if os.path.isdir(pk_logs_path):
                    # Check if there are any experiment directories with battle logs
                    for exp_name in os.listdir(pk_logs_path):
                        exp_path = os.path.join(pk_logs_path, exp_name)
                        if os.path.isdir(exp_path):
                            # Check for .jsonl files
                            has_logs = any(
                                f.endswith(".jsonl")
                                for f in os.listdir(exp_path)
                                if os.path.isfile(os.path.join(exp_path, f))
                            )
                            if has_logs:
                                arena_subsets.add(name)
                                break

        # Intersection: must have both data and battle logs
        valid_subsets = sorted(data_subsets & arena_subsets)
        self._subsets = valid_subsets
        return valid_subsets

    def get_subset_info(self, subset: str) -> Optional[SubsetInfo]:
        """
        Get information about a subset.

        Args:
            subset: Subset name

        Returns:
            SubsetInfo or None if subset doesn't exist
        """
        if subset in self._subset_info_cache:
            return self._subset_info_cache[subset]

        subset_path = os.path.join(self.arena_dir, subset)
        if not os.path.isdir(subset_path):
            return None

        # Get models
        model_manager = self._get_model_manager(subset)
        models = model_manager.models if model_manager else []

        # Get experiments
        pk_logs_dir = os.path.join(subset_path, "pk_logs")
        experiments = []
        if os.path.isdir(pk_logs_dir):
            for name in os.listdir(pk_logs_dir):
                exp_path = os.path.join(pk_logs_dir, name)
                if os.path.isdir(exp_path):
                    # Check for battle logs
                    has_logs = any(
                        f.endswith(".jsonl")
                        for f in os.listdir(exp_path)
                        if os.path.isfile(os.path.join(exp_path, f))
                    )
                    if has_logs:
                        experiments.append(name)
        experiments.sort()

        # Load state
        state_path = os.path.join(subset_path, "arena", "state.json")
        state = load_state(state_path)

        # Get image count range
        img_range = self._image_count_range.get(subset, (1, 1))

        # Get prompt sources
        prompt_sources = self._prompt_sources.get(subset, [])

        info = SubsetInfo(
            name=subset,
            models=models,
            experiments=experiments,
            total_battles=state.total_battles,
            state=state,
            min_input_images=img_range[0],
            max_input_images=img_range[1],
            prompt_sources=prompt_sources,
        )

        self._subset_info_cache[subset] = info
        return info

    def _get_dataset(self, subset: str) -> Optional[ParquetDataset]:
        """Get or create ParquetDataset for a subset."""
        if subset not in self._dataset_cache:
            try:
                self._dataset_cache[subset] = ParquetDataset(self.data_dir, subset)
            except Exception:
                return None
        return self._dataset_cache[subset]

    def _get_model_manager(self, subset: str) -> Optional[GlobalModelOutputManager]:
        """Get or create GlobalModelOutputManager for a subset."""
        if subset not in self._model_manager_cache:
            models_dir = os.path.join(self.arena_dir, subset, "models")
            if os.path.isdir(models_dir):
                self._model_manager_cache[subset] = GlobalModelOutputManager(models_dir)
            else:
                return None
        return self._model_manager_cache[subset]

    def _get_sample_data(self, subset: str, sample_index: int) -> dict[str, Any]:
        """Get cached sample metadata."""
        cache_key = (subset, sample_index)
        if cache_key in self._sample_cache:
            return self._sample_cache[cache_key]

        # Fallback - return defaults
        return {
            "instruction": "",
            "task_type": "",
            "input_image_count": 1,
            "prompt_source": None,
            "original_metadata": None,
        }

    def _load_battle_logs(self, subset: str, exp_name: str) -> list[BattleRecord]:
        """
        Load battle records from log files.

        Args:
            subset: Subset name
            exp_name: Experiment name

        Returns:
            List of BattleRecord objects
        """
        cache_key = (subset, exp_name)
        if cache_key in self._battle_cache:
            return self._battle_cache[cache_key]

        records: list[BattleRecord] = []
        exp_dir = os.path.join(self.arena_dir, subset, "pk_logs", exp_name)

        if not os.path.isdir(exp_dir):
            return records

        # Load slim battle logs
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
                            data = json.loads(line)
                            sample_index = data.get("sample_index", -1)

                            # Get cached sample data
                            sample_meta = self._get_sample_data(subset, sample_index)

                            record = BattleRecord(
                                subset=subset,
                                exp_name=exp_name,
                                sample_index=sample_index,
                                model_a=data.get("model_a", ""),
                                model_b=data.get("model_b", ""),
                                final_winner=data.get("final_winner", "tie"),
                                is_consistent=data.get("is_consistent", False),
                                timestamp=data.get("timestamp", ""),
                                instruction=sample_meta.get("instruction", ""),
                                task_type=sample_meta.get("task_type", ""),
                                input_image_count=sample_meta.get("input_image_count", 1),
                                prompt_source=sample_meta.get("prompt_source"),
                                original_metadata=sample_meta.get("original_metadata"),
                            )
                            if record.model_a and record.model_b:
                                records.append(record)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                continue

        # Sort by sample_index
        records.sort(key=lambda r: r.sample_index)

        # Cache records
        self._battle_cache[cache_key] = records

        # Build model index for fast filtering
        self._build_model_index(cache_key, records)

        return records

    def _build_model_index(
        self, cache_key: tuple[str, str], records: list[BattleRecord]
    ) -> None:
        """Build index for fast model-based filtering."""
        model_index: dict[str, list[int]] = {}

        for i, record in enumerate(records):
            for model in [record.model_a, record.model_b]:
                if model not in model_index:
                    model_index[model] = []
                model_index[model].append(i)

        self._model_index[cache_key] = model_index

    def _load_all_experiments_battles(self, subset: str) -> list[BattleRecord]:
        """
        Load battle records from all experiments for a subset.

        Args:
            subset: Subset name

        Returns:
            Combined list of BattleRecord objects from all experiments
        """
        info = self.get_subset_info(subset)
        if not info:
            return []

        all_records: list[BattleRecord] = []
        for exp_name in info.experiments:
            records = self._load_battle_logs(subset, exp_name)
            all_records.extend(records)

        # Sort by sample_index for consistent ordering
        all_records.sort(key=lambda r: (r.sample_index, r.exp_name, r.model_a, r.model_b))
        return all_records

    def _load_audit_log(
        self, subset: str, exp_name: str, model_a: str, model_b: str, sample_index: int
    ) -> Optional[dict[str, Any]]:
        """
        Load audit log for a specific battle.

        Args:
            subset: Subset name
            exp_name: Experiment name
            model_a: First model name
            model_b: Second model name
            sample_index: Sample index

        Returns:
            Audit data dict or None
        """
        cache_key = (subset, exp_name, model_a, model_b, sample_index)
        if cache_key in self._audit_cache:
            return self._audit_cache[cache_key]

        # Determine filename (models are sorted alphabetically)
        from genarena.utils import sanitize_name

        first, second = sorted([model_a, model_b])
        filename = f"{sanitize_name(first)}_vs_{sanitize_name(second)}.jsonl"
        filepath = os.path.join(
            self.arena_dir, subset, "pk_logs", exp_name, "raw_outputs", filename
        )

        if not os.path.isfile(filepath):
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("sample_index") == sample_index:
                            self._audit_cache[cache_key] = data
                            return data
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return None

    def get_battles(
        self,
        subset: str,
        exp_name: str,
        page: int = 1,
        page_size: int = 20,
        models: Optional[list[str]] = None,
        result_filter: Optional[str] = None,  # "wins", "losses", "ties"
        consistency_filter: Optional[bool] = None,
        min_images: Optional[int] = None,
        max_images: Optional[int] = None,
        prompt_source: Optional[str] = None,
    ) -> tuple[list[BattleRecord], int]:
        """
        Get paginated battle records with filtering.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            page: Page number (1-indexed)
            page_size: Number of records per page
            models: Filter by models (show battles involving ANY of these models)
            result_filter: Filter by result relative to models ("wins", "losses", "ties")
            consistency_filter: Filter by consistency (True/False/None for all)

        Returns:
            Tuple of (records, total_count)
        """
        # Handle "__all__" experiment - combine all experiments
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
            # For __all__, we don't use the model index optimization
            cache_key = None
        else:
            all_records = self._load_battle_logs(subset, exp_name)
            cache_key = (subset, exp_name)

        # Apply filters using index for better performance
        if models and cache_key and cache_key in self._model_index:
            model_set = set(models)
            model_index = self._model_index[cache_key]

            if len(models) == 1:
                # Single model: show battles involving this model
                candidate_indices = set(model_index.get(models[0], []))
                filtered = [all_records[i] for i in sorted(candidate_indices)]
            else:
                # 2+ models: show only battles BETWEEN these models (both participants must be in selected models)
                # Find union of all records involving any selected model first
                candidate_indices: set[int] = set()
                for model in models:
                    if model in model_index:
                        candidate_indices.update(model_index[model])
                # Then filter to keep only battles where BOTH models are in the selected set
                filtered = [
                    all_records[i] for i in sorted(candidate_indices)
                    if all_records[i].model_a in model_set and all_records[i].model_b in model_set
                ]

            # Apply result filter
            if result_filter:
                if len(models) == 1:
                    # Single model: filter by that model's wins/losses/ties
                    model = models[0]
                    if result_filter == "wins":
                        filtered = [r for r in filtered if r.final_winner == model]
                    elif result_filter == "losses":
                        filtered = [
                            r
                            for r in filtered
                            if r.final_winner != "tie" and r.final_winner != model
                        ]
                    elif result_filter == "ties":
                        filtered = [r for r in filtered if r.final_winner == "tie"]
                elif len(models) == 2:
                    # Two models: filter by winner (result_filter is the winning model name or "tie")
                    if result_filter == "ties":
                        filtered = [r for r in filtered if r.final_winner == "tie"]
                    elif result_filter in models:
                        # Filter by specific model winning
                        filtered = [r for r in filtered if r.final_winner == result_filter]
        elif models:
            # Fallback for __all__ mode or when index is not available
            model_set = set(models)
            if len(models) == 1:
                model = models[0]
                filtered = [r for r in all_records if model in r.models]
                # Apply result filter
                if result_filter:
                    if result_filter == "wins":
                        filtered = [r for r in filtered if r.final_winner == model]
                    elif result_filter == "losses":
                        filtered = [
                            r
                            for r in filtered
                            if r.final_winner != "tie" and r.final_winner != model
                        ]
                    elif result_filter == "ties":
                        filtered = [r for r in filtered if r.final_winner == "tie"]
            else:
                # 2+ models: show battles between these models
                filtered = [
                    r for r in all_records
                    if r.model_a in model_set and r.model_b in model_set
                ]
                # Apply result filter
                if result_filter:
                    if result_filter == "ties":
                        filtered = [r for r in filtered if r.final_winner == "tie"]
                    elif result_filter in models:
                        filtered = [r for r in filtered if r.final_winner == result_filter]
        else:
            filtered = all_records

        # Apply consistency filter
        if consistency_filter is not None:
            filtered = [r for r in filtered if r.is_consistent == consistency_filter]

        # Apply input image count filter
        if min_images is not None or max_images is not None:
            min_img = min_images if min_images is not None else 0
            max_img = max_images if max_images is not None else float('inf')
            filtered = [r for r in filtered if min_img <= r.input_image_count <= max_img]

        # Apply prompt_source filter
        if prompt_source:
            filtered = [r for r in filtered if r.prompt_source == prompt_source]

        total_count = len(filtered)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_records = filtered[start:end]

        return page_records, total_count

    def search_battles(
        self,
        subset: str,
        exp_name: str,
        query: str,
        page: int = 1,
        page_size: int = 20,
        models: Optional[list[str]] = None,
        consistency_filter: Optional[bool] = None,
        search_fields: Optional[list[str]] = None,
    ) -> tuple[list[BattleRecord], int]:
        """
        Search battle records by text query (full-text search).

        Searches across instruction, task_type, prompt_source, and original_metadata.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            query: Search query string (case-insensitive)
            page: Page number (1-indexed)
            page_size: Number of records per page
            models: Optional filter by models
            consistency_filter: Optional filter by consistency
            search_fields: Fields to search in (default: all searchable fields)

        Returns:
            Tuple of (matching_records, total_count)
        """
        if not query or not query.strip():
            # Empty query - return regular filtered results
            return self.get_battles(
                subset, exp_name, page, page_size,
                models=models, consistency_filter=consistency_filter
            )

        # Normalize query for case-insensitive search
        query_lower = query.lower().strip()
        # Create regex pattern for more flexible matching
        query_pattern = re.compile(re.escape(query_lower), re.IGNORECASE)

        # Determine which fields to search
        all_searchable_fields = ["instruction", "task_type", "prompt_source", "original_metadata"]
        fields_to_search = search_fields if search_fields else all_searchable_fields

        # Load all records
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Apply model filter first (for efficiency)
        if models:
            model_set = set(models)
            if len(models) == 1:
                all_records = [r for r in all_records if models[0] in r.models]
            else:
                all_records = [
                    r for r in all_records
                    if r.model_a in model_set and r.model_b in model_set
                ]

        # Apply consistency filter
        if consistency_filter is not None:
            all_records = [r for r in all_records if r.is_consistent == consistency_filter]

        # Search filter
        def matches_query(record: BattleRecord) -> bool:
            """Check if record matches the search query."""
            for field_name in fields_to_search:
                value = getattr(record, field_name, None)
                if value is None:
                    continue

                # Handle different field types
                if field_name == "original_metadata" and isinstance(value, dict):
                    # Search in JSON string representation of metadata
                    metadata_str = json.dumps(value, ensure_ascii=False).lower()
                    if query_pattern.search(metadata_str):
                        return True
                elif isinstance(value, str):
                    if query_pattern.search(value):
                        return True

            return False

        # Apply search filter
        filtered = [r for r in all_records if matches_query(r)]

        total_count = len(filtered)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_records = filtered[start:end]

        return page_records, total_count

    def search_prompts(
        self,
        subset: str,
        exp_name: str,
        query: str,
        page: int = 1,
        page_size: int = 10,
        filter_models: Optional[list[str]] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Search prompts/samples by text query.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            query: Search query string
            page: Page number
            page_size: Records per page
            filter_models: Optional filter by models

        Returns:
            Tuple of (matching_prompts, total_count)
        """
        if not query or not query.strip():
            # Empty query - return regular results
            return self.get_prompts(subset, exp_name, page, page_size, filter_models=filter_models)

        # Normalize query
        query_lower = query.lower().strip()
        query_pattern = re.compile(re.escape(query_lower), re.IGNORECASE)

        # Load records and group by sample
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Group by sample_index
        sample_records: dict[int, list[BattleRecord]] = {}
        for record in all_records:
            if record.sample_index not in sample_records:
                sample_records[record.sample_index] = []
            sample_records[record.sample_index].append(record)

        # Filter samples by query
        matching_samples = []
        for sample_index, records in sample_records.items():
            if not records:
                continue
            
            first_record = records[0]
            
            # Search in instruction, task_type, prompt_source, original_metadata
            match_found = False
            
            if first_record.instruction and query_pattern.search(first_record.instruction):
                match_found = True
            elif first_record.task_type and query_pattern.search(first_record.task_type):
                match_found = True
            elif first_record.prompt_source and query_pattern.search(first_record.prompt_source):
                match_found = True
            elif first_record.original_metadata:
                metadata_str = json.dumps(first_record.original_metadata, ensure_ascii=False).lower()
                if query_pattern.search(metadata_str):
                    match_found = True

            if match_found:
                matching_samples.append(sample_index)

        # Sort and paginate
        matching_samples.sort()
        total_count = len(matching_samples)

        start = (page - 1) * page_size
        end = start + page_size
        page_samples = matching_samples[start:end]

        # Build result for each sample using get_sample_all_models
        results = []
        for sample_index in page_samples:
            prompt_data = self.get_sample_all_models(subset, exp_name, sample_index, filter_models)
            results.append(prompt_data)

        return results, total_count

    def get_battle_detail(
        self, subset: str, exp_name: str, model_a: str, model_b: str, sample_index: int
    ) -> Optional[BattleRecord]:
        """
        Get detailed battle record including VLM outputs.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            model_a: First model name
            model_b: Second model name
            sample_index: Sample index

        Returns:
            BattleRecord with audit data, or None
        """
        # Find the battle record
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        record = None
        for r in all_records:
            if (
                r.sample_index == sample_index
                and set([r.model_a, r.model_b]) == set([model_a, model_b])
            ):
                record = r
                break

        if not record:
            return None

        # Load audit data (use the record's actual exp_name for audit log lookup)
        actual_exp_name = record.exp_name
        audit = self._load_audit_log(
            subset, actual_exp_name, record.model_a, record.model_b, sample_index
        )
        if audit:
            record.original_call = audit.get("original_call")
            record.swapped_call = audit.get("swapped_call")

        return record

    def get_image_path(
        self, subset: str, model: str, sample_index: int
    ) -> Optional[str]:
        """
        Get path to model output image.

        Args:
            subset: Subset name
            model: Model name
            sample_index: Sample index

        Returns:
            Image file path or None
        """
        model_manager = self._get_model_manager(subset)
        if model_manager:
            return model_manager.get_output_path(model, sample_index)
        return None

    def get_input_image(self, subset: str, sample_index: int) -> Optional[bytes]:
        """
        Get input image bytes for a sample.

        Uses pyarrow to read directly from parquet for better performance.
        Uses cached file mapping for fast lookup.

        Args:
            subset: Subset name
            sample_index: Sample index

        Returns:
            Image bytes or None
        """
        import pyarrow.parquet as pq

        # Use cached file mapping if available (fast path)
        cache_key = (subset, sample_index)
        if cache_key in self._sample_file_map:
            pf = self._sample_file_map[cache_key]
            result = self._read_image_from_parquet(pf, sample_index)
            if result is not None:
                return result

        # Fallback: search all parquet files (slow path)
        subset_path = os.path.join(self.data_dir, subset)
        if not os.path.isdir(subset_path):
            return None

        parquet_files = sorted([
            os.path.join(subset_path, f)
            for f in os.listdir(subset_path)
            if f.startswith("data-") and f.endswith(".parquet")
        ])

        for pf in parquet_files:
            result = self._read_image_from_parquet(pf, sample_index)
            if result is not None:
                return result

        return None

    def _read_image_from_parquet(self, parquet_file: str, sample_index: int) -> Optional[bytes]:
        """Read a single image from a parquet file."""
        import pyarrow.parquet as pq

        try:
            table = pq.read_table(parquet_file, columns=["index", "input_images"])
            indices = table.column("index").to_pylist()

            if sample_index not in indices:
                return None

            row_idx = indices.index(sample_index)
            input_images = table.column("input_images")[row_idx].as_py()

            if not input_images or len(input_images) == 0:
                return None

            img_data = input_images[0]

            # Handle different formats
            if isinstance(img_data, bytes):
                return img_data
            elif isinstance(img_data, dict):
                # HuggingFace Image format: {"bytes": ..., "path": ...}
                if "bytes" in img_data and img_data["bytes"]:
                    return img_data["bytes"]
                elif "path" in img_data and img_data["path"]:
                    path = img_data["path"]
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            return f.read()

        except Exception as e:
            logger.debug(f"Error reading image from {parquet_file}: {e}")

        return None

    def get_input_image_count(self, subset: str, sample_index: int) -> int:
        """Get the number of input images for a sample."""
        import pyarrow.parquet as pq

        cache_key = (subset, sample_index)
        if cache_key in self._sample_file_map:
            pf = self._sample_file_map[cache_key]
            try:
                table = pq.read_table(pf, columns=["index", "input_images"])
                indices = table.column("index").to_pylist()
                if sample_index in indices:
                    row_idx = indices.index(sample_index)
                    input_images = table.column("input_images")[row_idx].as_py()
                    return len(input_images) if input_images else 0
            except Exception:
                pass
        return 1  # Default to 1

    def get_input_image_by_idx(self, subset: str, sample_index: int, img_idx: int = 0) -> Optional[bytes]:
        """Get a specific input image by index."""
        import pyarrow.parquet as pq

        cache_key = (subset, sample_index)
        if cache_key not in self._sample_file_map:
            return None

        pf = self._sample_file_map[cache_key]
        try:
            table = pq.read_table(pf, columns=["index", "input_images"])
            indices = table.column("index").to_pylist()

            if sample_index not in indices:
                return None

            row_idx = indices.index(sample_index)
            input_images = table.column("input_images")[row_idx].as_py()

            if not input_images or img_idx >= len(input_images):
                return None

            img_data = input_images[img_idx]

            if isinstance(img_data, bytes):
                return img_data
            elif isinstance(img_data, dict):
                if "bytes" in img_data and img_data["bytes"]:
                    return img_data["bytes"]
                elif "path" in img_data and img_data["path"]:
                    path = img_data["path"]
                    if os.path.isfile(path):
                        with open(path, "rb") as f:
                            return f.read()
        except Exception as e:
            logger.debug(f"Error reading image: {e}")

        return None

    def get_head_to_head(
        self, subset: str, exp_name: str, model_a: str, model_b: str
    ) -> dict[str, Any]:
        """
        Get head-to-head statistics between two models.

        Returns:
            Dict with wins_a, wins_b, ties, total, win_rate_a, win_rate_b
        """
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
            # For __all__, we need to filter manually
            h2h_records = [
                r for r in all_records
                if set([r.model_a, r.model_b]) == set([model_a, model_b])
            ]
        else:
            all_records = self._load_battle_logs(subset, exp_name)
            cache_key = (subset, exp_name)
            model_index = self._model_index.get(cache_key, {})

            # Find battles between these two models
            indices_a = set(model_index.get(model_a, []))
            indices_b = set(model_index.get(model_b, []))
            h2h_indices = indices_a & indices_b
            h2h_records = [all_records[idx] for idx in h2h_indices]

        wins_a = 0
        wins_b = 0
        ties = 0

        for record in h2h_records:
            if record.final_winner == model_a:
                wins_a += 1
            elif record.final_winner == model_b:
                wins_b += 1
            else:
                ties += 1

        total = wins_a + wins_b + ties

        return {
            "model_a": model_a,
            "model_b": model_b,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "total": total,
            "win_rate_a": wins_a / total if total > 0 else 0,
            "win_rate_b": wins_b / total if total > 0 else 0,
            "tie_rate": ties / total if total > 0 else 0,
        }

    def get_win_rate_matrix(
        self,
        subset: str,
        exp_name: str = "__all__",
        filter_models: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Compute win rate matrix for all model pairs.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            filter_models: Optional list of models to include

        Returns:
            Dict with:
            - models: List of model names (sorted by ELO)
            - matrix: 2D array where matrix[i][j] = win rate of model i vs model j
            - counts: 2D array where counts[i][j] = number of battles between i and j
            - wins: 2D array where wins[i][j] = wins of model i vs model j
        """
        # Load all records
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Determine models to include
        info = self.get_subset_info(subset)
        if filter_models:
            models = [m for m in filter_models if m in info.models]
        else:
            models = list(info.models)

        # Get ELO leaderboard to sort models by ELO
        leaderboard = self.get_elo_leaderboard(subset, models)
        models = [entry["model"] for entry in leaderboard]

        n = len(models)
        model_to_idx = {m: i for i, m in enumerate(models)}

        # Initialize matrices
        wins_matrix = [[0] * n for _ in range(n)]
        counts_matrix = [[0] * n for _ in range(n)]

        # Count wins for each pair
        model_set = set(models)
        for record in all_records:
            if record.model_a not in model_set or record.model_b not in model_set:
                continue

            i = model_to_idx[record.model_a]
            j = model_to_idx[record.model_b]

            # Count total battles (symmetric)
            counts_matrix[i][j] += 1
            counts_matrix[j][i] += 1

            # Count wins
            if record.final_winner == record.model_a:
                wins_matrix[i][j] += 1
            elif record.final_winner == record.model_b:
                wins_matrix[j][i] += 1
            else:
                # Tie counts as 0.5 win for each
                wins_matrix[i][j] += 0.5
                wins_matrix[j][i] += 0.5

        # Compute win rate matrix
        win_rate_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if counts_matrix[i][j] > 0:
                    win_rate_matrix[i][j] = wins_matrix[i][j] / counts_matrix[i][j]
                elif i == j:
                    win_rate_matrix[i][j] = 0.5  # Self vs self

        return {
            "models": models,
            "matrix": win_rate_matrix,
            "counts": counts_matrix,
            "wins": wins_matrix,
        }

    def get_elo_by_source(
        self,
        subset: str,
        exp_name: str = "__all__",
    ) -> dict[str, Any]:
        """
        Compute ELO rankings grouped by prompt_source.

        Args:
            subset: Subset name
            exp_name: Experiment name

        Returns:
            Dict with:
            - sources: List of source names
            - leaderboards: Dict mapping source -> list of model ELO entries
            - sample_counts: Dict mapping source -> number of samples
            - battle_counts: Dict mapping source -> number of battles
        """
        from genarena.bt_elo import compute_bt_elo_ratings

        # Load all records
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Group battles by prompt_source
        battles_by_source: dict[str, list[tuple[str, str, str]]] = {}
        sample_counts: dict[str, set[int]] = {}

        for record in all_records:
            source = record.prompt_source or "unknown"
            if source not in battles_by_source:
                battles_by_source[source] = []
                sample_counts[source] = set()

            # Convert winner to bt_elo format
            if record.final_winner == record.model_a:
                winner = "model_a"
            elif record.final_winner == record.model_b:
                winner = "model_b"
            else:
                winner = "tie"

            battles_by_source[source].append((record.model_a, record.model_b, winner))
            sample_counts[source].add(record.sample_index)

        # Compute ELO for each source
        leaderboards: dict[str, list[dict[str, Any]]] = {}
        battle_counts: dict[str, int] = {}

        for source, battles in battles_by_source.items():
            if not battles:
                continue

            battle_counts[source] = len(battles)

            try:
                ratings = compute_bt_elo_ratings(battles)

                # Build leaderboard
                entries = []
                for model, elo in ratings.items():
                    # Count wins/losses/ties for this model in this source
                    wins = losses = ties = 0
                    for ma, mb, w in battles:
                        if model == ma:
                            if w == "model_a":
                                wins += 1
                            elif w == "model_b":
                                losses += 1
                            else:
                                ties += 1
                        elif model == mb:
                            if w == "model_b":
                                wins += 1
                            elif w == "model_a":
                                losses += 1
                            else:
                                ties += 1

                    total = wins + losses + ties
                    entries.append({
                        "model": model,
                        "elo": round(elo, 1),
                        "wins": wins,
                        "losses": losses,
                        "ties": ties,
                        "total": total,
                        "win_rate": (wins + 0.5 * ties) / total if total > 0 else 0,
                    })

                # Sort by ELO descending
                entries.sort(key=lambda x: -x["elo"])
                leaderboards[source] = entries

            except Exception as e:
                logger.warning(f"Failed to compute ELO for source {source}: {e}")
                continue

        # Sort sources by battle count
        sources = sorted(battle_counts.keys(), key=lambda s: -battle_counts[s])

        return {
            "sources": sources,
            "leaderboards": leaderboards,
            "sample_counts": {s: len(sample_counts[s]) for s in sources},
            "battle_counts": battle_counts,
        }

    def _load_elo_snapshot(self, snapshot_path: str) -> Optional[dict[str, Any]]:
        """
        Load ELO snapshot from a JSON file.

        Args:
            snapshot_path: Path to elo_snapshot.json

        Returns:
            Dict with elo ratings and metadata, or None if not found
        """
        if not os.path.isfile(snapshot_path):
            return None

        try:
            with open(snapshot_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                return None

            # Extract ELO ratings (support both {"elo": {...}} and direct {model: elo} format)
            elo_data = data.get("elo") if isinstance(data.get("elo"), dict) else data
            if not isinstance(elo_data, dict):
                return None

            return {
                "elo": {str(k): float(v) for k, v in elo_data.items()},
                "battle_count": data.get("battle_count", 0),
                "model_count": data.get("model_count", len(elo_data)),
                "exp_name": data.get("exp_name", ""),
            }
        except Exception as e:
            logger.debug(f"Failed to load ELO snapshot from {snapshot_path}: {e}")
            return None

    def get_elo_history(
        self,
        subset: str,
        exp_name: str = "__all__",
        granularity: str = "experiment",
        filter_models: Optional[list[str]] = None,
        max_points: int = 50,
    ) -> dict[str, Any]:
        """
        Get ELO history over experiments by reading pre-computed elo_snapshot.json files.

        Args:
            subset: Subset name
            exp_name: Experiment name (only "__all__" or "experiment" granularity supported)
            granularity: Grouping method ("experiment" reads from snapshots; time-based not supported)
            filter_models: Optional models to track
            max_points: Maximum number of time points to return

        Returns:
            Dict with:
            - timestamps: List of experiment names
            - models: Dict mapping model -> list of ELO values
            - battle_counts: List of cumulative battle counts
        """
        # Get subset info for experiment order
        info = self.get_subset_info(subset)
        if not info:
            return {"timestamps": [], "models": {}, "battle_counts": []}

        # Only support experiment-level granularity (reading from snapshots)
        # Time-based granularity would require real-time computation which we want to avoid
        if granularity != "experiment":
            logger.warning(
                f"Time-based granularity '{granularity}' is not supported for ELO history. "
                f"Falling back to 'experiment' granularity."
            )

        # Get ordered list of experiments
        experiments = info.experiments
        if not experiments:
            return {"timestamps": [], "models": {}, "battle_counts": []}

        # If too many experiments, sample them
        if len(experiments) > max_points:
            step = len(experiments) // max_points
            sampled = [experiments[i] for i in range(0, len(experiments), step)]
            if sampled[-1] != experiments[-1]:
                sampled.append(experiments[-1])
            experiments = sampled

        # Load ELO snapshots for each experiment
        timestamps: list[str] = []
        model_elos: dict[str, list[Optional[float]]] = {}
        battle_counts: list[int] = []

        pk_logs_dir = os.path.join(self.arena_dir, subset, "pk_logs")

        for exp in experiments:
            snapshot_path = os.path.join(pk_logs_dir, exp, "elo_snapshot.json")
            snapshot = self._load_elo_snapshot(snapshot_path)

            if snapshot is None:
                # Skip experiments without snapshots
                continue

            elo_ratings = snapshot["elo"]
            battle_count = snapshot["battle_count"]

            timestamps.append(exp)
            battle_counts.append(battle_count)

            # Update model ELOs
            all_models_so_far = set(model_elos.keys()) | set(elo_ratings.keys())
            for model in all_models_so_far:
                if model not in model_elos:
                    # New model: fill with None for previous timestamps
                    model_elos[model] = [None] * (len(timestamps) - 1)
                model_elos[model].append(elo_ratings.get(model))

            # Ensure all models have the same length
            for model in model_elos:
                if len(model_elos[model]) < len(timestamps):
                    model_elos[model].append(None)

        # Filter to requested models if specified
        if filter_models:
            filter_set = set(filter_models)
            model_elos = {m: v for m, v in model_elos.items() if m in filter_set}

        return {
            "timestamps": timestamps,
            "models": model_elos,
            "battle_counts": battle_counts,
        }

    def get_cross_subset_info(
        self,
        subsets: list[str],
    ) -> dict[str, Any]:
        """
        Get information about models across multiple subsets.

        Args:
            subsets: List of subset names

        Returns:
            Dict with:
            - common_models: Models present in all subsets
            - all_models: Models present in any subset
            - per_subset_models: Dict mapping subset -> list of models
            - per_subset_battles: Dict mapping subset -> battle count
        """
        per_subset_models: dict[str, set[str]] = {}
        per_subset_battles: dict[str, int] = {}

        for subset in subsets:
            info = self.get_subset_info(subset)
            if info:
                per_subset_models[subset] = set(info.models)
                per_subset_battles[subset] = info.total_battles

        if not per_subset_models:
            return {
                "common_models": [],
                "all_models": [],
                "per_subset_models": {},
                "per_subset_battles": {},
            }

        # Compute intersection and union
        all_model_sets = list(per_subset_models.values())
        common_models = set.intersection(*all_model_sets) if all_model_sets else set()
        all_models = set.union(*all_model_sets) if all_model_sets else set()

        return {
            "common_models": sorted(common_models),
            "all_models": sorted(all_models),
            "per_subset_models": {s: sorted(m) for s, m in per_subset_models.items()},
            "per_subset_battles": per_subset_battles,
            "total_battles": sum(per_subset_battles.values()),
        }

    def get_cross_subset_elo(
        self,
        subsets: list[str],
        exp_name: str = "__all__",
        model_scope: str = "all",
    ) -> dict[str, Any]:
        """
        Compute ELO rankings across multiple subsets.

        Args:
            subsets: List of subset names
            exp_name: Experiment name (use "__all__" for all)
            model_scope: "common" = only models in all subsets, "all" = all models

        Returns:
            Dict with merged leaderboard and per-subset comparison
        """
        # Check cache first
        cache_key = (tuple(sorted(subsets)), exp_name, model_scope)
        if cache_key in self._cross_subset_elo_cache:
            return self._cross_subset_elo_cache[cache_key]

        from genarena.bt_elo import compute_bt_elo_ratings

        # Get cross-subset info
        cross_info = self.get_cross_subset_info(subsets)

        # Determine models to include
        if model_scope == "common":
            included_models = set(cross_info["common_models"])
        else:
            included_models = set(cross_info["all_models"])

        if not included_models:
            return {
                "subsets": subsets,
                "model_scope": model_scope,
                "common_models": cross_info["common_models"],
                "all_models": cross_info["all_models"],
                "total_battles": 0,
                "leaderboard": [],
                "per_subset_elo": {},
            }

        # Collect all battles
        all_battles = []
        model_presence: dict[str, set[str]] = {}  # model -> set of subsets it's in

        for subset in subsets:
            if exp_name == "__all__":
                records = self._load_all_experiments_battles(subset)
            else:
                records = self._load_battle_logs(subset, exp_name)

            for record in records:
                # Skip if either model is not in included set
                if model_scope == "common":
                    if record.model_a not in included_models or record.model_b not in included_models:
                        continue

                # Convert to bt_elo format
                if record.final_winner == record.model_a:
                    winner = "model_a"
                elif record.final_winner == record.model_b:
                    winner = "model_b"
                else:
                    winner = "tie"

                all_battles.append((record.model_a, record.model_b, winner))

                # Track model presence
                for m in [record.model_a, record.model_b]:
                    if m not in model_presence:
                        model_presence[m] = set()
                    model_presence[m].add(subset)

        if not all_battles:
            return {
                "subsets": subsets,
                "model_scope": model_scope,
                "common_models": cross_info["common_models"],
                "all_models": cross_info["all_models"],
                "total_battles": 0,
                "leaderboard": [],
                "per_subset_elo": {},
            }

        # Compute merged ELO
        try:
            ratings = compute_bt_elo_ratings(all_battles)
        except Exception as e:
            logger.error(f"Failed to compute cross-subset ELO: {e}")
            return {
                "subsets": subsets,
                "model_scope": model_scope,
                "error": str(e),
                "total_battles": len(all_battles),
                "leaderboard": [],
            }

        # Count wins/losses/ties per model
        model_stats: dict[str, dict[str, int]] = {}
        for ma, mb, winner in all_battles:
            for m in [ma, mb]:
                if m not in model_stats:
                    model_stats[m] = {"wins": 0, "losses": 0, "ties": 0}

            if winner == "model_a":
                model_stats[ma]["wins"] += 1
                model_stats[mb]["losses"] += 1
            elif winner == "model_b":
                model_stats[mb]["wins"] += 1
                model_stats[ma]["losses"] += 1
            else:
                model_stats[ma]["ties"] += 1
                model_stats[mb]["ties"] += 1

        # Build leaderboard
        leaderboard = []
        for model, elo in ratings.items():
            stats = model_stats.get(model, {"wins": 0, "losses": 0, "ties": 0})
            total = stats["wins"] + stats["losses"] + stats["ties"]
            leaderboard.append({
                "model": model,
                "elo": round(elo, 1),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total": total,
                "win_rate": (stats["wins"] + 0.5 * stats["ties"]) / total if total > 0 else 0,
                "subset_presence": sorted(model_presence.get(model, set())),
            })

        leaderboard.sort(key=lambda x: -x["elo"])

        # Get per-subset ELO for comparison
        per_subset_elo: dict[str, dict[str, float]] = {}
        for subset in subsets:
            subset_lb = self.get_elo_leaderboard(subset)
            per_subset_elo[subset] = {entry["model"]: entry["elo"] for entry in subset_lb}

        result = {
            "subsets": subsets,
            "model_scope": model_scope,
            "common_models": cross_info["common_models"],
            "all_models": cross_info["all_models"],
            "total_battles": len(all_battles),
            "leaderboard": leaderboard,
            "per_subset_elo": per_subset_elo,
        }

        # Cache the result
        self._cross_subset_elo_cache[cache_key] = result
        return result

    def get_stats(self, subset: str, exp_name: Optional[str] = None) -> dict[str, Any]:
        """
        Get statistics for a subset.

        Args:
            subset: Subset name
            exp_name: Optional experiment name (if None, uses overall state; "__all__" for all experiments)

        Returns:
            Statistics dictionary
        """
        info = self.get_subset_info(subset)
        if not info:
            return {}

        if exp_name == "__all__":
            # Combine stats from all experiments
            records = self._load_all_experiments_battles(subset)
            total_battles = len(records)
            consistent = sum(1 for r in records if r.is_consistent)
            ties = sum(1 for r in records if r.final_winner == "tie")
        elif exp_name:
            records = self._load_battle_logs(subset, exp_name)
            total_battles = len(records)
            consistent = sum(1 for r in records if r.is_consistent)
            ties = sum(1 for r in records if r.final_winner == "tie")
        else:
            total_battles = info.total_battles
            consistent = 0
            ties = 0

        return {
            "subset": subset,
            "models": info.models,
            "experiments": info.experiments,
            "total_battles": total_battles,
            "consistent_battles": consistent,
            "tie_battles": ties,
            "consistency_rate": consistent / total_battles if total_battles > 0 else 0,
        }

    def get_model_win_stats(
        self, subset: str, exp_name: str, sample_index: int,
        filter_models: Optional[list[str]] = None
    ) -> dict[str, dict[str, Any]]:
        """
        Get win/loss statistics for all models on a specific sample.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            sample_index: Sample index
            filter_models: Optional list of models to filter (only count battles between these models)

        Returns:
            Dict mapping model name to stats (wins, losses, ties, total, win_rate)
        """
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Filter records for this sample
        sample_records = [r for r in all_records if r.sample_index == sample_index]

        # If filter_models is specified, only count battles between those models
        if filter_models:
            filter_set = set(filter_models)
            sample_records = [
                r for r in sample_records
                if r.model_a in filter_set and r.model_b in filter_set
            ]

        # Collect stats per model
        model_stats: dict[str, dict[str, int]] = {}

        for record in sample_records:
            for model in [record.model_a, record.model_b]:
                if model not in model_stats:
                    model_stats[model] = {"wins": 0, "losses": 0, "ties": 0}

            if record.final_winner == "tie":
                model_stats[record.model_a]["ties"] += 1
                model_stats[record.model_b]["ties"] += 1
            elif record.final_winner == record.model_a:
                model_stats[record.model_a]["wins"] += 1
                model_stats[record.model_b]["losses"] += 1
            elif record.final_winner == record.model_b:
                model_stats[record.model_b]["wins"] += 1
                model_stats[record.model_a]["losses"] += 1

        # Calculate win rate and total
        result: dict[str, dict[str, Any]] = {}
        for model, stats in model_stats.items():
            total = stats["wins"] + stats["losses"] + stats["ties"]
            win_rate = stats["wins"] / total if total > 0 else 0
            result[model] = {
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total": total,
                "win_rate": win_rate,
            }

        return result

    def get_sample_all_models(
        self, subset: str, exp_name: str, sample_index: int,
        filter_models: Optional[list[str]] = None,
        stats_scope: str = "filtered"
    ) -> dict[str, Any]:
        """
        Get all model outputs for a specific sample, sorted by win rate.

        Args:
            subset: Subset name
            exp_name: Experiment name
            sample_index: Sample index
            filter_models: Optional list of models to filter (show only these models)
            stats_scope: 'filtered' = only count battles between filtered models,
                        'all' = count all battles (but show only filtered models)

        Returns:
            Dict with sample info and all model outputs sorted by win rate
        """
        # Get sample metadata
        sample_meta = self._get_sample_data(subset, sample_index)

        # Determine which models to use for stats calculation
        # If stats_scope is 'all', don't filter battles by models
        stats_filter = filter_models if stats_scope == "filtered" else None
        model_stats = self.get_model_win_stats(subset, exp_name, sample_index, stats_filter)

        # Get all models that have outputs
        model_manager = self._get_model_manager(subset)
        available_models = []

        if model_manager:
            # Determine which models to include
            models_to_check = model_manager.models
            if filter_models:
                filter_set = set(filter_models)
                models_to_check = [m for m in models_to_check if m in filter_set]

            for model in models_to_check:
                output_path = model_manager.get_output_path(model, sample_index)
                if output_path and os.path.isfile(output_path):
                    stats = model_stats.get(model, {
                        "wins": 0, "losses": 0, "ties": 0, "total": 0, "win_rate": 0
                    })
                    available_models.append({
                        "model": model,
                        "wins": stats["wins"],
                        "losses": stats["losses"],
                        "ties": stats["ties"],
                        "total": stats["total"],
                        "win_rate": stats["win_rate"],
                    })

        # Sort by win rate (descending), then by wins (descending), then by model name
        available_models.sort(key=lambda x: (-x["win_rate"], -x["wins"], x["model"]))

        return {
            "subset": subset,
            "exp_name": exp_name,
            "sample_index": sample_index,
            "instruction": sample_meta.get("instruction", ""),
            "task_type": sample_meta.get("task_type", ""),
            "input_image_count": sample_meta.get("input_image_count", 1),
            "prompt_source": sample_meta.get("prompt_source"),
            "original_metadata": sample_meta.get("original_metadata"),
            "models": available_models,
        }

    def get_model_battles_for_sample(
        self,
        subset: str,
        exp_name: str,
        sample_index: int,
        model: str,
        opponent_models: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Get all battle records for a specific model on a specific sample.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            sample_index: Sample index
            model: The model to get battles for
            opponent_models: Optional list of opponent models to filter by

        Returns:
            Dict with model info and list of battle records
        """
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Filter records for this sample and involving this model
        model_battles = []
        all_opponents = set()

        for record in all_records:
            if record.sample_index != sample_index:
                continue
            if model not in [record.model_a, record.model_b]:
                continue

            # Determine opponent
            opponent = record.model_b if record.model_a == model else record.model_a
            all_opponents.add(opponent)

            # Apply opponent filter if specified
            if opponent_models and opponent not in opponent_models:
                continue

            # Determine result for this model
            if record.final_winner == "tie":
                result = "tie"
            elif record.final_winner == model:
                result = "win"
            else:
                result = "loss"

            # Build battle data with judge outputs
            battle_data = {
                "opponent": opponent,
                "result": result,
                "is_consistent": record.is_consistent,
                "model_a": record.model_a,
                "model_b": record.model_b,
                "final_winner": record.final_winner,
                "exp_name": record.exp_name,
            }

            # Load audit logs if not already loaded on the record
            if not record.original_call and not record.swapped_call:
                actual_exp_name = record.exp_name
                audit = self._load_audit_log(
                    subset, actual_exp_name, record.model_a, record.model_b, sample_index
                )
                if audit:
                    battle_data["original_call"] = audit.get("original_call")
                    battle_data["swapped_call"] = audit.get("swapped_call")
            else:
                # Use existing data if available
                if record.original_call:
                    battle_data["original_call"] = record.original_call
                if record.swapped_call:
                    battle_data["swapped_call"] = record.swapped_call

            model_battles.append(battle_data)

        # Sort battles by opponent name
        model_battles.sort(key=lambda x: x["opponent"])

        # Get model stats
        model_stats = self.get_model_win_stats(subset, exp_name, sample_index)
        stats = model_stats.get(model, {
            "wins": 0, "losses": 0, "ties": 0, "total": 0, "win_rate": 0
        })

        return {
            "model": model,
            "sample_index": sample_index,
            "wins": stats["wins"],
            "losses": stats["losses"],
            "ties": stats["ties"],
            "total": stats["total"],
            "win_rate": stats["win_rate"],
            "battles": model_battles,
            "all_opponents": sorted(list(all_opponents)),
        }

    def get_elo_leaderboard(
        self,
        subset: str,
        filter_models: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Get ELO leaderboard for a subset from state.json.

        Args:
            subset: Subset name
            filter_models: Optional list of models to filter (show only these models)

        Returns:
            List of model stats sorted by ELO rating (descending)
        """
        info = self.get_subset_info(subset)
        if not info or not info.state:
            return []

        state = info.state
        leaderboard = []

        for model_name, model_stats in state.models.items():
            # Apply filter if specified
            if filter_models and model_name not in filter_models:
                continue

            leaderboard.append({
                "model": model_name,
                "elo": model_stats.elo,
                "wins": model_stats.wins,
                "losses": model_stats.losses,
                "ties": model_stats.ties,
                "total_battles": model_stats.total_battles,
                "win_rate": model_stats.win_rate,
            })

        # Sort by ELO rating (descending)
        leaderboard.sort(key=lambda x: -x["elo"])

        # Add rank
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    def get_model_vs_stats(
        self,
        subset: str,
        model: str,
        exp_name: str = "__all__",
    ) -> dict[str, Any]:
        """
        Get win/loss/tie stats of a specific model against all other models.

        Args:
            subset: Subset name
            model: Target model name
            exp_name: Experiment name (default "__all__" for all experiments)

        Returns:
            Dict with model stats and versus stats against each opponent
        """
        # Get overall ELO stats
        info = self.get_subset_info(subset)
        if not info or not info.state:
            return {}

        state = info.state
        if model not in state.models:
            return {}

        model_stats = state.models[model]

        # Load battle records
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Calculate stats against each opponent
        vs_stats: dict[str, dict[str, int]] = {}

        for record in all_records:
            if model not in [record.model_a, record.model_b]:
                continue

            opponent = record.model_b if record.model_a == model else record.model_a

            if opponent not in vs_stats:
                vs_stats[opponent] = {"wins": 0, "losses": 0, "ties": 0}

            if record.final_winner == "tie":
                vs_stats[opponent]["ties"] += 1
            elif record.final_winner == model:
                vs_stats[opponent]["wins"] += 1
            else:
                vs_stats[opponent]["losses"] += 1

        # Convert to list with win rates and opponent ELO
        vs_list = []
        for opponent, stats in vs_stats.items():
            total = stats["wins"] + stats["losses"] + stats["ties"]
            opponent_elo = state.models[opponent].elo if opponent in state.models else 1000.0
            vs_list.append({
                "opponent": opponent,
                "opponent_elo": opponent_elo,
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total": total,
                "win_rate": stats["wins"] / total if total > 0 else 0,
            })

        # Sort by opponent ELO (descending)
        vs_list.sort(key=lambda x: -x["opponent_elo"])

        return {
            "model": model,
            "elo": model_stats.elo,
            "wins": model_stats.wins,
            "losses": model_stats.losses,
            "ties": model_stats.ties,
            "total_battles": model_stats.total_battles,
            "win_rate": model_stats.win_rate,
            "vs_stats": vs_list,
        }

    def get_all_subsets_leaderboards(self) -> dict[str, Any]:
        """
        Get leaderboard data for all subsets (for Overview page).

        Returns:
            Dict with:
            - subsets: List of subset names
            - models: List of all unique model names across all subsets
            - data: Dict mapping subset -> {model -> {elo, rank, wins, losses, ties, ...}}
            - subset_info: Dict mapping subset -> {total_battles, model_count}
        """
        subsets = self.discover_subsets()
        all_models: set[str] = set()
        data: dict[str, dict[str, dict[str, Any]]] = {}
        subset_info: dict[str, dict[str, Any]] = {}

        for subset in subsets:
            leaderboard = self.get_elo_leaderboard(subset)
            info = self.get_subset_info(subset)

            if not leaderboard:
                continue

            # Build subset data
            subset_data: dict[str, dict[str, Any]] = {}
            for entry in leaderboard:
                model = entry["model"]
                all_models.add(model)
                subset_data[model] = {
                    "elo": entry["elo"],
                    "rank": entry["rank"],
                    "wins": entry["wins"],
                    "losses": entry["losses"],
                    "ties": entry["ties"],
                    "total_battles": entry["total_battles"],
                    "win_rate": entry["win_rate"],
                }

            data[subset] = subset_data
            subset_info[subset] = {
                "total_battles": info.total_battles if info else 0,
                "model_count": len(leaderboard),
            }

        # Sort models by average ELO across all subsets (descending)
        model_avg_elo: dict[str, tuple[float, int]] = {}  # model -> (sum_elo, count)
        for model in all_models:
            total_elo = 0.0
            count = 0
            for subset in subsets:
                if subset in data and model in data[subset]:
                    total_elo += data[subset][model]["elo"]
                    count += 1
            if count > 0:
                model_avg_elo[model] = (total_elo / count, count)
            else:
                model_avg_elo[model] = (0.0, 0)

        sorted_models = sorted(
            all_models,
            key=lambda m: (-model_avg_elo[m][0], -model_avg_elo[m][1], m)
        )

        return {
            "subsets": subsets,
            "models": sorted_models,
            "data": data,
            "subset_info": subset_info,
        }

    def get_prompts(
        self,
        subset: str,
        exp_name: str,
        page: int = 1,
        page_size: int = 10,
        min_images: Optional[int] = None,
        max_images: Optional[int] = None,
        prompt_source: Optional[str] = None,
        filter_models: Optional[list[str]] = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        Get paginated list of prompts/samples with all model outputs.

        Args:
            subset: Subset name
            exp_name: Experiment name (use "__all__" for all experiments)
            page: Page number (1-indexed)
            page_size: Number of records per page
            min_images: Minimum number of input images
            max_images: Maximum number of input images
            prompt_source: Filter by prompt source
            filter_models: Optional list of models to filter (show only these models)

        Returns:
            Tuple of (prompts_list, total_count)
        """
        # Get all sample indices from battle logs
        if exp_name == "__all__":
            all_records = self._load_all_experiments_battles(subset)
        else:
            all_records = self._load_battle_logs(subset, exp_name)

        # Collect unique sample indices
        sample_indices = set()
        for record in all_records:
            sample_indices.add(record.sample_index)

        # Sort sample indices
        sorted_indices = sorted(sample_indices)

        # Apply filters
        filtered_indices = []
        for idx in sorted_indices:
            sample_meta = self._get_sample_data(subset, idx)
            img_count = sample_meta.get("input_image_count", 1)
            source = sample_meta.get("prompt_source")

            # Apply image count filter
            if min_images is not None and img_count < min_images:
                continue
            if max_images is not None and img_count > max_images:
                continue

            # Apply prompt source filter
            if prompt_source and source != prompt_source:
                continue

            filtered_indices.append(idx)

        total_count = len(filtered_indices)

        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        page_indices = filtered_indices[start:end]

        # Build prompt data for each sample
        prompts = []
        for idx in page_indices:
            prompt_data = self.get_sample_all_models(subset, exp_name, idx, filter_models)
            prompts.append(prompt_data)

        return prompts, total_count


class HFArenaDataLoader(ArenaDataLoader):
    """
    Data loader for HuggingFace Spaces deployment.

    Extends ArenaDataLoader to:
    - Build image URL index from HF file list
    - Return HF CDN URLs for model output images instead of local paths
    """

    def __init__(
        self,
        arena_dir: str,
        data_dir: str,
        hf_repo: str,
        image_files: list[str],
        preload: bool = True,
    ):
        """
        Initialize the HF data loader.

        Args:
            arena_dir: Path to arena directory (metadata only, no images)
            data_dir: Path to data directory containing parquet files
            hf_repo: HuggingFace repo ID for image CDN URLs
            image_files: List of image file paths in the HF repo
            preload: If True, preload all data at initialization
        """
        self.hf_repo = hf_repo
        self._image_url_index = self._build_image_index(image_files)
        super().__init__(arena_dir, data_dir, preload=preload)

    def _build_image_index(
        self, image_files: list[str]
    ) -> dict[tuple[str, str, int], str]:
        """
        Build index: (subset, model, sample_index) -> hf_file_path

        Expected path format: {subset}/models/{exp_name}/{model}/{index}.png

        Args:
            image_files: List of image file paths from HF repo

        Returns:
            Dict mapping (subset, model, sample_index) to HF file path
        """
        from genarena.models import parse_image_index

        index: dict[tuple[str, str, int], str] = {}

        for path in image_files:
            parts = path.split("/")
            # Expected: subset/models/exp_name/model/000000.png
            if len(parts) >= 5 and parts[1] == "models":
                subset = parts[0]
                # exp_name = parts[2]  # Not needed for lookup
                model = parts[3]
                filename = parts[4]
                idx = parse_image_index(filename)
                if idx is not None:
                    # If duplicate, later entries overwrite earlier ones
                    index[(subset, model, idx)] = path

        logger.info(f"Built image URL index with {len(index)} entries")
        return index

    def get_model_image_url(
        self, subset: str, model: str, sample_index: int
    ) -> Optional[str]:
        """
        Get HF CDN URL for model output image.

        Args:
            subset: Subset name
            model: Model name
            sample_index: Sample index

        Returns:
            HF CDN URL or None if not found
        """
        path = self._image_url_index.get((subset, model, sample_index))
        if path:
            return f"https://huggingface.co/datasets/{self.hf_repo}/resolve/main/{path}"
        return None

    def get_image_path(
        self, subset: str, model: str, sample_index: int
    ) -> Optional[str]:
        """
        Override to return None since images are served via CDN.

        For HF deployment, use get_model_image_url() instead.
        """
        # Return None to indicate image should be fetched via CDN
        return None

    def _get_available_models_for_subset(self, subset: str) -> list[str]:
        """
        Get list of models that have images in the HF CDN for this subset.

        Returns:
            List of model names
        """
        models = set()
        for (s, model, _) in self._image_url_index.keys():
            if s == subset:
                models.add(model)
        return sorted(models)

    def _has_model_image(self, subset: str, model: str, sample_index: int) -> bool:
        """
        Check if a model has an image for a specific sample in the HF CDN.

        Args:
            subset: Subset name
            model: Model name
            sample_index: Sample index

        Returns:
            True if image exists in CDN index
        """
        return (subset, model, sample_index) in self._image_url_index

    def get_sample_all_models(
        self, subset: str, exp_name: str, sample_index: int,
        filter_models: Optional[list[str]] = None,
        stats_scope: str = "filtered"
    ) -> dict[str, Any]:
        """
        Get all model outputs for a specific sample, sorted by win rate.

        Override for HF deployment to use CDN image index instead of local files.

        Args:
            subset: Subset name
            exp_name: Experiment name
            sample_index: Sample index
            filter_models: Optional list of models to filter (show only these models)
            stats_scope: 'filtered' = only count battles between filtered models,
                        'all' = count all battles (but show only filtered models)

        Returns:
            Dict with sample info and all model outputs sorted by win rate
        """
        # Get sample metadata
        sample_meta = self._get_sample_data(subset, sample_index)

        # Determine which models to use for stats calculation
        stats_filter = filter_models if stats_scope == "filtered" else None
        model_stats = self.get_model_win_stats(subset, exp_name, sample_index, stats_filter)

        # Get all models that have outputs in CDN
        available_models_list = self._get_available_models_for_subset(subset)

        # Apply filter if specified
        if filter_models:
            filter_set = set(filter_models)
            available_models_list = [m for m in available_models_list if m in filter_set]

        # Build model info for models that have images for this sample
        available_models = []
        for model in available_models_list:
            # Check if model has image for this sample in CDN index
            if self._has_model_image(subset, model, sample_index):
                stats = model_stats.get(model, {
                    "wins": 0, "losses": 0, "ties": 0, "total": 0, "win_rate": 0
                })
                available_models.append({
                    "model": model,
                    "wins": stats["wins"],
                    "losses": stats["losses"],
                    "ties": stats["ties"],
                    "total": stats["total"],
                    "win_rate": stats["win_rate"],
                })

        # Sort by win rate (descending), then by wins (descending), then by model name
        available_models.sort(key=lambda x: (-x["win_rate"], -x["wins"], x["model"]))

        return {
            "subset": subset,
            "exp_name": exp_name,
            "sample_index": sample_index,
            "instruction": sample_meta.get("instruction", ""),
            "task_type": sample_meta.get("task_type", ""),
            "input_image_count": sample_meta.get("input_image_count", 1),
            "prompt_source": sample_meta.get("prompt_source"),
            "original_metadata": sample_meta.get("original_metadata"),
            "models": available_models,
        }
