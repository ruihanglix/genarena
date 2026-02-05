# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Data loading module for parquet datasets."""

import io
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import pyarrow.parquet as pq
from PIL import Image as PILImage


@dataclass
class DataSample:
    """Data sample from parquet dataset."""

    index: int
    task_type: str
    instruction: str
    input_images: list[bytes]  # List of image bytes from parquet
    prompt_source: Optional[str] = None
    original_metadata: Optional[dict[str, Any]] = None


def _convert_to_bytes(img: Any) -> Optional[bytes]:
    """
    Convert various image formats to bytes.

    Handles:
    - bytes: return as-is
    - PIL.Image: convert to PNG bytes
    - dict with 'bytes' key: extract bytes
    - dict with 'image' key: recursively process
    - io.BytesIO: read bytes from buffer
    - str (file path): read file bytes
    - pyarrow struct: extract bytes from pyarrow internal format

    Args:
        img: Image in any supported format

    Returns:
        Image bytes, or None if conversion fails
    """
    if img is None:
        return None

    # Already bytes
    if isinstance(img, bytes):
        return img

    # BytesIO object
    if isinstance(img, io.BytesIO):
        img.seek(0)
        return img.read()

    # PIL Image
    if isinstance(img, PILImage.Image):
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    # Dict formats (from HuggingFace datasets Image() type)
    if isinstance(img, dict):
        # Try 'bytes' key first
        if "bytes" in img:
            raw = img["bytes"]
            if isinstance(raw, bytes):
                return raw
            elif isinstance(raw, io.BytesIO):
                raw.seek(0)
                return raw.read()
            # Recurse for nested structures
            return _convert_to_bytes(raw)

        # Try 'image' key (some formats use this)
        if "image" in img:
            return _convert_to_bytes(img["image"])

        # Try 'path' key if it's a file path
        if "path" in img and img["path"] and isinstance(img["path"], str):
            path = img["path"]
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    return f.read()

    # String (file path)
    if isinstance(img, str):
        if os.path.isfile(img):
            with open(img, "rb") as f:
                return f.read()

    # Handle pyarrow struct (when reading HuggingFace datasets parquet with pyarrow)
    # PyArrow may return a struct with 'bytes' and 'path' fields
    try:
        # Try to access as pyarrow scalar
        if hasattr(img, "as_py"):
            return _convert_to_bytes(img.as_py())

        # Try to access struct fields
        if hasattr(img, "__getitem__") and not isinstance(img, (str, bytes, dict, list, tuple)):
            # Try to get 'bytes' field
            try:
                bytes_val = img["bytes"]
                if bytes_val is not None:
                    return _convert_to_bytes(bytes_val)
            except (KeyError, TypeError, IndexError):
                pass
    except Exception:
        pass

    # Numpy array (convert to PIL first)
    try:
        import numpy as np
        if isinstance(img, np.ndarray):
            pil_img = PILImage.fromarray(img)
            buffer = io.BytesIO()
            pil_img.save(buffer, format="PNG")
            return buffer.getvalue()
    except (ImportError, Exception):
        pass

    # Last resort: try to get bytes attribute
    if hasattr(img, "tobytes"):
        try:
            return img.tobytes()
        except Exception:
            pass

    # Debug: log the type we couldn't handle
    warnings.warn(f"_convert_to_bytes: Unknown image type: {type(img)}, repr: {repr(img)[:200]}")
    return None


def discover_subsets(data_dir: str) -> list[str]:
    """
    Discover all subset directories in the data directory.

    A valid subset directory should contain at least one data-*.parquet file.

    Args:
        data_dir: Path to the parquet data directory

    Returns:
        List of subset names (directory names)
    """
    subsets = []

    if not os.path.isdir(data_dir):
        warnings.warn(f"Data directory does not exist: {data_dir}")
        return subsets

    for name in os.listdir(data_dir):
        subset_path = os.path.join(data_dir, name)
        if os.path.isdir(subset_path):
            # Check if directory contains parquet files
            parquet_files = [
                f for f in os.listdir(subset_path)
                if f.startswith("data-") and f.endswith(".parquet")
            ]
            if parquet_files:
                subsets.append(name)

    return sorted(subsets)


class ParquetDataset:
    """
    Dataset class for loading parquet formatted evaluation data.

    Expected parquet columns:
    - task_type: str
    - instruction: str
    - input_images: list of image bytes
    - index: int (optional, will use row index if not present)
    - prompt_source: str (optional)
    - original_metadata: dict (optional)
    """

    def __init__(
        self,
        data_dir: str,
        subset: str,
        parquet_files: Optional[list[str]] = None,
        load_mode: str = "full",
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Root directory containing subset directories
            subset: Name of the subset to load (e.g., 'basic', 'reasoning')
            parquet_files: Optional explicit parquet file list to load. If provided,
                only these files are loaded (useful for per-parquet multiprocessing).
            load_mode: "full" loads all columns (and may decode images via HF datasets),
                "index_only" only scans the parquet "index" column for fast sharding.
        """
        self.data_dir = data_dir
        self.subset = subset
        self.subset_path = os.path.join(data_dir, subset)
        self.load_mode = load_mode

        # Load and concatenate all parquet files in the subset
        self._data: Optional[pd.DataFrame] = None
        # Bookkeeping for sharding by parquet file
        self._parquet_files: list[str] = list(parquet_files) if parquet_files else []
        self._index_to_parquet: dict[int, str] = {}
        self._all_indices: list[int] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load all parquet files in the subset directory.

        Uses HuggingFace datasets library to properly decode Image features,
        then converts to pandas DataFrame for consistent access.
        """
        if not os.path.isdir(self.subset_path):
            warnings.warn(f"Subset directory does not exist: {self.subset_path}")
            self._data = pd.DataFrame()
            return

        # Find all data-*.parquet files unless explicitly provided
        if self._parquet_files:
            parquet_files = list(self._parquet_files)
        else:
            parquet_files = sorted([
                os.path.join(self.subset_path, f)
                for f in os.listdir(self.subset_path)
                if f.startswith("data-") and f.endswith(".parquet")
            ])
            self._parquet_files = parquet_files

        if not parquet_files:
            warnings.warn(f"No parquet files found in: {self.subset_path}")
            self._data = pd.DataFrame()
            return

        # Fast path: only scan parquet index for sharding/grouping.
        if self.load_mode == "index_only":
            index_to_parquet: dict[int, str] = {}
            all_indices: list[int] = []
            for pf in parquet_files:
                try:
                    table = pq.read_table(pf, columns=["index"])
                    col = table.column(0).to_pylist()
                    for v in col:
                        try:
                            idx_int = int(v)
                        except Exception:
                            continue
                        all_indices.append(idx_int)
                        if idx_int not in index_to_parquet:
                            index_to_parquet[idx_int] = pf
                except Exception as e:
                    warnings.warn(f"Failed to scan parquet index column {pf}: {e}")

            self._index_to_parquet = index_to_parquet
            self._all_indices = all_indices
            self._data = pd.DataFrame()
            return

        # Try to use HuggingFace datasets library first (properly decodes Image features)
        try:
            from datasets import load_dataset

            # Faster full load: let datasets load all files together.
            ds = load_dataset(
                "parquet",
                data_files={"train": parquet_files},
                split="train",
            )

            records: list[dict[str, Any]] = []
            for i in range(len(ds)):
                records.append(dict(ds[i]))

            self._data = pd.DataFrame(records)
            # Mapping is not guaranteed in this mode; multi-process sharding uses index_only mode.
            if "index" in self._data.columns:
                try:
                    self._all_indices = [int(v) for v in self._data["index"].tolist()]
                except Exception:
                    self._all_indices = []

        except ImportError:
            # Fall back to pyarrow if datasets not available
            warnings.warn(
                "HuggingFace datasets library not available, "
                "falling back to pyarrow (Image features may not decode correctly)"
            )
            dfs: list[pd.DataFrame] = []
            index_to_parquet: dict[int, str] = {}
            for pf in parquet_files:
                try:
                    df = pq.read_table(pf).to_pandas()
                    # Preserve source mapping for sharding
                    df["__source_parquet"] = pf
                    if "index" in df.columns:
                        for v in df["index"].tolist():
                            try:
                                idx_int = int(v)
                                if idx_int not in index_to_parquet:
                                    index_to_parquet[idx_int] = pf
                            except Exception:
                                continue
                    dfs.append(df)
                except Exception as e:
                    warnings.warn(f"Failed to read parquet file {pf}: {e}")

            if dfs:
                self._data = pd.concat(dfs, ignore_index=True)
            else:
                self._data = pd.DataFrame()
            self._index_to_parquet = index_to_parquet
            if "index" in self._data.columns:
                try:
                    self._all_indices = [int(v) for v in self._data["index"].tolist()]
                except Exception:
                    self._all_indices = []

    @property
    def parquet_files(self) -> list[str]:
        """Return the list of parquet files that back this dataset."""
        return list(self._parquet_files)

    def get_parquet_file_for_index(self, sample_index: int) -> Optional[str]:
        """
        Get the source parquet file for a given sample_index.

        Args:
            sample_index: The sample "index" field value.

        Returns:
            Parquet file path if known, else None.
        """
        try:
            return self._index_to_parquet.get(int(sample_index))
        except Exception:
            return None

    def group_indices_by_parquet(self, indices: Optional[list[int]] = None) -> dict[str, list[int]]:
        """
        Group indices by their source parquet file.

        Args:
            indices: Optional subset of indices to group. If None, uses all indices.

        Returns:
            Dict mapping parquet file path -> list of sample indices (in input order).
            Indices whose parquet source is unknown are grouped under key "".
        """
        if indices is None:
            indices = self.get_all_indices()

        grouped: dict[str, list[int]] = defaultdict(list)
        for idx in indices:
            pf = self.get_parquet_file_for_index(idx) or ""
            grouped[pf].append(idx)
        return dict(grouped)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self.load_mode == "index_only":
            return len(self._all_indices)
        return len(self._data) if self._data is not None else 0

    def __getitem__(self, idx: int) -> DataSample:
        """
        Get a sample by index.

        Args:
            idx: Index of the sample

        Returns:
            DataSample object
        """
        if self._data is None or idx >= len(self._data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        row = self._data.iloc[idx]

        # Extract fields with defaults
        sample_index = row.get("index", idx)
        task_type = row.get("task_type", "edit")
        instruction = row.get("instruction", "")

        # Handle input_images - could be various formats
        # PyArrow reads HuggingFace Image() as dict: {"bytes": <bytes>, "path": None}
        input_images_raw = row.get("input_images", [])
        input_images = []

        if input_images_raw is None:
            pass  # input_images stays empty
        elif isinstance(input_images_raw, (list, tuple)):
            # List of images
            for img in input_images_raw:
                img_bytes = _convert_to_bytes(img)
                if img_bytes is not None:
                    input_images.append(img_bytes)
                else:
                    # Log warning but continue
                    warnings.warn(f"Failed to convert image at index {sample_index}: {type(img)}")
        else:
            # Single image
            img_bytes = _convert_to_bytes(input_images_raw)
            if img_bytes is not None:
                input_images.append(img_bytes)

        prompt_source = row.get("prompt_source", None)
        original_metadata = row.get("original_metadata", None)

        return DataSample(
            index=int(sample_index),
            task_type=str(task_type),
            instruction=str(instruction),
            input_images=input_images,
            prompt_source=prompt_source,
            original_metadata=original_metadata,
        )

    def get_by_index(self, sample_index: int) -> Optional[DataSample]:
        """
        Get a sample by its index field (not row position).

        Args:
            sample_index: The 'index' field value to search for

        Returns:
            DataSample if found, None otherwise
        """
        if self.load_mode == "index_only":
            return None

        if self._data is None or len(self._data) == 0:
            return None

        # Check if 'index' column exists
        if "index" in self._data.columns:
            matches = self._data[self._data["index"] == sample_index]
            if not matches.empty:
                row_idx = matches.index[0]
                return self[row_idx]

        # Fall back to using position as index
        if 0 <= sample_index < len(self._data):
            return self[sample_index]

        return None

    def get_all_indices(self) -> list[int]:
        """
        Get all sample indices in the dataset.

        Returns:
            List of index values
        """
        if self.load_mode == "index_only":
            return list(self._all_indices)

        if self._data is None or len(self._data) == 0:
            return []

        if "index" in self._data.columns:
            return self._data["index"].tolist()
        else:
            return list(range(len(self._data)))

    @property
    def is_empty(self) -> bool:
        """Check if the dataset is empty."""
        return len(self) == 0
