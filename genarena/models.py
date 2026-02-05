# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Model output management module."""

from __future__ import annotations

import os
import re
import warnings
from datetime import date
from typing import Optional

from genarena.experiments import parse_exp_date_suffix


def parse_image_index(filename: str) -> Optional[int]:
    """
    Extract numeric index from image filename.

    Supports both zero-padded format (e.g., '000001.png') and simple numeric
    format (e.g., '1.png', '42.png').

    Args:
        filename: Image filename (e.g., '000001.png' or '1.png')

    Returns:
        Integer index, or None if parsing fails
    """
    # Remove extension and extract numeric part
    name = os.path.splitext(filename)[0]

    # Match pure numeric names (with or without leading zeros)
    if name.isdigit():
        return int(name)

    # Try to extract leading number
    match = re.match(r'^(\d+)', name)
    if match:
        return int(match.group(1))

    return None


def discover_models(models_dir: str) -> list[str]:
    """
    Discover all model subdirectories in the models directory.

    Args:
        models_dir: Path to the models directory

    Returns:
        List of model names (directory names)
    """
    models = []

    if not os.path.isdir(models_dir):
        warnings.warn(f"Models directory does not exist: {models_dir}")
        return models

    for name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, name)
        if os.path.isdir(model_path):
            # Check if directory contains any image files
            has_images = any(
                f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
                for f in os.listdir(model_path)
            )
            if has_images:
                models.append(name)

    return sorted(models)


class ModelOutputManager:
    """
    Manager for model output images.

    Handles discovery and retrieval of model output images,
    supporting various naming formats.
    """

    def __init__(self, models_dir: str):
        """
        Initialize the manager and scan for model outputs.

        Args:
            models_dir: Path to the models directory
        """
        self.models_dir = models_dir

        # Model name -> {index -> filepath}
        self._index_map: dict[str, dict[int, str]] = {}

        # Discover models
        self._models: list[str] = []
        self._scan_models()

    def _scan_models(self) -> None:
        """Scan models directory and build index mapping."""
        if not os.path.isdir(self.models_dir):
            warnings.warn(f"Models directory does not exist: {self.models_dir}")
            return

        for model_name in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, model_name)
            if not os.path.isdir(model_path):
                continue

            # Build index map for this model
            index_map: dict[int, str] = {}

            for filename in os.listdir(model_path):
                # Only process image files
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    continue

                idx = parse_image_index(filename)
                if idx is not None:
                    filepath = os.path.join(model_path, filename)
                    # If duplicate index, prefer .png
                    if idx in index_map:
                        if filename.lower().endswith('.png'):
                            index_map[idx] = filepath
                    else:
                        index_map[idx] = filepath

            if index_map:
                self._models.append(model_name)
                self._index_map[model_name] = index_map

        self._models.sort()

    @property
    def models(self) -> list[str]:
        """Get list of discovered model names."""
        return self._models.copy()

    def get_output_path(self, model: str, index: int) -> Optional[str]:
        """
        Get the output image path for a model at a given index.

        Args:
            model: Model name
            index: Sample index

        Returns:
            Path to the output image, or None if not found
        """
        if model not in self._index_map:
            return None

        return self._index_map[model].get(index)

    def get_model_indices(self, model: str) -> set[int]:
        """
        Get all available indices for a model.

        Args:
            model: Model name

        Returns:
            Set of available indices
        """
        if model not in self._index_map:
            return set()

        return set(self._index_map[model].keys())

    def validate_coverage(
        self,
        model_a: str,
        model_b: str,
        indices: list[int]
    ) -> list[int]:
        """
        Validate which indices have outputs from both models.

        Args:
            model_a: First model name
            model_b: Second model name
            indices: List of indices to validate

        Returns:
            List of indices where both models have outputs
        """
        indices_a = self.get_model_indices(model_a)
        indices_b = self.get_model_indices(model_b)

        valid_indices = []
        missing_a = []
        missing_b = []

        for idx in indices:
            has_a = idx in indices_a
            has_b = idx in indices_b

            if has_a and has_b:
                valid_indices.append(idx)
            else:
                if not has_a:
                    missing_a.append(idx)
                if not has_b:
                    missing_b.append(idx)

        # Log warnings for missing outputs
        if missing_a:
            warnings.warn(
                f"Model '{model_a}' missing outputs for {len(missing_a)} indices: "
                f"{missing_a[:5]}{'...' if len(missing_a) > 5 else ''}"
            )
        if missing_b:
            warnings.warn(
                f"Model '{model_b}' missing outputs for {len(missing_b)} indices: "
                f"{missing_b[:5]}{'...' if len(missing_b) > 5 else ''}"
            )

        return valid_indices

    def has_model(self, model: str) -> bool:
        """Check if a model exists in the manager."""
        return model in self._index_map

    def refresh(self) -> None:
        """Re-scan the models directory."""
        self._models = []
        self._index_map = {}
        self._scan_models()


class GlobalModelOutputManager:
    """
    Manager for model outputs stored under an experiment-scoped layout:

        models/<exp_name>/<model_name>/<image files>

    This manager enforces the GenArena v2 constraint:
    - Within one subset, **model names must be globally unique across exp folders**.
      If the same model directory name appears under two different exp directories,
      this manager raises ValueError.
    """

    def __init__(self, models_root_dir: str):
        """
        Args:
            models_root_dir: Path to `arena_dir/<subset>/models`
        """
        self.models_root_dir = models_root_dir

        # model_name -> exp_name
        self._model_to_exp: dict[str, str] = {}
        # model_name -> {index -> filepath}
        self._index_map: dict[str, dict[int, str]] = {}
        # exp_name -> list[model_name]
        self._exp_to_models: dict[str, list[str]] = {}
        # cached sorted model list
        self._models: list[str] = []

        self._scan()

    def _scan(self) -> None:
        if not os.path.isdir(self.models_root_dir):
            warnings.warn(f"Models directory does not exist: {self.models_root_dir}")
            return

        for exp_name in os.listdir(self.models_root_dir):
            if exp_name.startswith("."):
                continue
            exp_dir = os.path.join(self.models_root_dir, exp_name)
            if not os.path.isdir(exp_dir):
                continue

            for model_name in os.listdir(exp_dir):
                if model_name.startswith("."):
                    continue
                model_dir = os.path.join(exp_dir, model_name)
                if not os.path.isdir(model_dir):
                    continue

                # Build index map for this model directory
                index_map: dict[int, str] = {}
                try:
                    filenames = os.listdir(model_dir)
                except Exception:
                    continue

                for filename in filenames:
                    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                        continue
                    idx = parse_image_index(filename)
                    if idx is None:
                        continue
                    filepath = os.path.join(model_dir, filename)
                    # If duplicate index, prefer .png
                    if idx in index_map:
                        if filename.lower().endswith(".png"):
                            index_map[idx] = filepath
                    else:
                        index_map[idx] = filepath

                # Ignore empty model directories (no images found)
                if not index_map:
                    continue

                if model_name in self._model_to_exp:
                    prev_exp = self._model_to_exp[model_name]
                    raise ValueError(
                        f"Duplicate model name across experiments: '{model_name}' found in both "
                        f"'{prev_exp}' and '{exp_name}'. Model names must be unique across exp folders."
                    )

                self._model_to_exp[model_name] = exp_name
                self._index_map[model_name] = index_map
                self._exp_to_models.setdefault(exp_name, []).append(model_name)

        for exp in self._exp_to_models:
            self._exp_to_models[exp].sort()

        self._models = sorted(self._model_to_exp.keys())

    @property
    def models(self) -> list[str]:
        """Get list of discovered model names."""
        return self._models.copy()

    @property
    def experiments(self) -> list[str]:
        """Get list of experiment names that contain at least one model."""
        return sorted(self._exp_to_models.keys())

    def get_model_exp(self, model: str) -> Optional[str]:
        """Return exp_name containing the model, or None if unknown."""
        return self._model_to_exp.get(model)

    def get_experiment_models(self, exp_name: str) -> list[str]:
        """Return models discovered under a specific exp directory."""
        return self._exp_to_models.get(exp_name, []).copy()

    def has_model(self, model: str) -> bool:
        return model in self._index_map

    def get_output_path(self, model: str, index: int) -> Optional[str]:
        if model not in self._index_map:
            return None
        return self._index_map[model].get(index)

    def get_model_indices(self, model: str) -> set[int]:
        if model not in self._index_map:
            return set()
        return set(self._index_map[model].keys())

    def validate_coverage(self, model_a: str, model_b: str, indices: list[int]) -> list[int]:
        indices_a = self.get_model_indices(model_a)
        indices_b = self.get_model_indices(model_b)

        valid_indices = []
        missing_a = []
        missing_b = []

        for idx in indices:
            has_a = idx in indices_a
            has_b = idx in indices_b

            if has_a and has_b:
                valid_indices.append(idx)
            else:
                if not has_a:
                    missing_a.append(idx)
                if not has_b:
                    missing_b.append(idx)

        if missing_a:
            warnings.warn(
                f"Model '{model_a}' missing outputs for {len(missing_a)} indices: "
                f"{missing_a[:5]}{'...' if len(missing_a) > 5 else ''}"
            )
        if missing_b:
            warnings.warn(
                f"Model '{model_b}' missing outputs for {len(missing_b)} indices: "
                f"{missing_b[:5]}{'...' if len(missing_b) > 5 else ''}"
            )

        return valid_indices

    def refresh(self) -> None:
        """Re-scan the models root directory."""
        self._model_to_exp = {}
        self._index_map = {}
        self._exp_to_models = {}
        self._models = []
        self._scan()

    def get_models_up_to_date(self, exp_date: date) -> list[str]:
        """Return models from experiments with date <= exp_date.

        This is used to ensure that when running battles for an old experiment,
        we only consider models from experiments that existed at that time
        (same date or earlier), not models from future experiments.

        Args:
            exp_date: The cutoff date (inclusive).

        Returns:
            List of model names from experiments with date <= exp_date.
        """
        result: list[str] = []
        for exp_name, models in self._exp_to_models.items():
            d = parse_exp_date_suffix(exp_name)
            if d is not None and d <= exp_date:
                result.extend(models)
        return sorted(result)
