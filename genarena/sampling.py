# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Adaptive sampling configuration and utilities for battle scheduling.

This module provides configuration and logic for adaptive sampling strategies
to reduce the number of battles needed while maintaining statistical precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SamplingConfig:
    """Configuration for battle sampling strategy.
    
    Supports two modes:
    - "full": Traditional full pairwise comparison (打满所有 samples)
    - "adaptive": Adaptive sampling based on CI convergence (自适应采样)
    
    Attributes:
        mode: Sampling mode, either "full" or "adaptive".
        
        # Adaptive mode parameters
        min_samples: Minimum samples per model pair before checking CI.
        max_samples: Maximum samples per model pair (hard cap).
        batch_size: Number of samples to add in each adaptive iteration.
        target_ci_width: Target 95% CI width (full width, not ±).
            Sampling stops when CI width <= target_ci_width.
        num_bootstrap: Number of bootstrap iterations for CI computation.
        
        # Full mode parameters  
        sample_size: Fixed number of samples per pair (None = all available).
        
        # Milestone parameters
        milestone_min_samples: Minimum samples per pair for milestone experiments.
            Used to ensure milestone snapshots have sufficient precision.
    """
    
    mode: str = "adaptive"
    
    # Adaptive mode parameters
    min_samples: int = 100
    max_samples: int = 1500
    batch_size: int = 100
    target_ci_width: float = 15.0  # ±7.5 Elo
    num_bootstrap: int = 100
    
    # Full mode parameters
    sample_size: Optional[int] = None
    
    # Milestone parameters
    milestone_min_samples: int = 1000
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.mode not in ("full", "adaptive"):
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'full' or 'adaptive'.")
        
        if self.min_samples < 1:
            raise ValueError(f"min_samples must be >= 1, got {self.min_samples}")
        
        if self.max_samples < self.min_samples:
            raise ValueError(
                f"max_samples ({self.max_samples}) must be >= min_samples ({self.min_samples})"
            )
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        if self.target_ci_width <= 0:
            raise ValueError(f"target_ci_width must be > 0, got {self.target_ci_width}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode,
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "batch_size": self.batch_size,
            "target_ci_width": self.target_ci_width,
            "num_bootstrap": self.num_bootstrap,
            "sample_size": self.sample_size,
            "milestone_min_samples": self.milestone_min_samples,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SamplingConfig":
        """Create from dictionary."""
        return cls(
            mode=data.get("mode", "adaptive"),
            min_samples=data.get("min_samples", 100),
            max_samples=data.get("max_samples", 1500),
            batch_size=data.get("batch_size", 100),
            target_ci_width=data.get("target_ci_width", 15.0),
            num_bootstrap=data.get("num_bootstrap", 100),
            sample_size=data.get("sample_size"),
            milestone_min_samples=data.get("milestone_min_samples", 1000),
        )
    
    @classmethod
    def full_mode(cls, sample_size: Optional[int] = None) -> "SamplingConfig":
        """Create a full-mode configuration.
        
        Args:
            sample_size: Fixed sample size per pair (None = all available).
        
        Returns:
            SamplingConfig in full mode.
        """
        return cls(mode="full", sample_size=sample_size)
    
    @classmethod
    def adaptive_mode(
        cls,
        target_ci_width: float = 15.0,
        min_samples: int = 100,
        max_samples: int = 1500,
    ) -> "SamplingConfig":
        """Create an adaptive-mode configuration.
        
        Args:
            target_ci_width: Target 95% CI width.
            min_samples: Minimum samples before checking CI.
            max_samples: Maximum samples per pair.
        
        Returns:
            SamplingConfig in adaptive mode.
        """
        return cls(
            mode="adaptive",
            target_ci_width=target_ci_width,
            min_samples=min_samples,
            max_samples=max_samples,
        )


@dataclass
class PairSamplingState:
    """Tracks sampling state for a single model pair.
    
    This state is derived from existing battle logs, enabling resume.
    """
    
    model_a: str
    model_b: str
    current_samples: int = 0
    ci_width: Optional[float] = None
    converged: bool = False
    
    def needs_more_samples(self, config: SamplingConfig) -> bool:
        """Check if this pair needs more samples.
        
        Args:
            config: Sampling configuration.
        
        Returns:
            True if more samples are needed.
        """
        if config.mode == "full":
            if config.sample_size is None:
                return True  # Will be bounded by available samples
            return self.current_samples < config.sample_size
        
        # Adaptive mode
        if self.current_samples < config.min_samples:
            return True
        
        if self.current_samples >= config.max_samples:
            return False
        
        if self.converged:
            return False
        
        if self.ci_width is not None and self.ci_width <= config.target_ci_width:
            self.converged = True
            return False
        
        return True
    
    def get_samples_to_run(self, config: SamplingConfig, available: int) -> int:
        """Calculate how many samples to run in the next batch.
        
        Args:
            config: Sampling configuration.
            available: Number of available samples (dataset size).
        
        Returns:
            Number of samples to run (can be 0).
        """
        if not self.needs_more_samples(config):
            return 0
        
        if config.mode == "full":
            target = config.sample_size if config.sample_size else available
            return max(0, min(target, available) - self.current_samples)
        
        # Adaptive mode
        if self.current_samples < config.min_samples:
            # Initial batch: reach min_samples
            target = min(config.min_samples, available)
            return max(0, target - self.current_samples)
        
        # Subsequent batches: add batch_size
        target = min(self.current_samples + config.batch_size, config.max_samples, available)
        return max(0, target - self.current_samples)


@dataclass 
class AdaptiveSamplingScheduler:
    """Scheduler for adaptive sampling across all model pairs.
    
    Manages the sampling state for all pairs and determines which
    pairs need more battles based on CI convergence.
    """
    
    config: SamplingConfig
    pair_states: dict[tuple[str, str], PairSamplingState] = field(default_factory=dict)
    
    def get_or_create_state(self, model_a: str, model_b: str) -> PairSamplingState:
        """Get or create sampling state for a model pair.
        
        Args:
            model_a: First model name.
            model_b: Second model name.
        
        Returns:
            PairSamplingState for this pair.
        """
        # Normalize pair order
        key = (min(model_a, model_b), max(model_a, model_b))
        
        if key not in self.pair_states:
            self.pair_states[key] = PairSamplingState(
                model_a=key[0],
                model_b=key[1],
            )
        
        return self.pair_states[key]
    
    def update_state(
        self,
        model_a: str,
        model_b: str,
        current_samples: int,
        ci_width: Optional[float] = None,
    ) -> None:
        """Update sampling state for a model pair.
        
        Args:
            model_a: First model name.
            model_b: Second model name.
            current_samples: Current number of samples.
            ci_width: Current CI width (if computed).
        """
        state = self.get_or_create_state(model_a, model_b)
        state.current_samples = current_samples
        state.ci_width = ci_width
        
        # Check convergence
        if ci_width is not None and ci_width <= self.config.target_ci_width:
            state.converged = True
    
    def get_pairs_needing_samples(self) -> list[tuple[str, str]]:
        """Get list of pairs that need more samples.
        
        Returns:
            List of (model_a, model_b) tuples needing more samples.
        """
        result = []
        for key, state in self.pair_states.items():
            if state.needs_more_samples(self.config):
                result.append(key)
        return result
    
    def get_pair_with_widest_ci(self) -> Optional[tuple[str, str]]:
        """Get the pair with the widest CI that hasn't converged.
        
        Returns:
            (model_a, model_b) tuple, or None if all converged.
        """
        widest_pair = None
        widest_ci = -1.0
        
        for key, state in self.pair_states.items():
            if state.converged:
                continue
            if state.current_samples >= self.config.max_samples:
                continue
            if state.ci_width is not None and state.ci_width > widest_ci:
                widest_ci = state.ci_width
                widest_pair = key
        
        return widest_pair
    
    def all_converged(self) -> bool:
        """Check if all pairs have converged.
        
        Returns:
            True if all pairs have converged or reached max_samples.
        """
        for state in self.pair_states.values():
            if state.needs_more_samples(self.config):
                return False
        return True
    
    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics of current sampling state.
        
        Returns:
            Dictionary with summary statistics.
        """
        total_pairs = len(self.pair_states)
        converged_pairs = sum(1 for s in self.pair_states.values() if s.converged)
        maxed_pairs = sum(
            1 for s in self.pair_states.values() 
            if s.current_samples >= self.config.max_samples
        )
        
        ci_widths = [s.ci_width for s in self.pair_states.values() if s.ci_width is not None]
        total_samples = sum(s.current_samples for s in self.pair_states.values())
        
        return {
            "total_pairs": total_pairs,
            "converged_pairs": converged_pairs,
            "maxed_pairs": maxed_pairs,
            "pending_pairs": total_pairs - converged_pairs - maxed_pairs,
            "total_samples": total_samples,
            "mean_ci_width": sum(ci_widths) / len(ci_widths) if ci_widths else None,
            "max_ci_width": max(ci_widths) if ci_widths else None,
            "min_ci_width": min(ci_widths) if ci_widths else None,
        }
