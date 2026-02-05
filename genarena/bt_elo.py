# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Bradley-Terry (BT) Elo rating utilities.

This module intentionally has **no** dependencies on code outside this package,
so `genarena_arena_evaluation` can be split out as an independent package.

The implementation follows the common BT-to-Elo conversion used by
VideoAutoArena-style scoring:
- Build a pairwise "win matrix" where a win counts as 2 points and a tie counts
  as 1 point for each model.
- Fit BT parameters via a simple MM (minorization-maximization) iterative update
  (no sklearn dependency).
- Convert BT parameters to Elo scale: Elo = SCALE * beta + INIT_RATING, where
  beta is on a log(BASE) scale.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

# BT-to-Elo conversion constants (VideoAutoArena-style defaults)
SCALE: float = 400.0
BASE: float = 10.0
INIT_RATING: float = 1000.0

# Public type: (model_a, model_b, winner) where winner is "model_a"/"model_b"/"tie"
BattleTuple = tuple[str, str, str]


def compute_bt_elo_ratings(
    battles: Sequence[BattleTuple],
    *,
    models: Optional[Iterable[str]] = None,
    scale: float = SCALE,
    base: float = BASE,
    init_rating: float = INIT_RATING,
    fixed_ratings: Optional[dict[str, float]] = None,
    max_iters: int = 100,
    tol: float = 1e-6,
) -> dict[str, float]:
    """Compute BT-derived Elo ratings from battle outcomes.

    Args:
        battles: List of (model_a, model_b, winner) tuples.
            winner must be one of: "model_a", "model_b", "tie".
        models: Optional iterable of model names to include even if they have no battles.
        scale: Elo scale factor (default 400).
        base: Log base used for BT parameterization (default 10).
        init_rating: Elo offset / initial rating (default 1000).
        fixed_ratings: Optional mapping of model name -> Elo rating for fixed anchors.
            When provided and non-empty, those models' Elo values are kept **exactly**
            unchanged, and only the remaining models are fit via BT MM updates.
        max_iters: Maximum MM iterations.
        tol: Convergence tolerance on the probability vector.

    Returns:
        Dict mapping model name to Elo rating.
    """
    model_set = set(models or [])
    for a, b, _ in battles:
        if a:
            model_set.add(a)
        if b:
            model_set.add(b)

    model_list = sorted(model_set)
    if not model_list:
        return {}
    if len(model_list) == 1:
        return {model_list[0]: init_rating}

    fixed_ratings = fixed_ratings or {}
    fixed_set = set(fixed_ratings.keys())
    if fixed_set:
        # Filter anchors to only those in the requested model universe
        fixed_set = fixed_set & set(model_list)
        fixed_ratings = {m: float(fixed_ratings[m]) for m in fixed_set}

    if fixed_set:
        # === Anchored BT fit in strength space ===
        # We represent BT strengths as positive numbers s_m such that:
        #   P(i beats j) = s_i / (s_i + s_j)
        # and convert to Elo via:
        #   Elo = init_rating + scale * log_base(s)
        #
        # This is equivalent to the existing implementation (which uses probabilities
        # normalized to sum=1) but avoids global renormalization so fixed anchors
        # can remain unchanged even when new models are introduced.
        W: dict[str, dict[str, float]] = {m: {} for m in model_list}
        for i in model_list:
            for j in model_list:
                if i != j:
                    W[i][j] = 0.0

        # Adjacency for connectivity checks (n_ij > 0 indicates at least one battle)
        adj: dict[str, set[str]] = {m: set() for m in model_list}

        for model_a, model_b, winner in battles:
            if not model_a or not model_b or model_a == model_b:
                continue
            if model_a not in W or model_b not in W:
                continue
            if winner == "model_a":
                W[model_a][model_b] += 2.0
            elif winner == "model_b":
                W[model_b][model_a] += 2.0
            else:
                W[model_a][model_b] += 1.0
                W[model_b][model_a] += 1.0
            adj[model_a].add(model_b)
            adj[model_b].add(model_a)

        log_base = math.log(base)
        # Initialize strengths
        s: dict[str, float] = {}
        for m in model_list:
            if m in fixed_set:
                beta = (fixed_ratings[m] - init_rating) / scale
                s[m] = base ** beta
            else:
                s[m] = 1.0

        free = [m for m in model_list if m not in fixed_set]

        for _ in range(max_iters):
            max_diff = 0.0
            for i in free:
                num = 0.0
                denom = 0.0
                for j in model_list:
                    if i == j:
                        continue
                    w_ij = W[i][j]
                    w_ji = W[j][i]
                    n_ij = w_ij + w_ji
                    if n_ij <= 0:
                        continue
                    num += w_ij
                    denom += n_ij / (s[i] + s[j])

                s_new = (num / denom) if denom > 0 else s[i]
                max_diff = max(max_diff, abs(s_new - s[i]))
                s[i] = s_new

            if max_diff <= tol:
                break

        # If any free-model connected component has no path to fixed anchors,
        # its absolute scale is unidentifiable. Normalize such components to mean(s)=1
        # to keep their average Elo at init_rating, without affecting likelihood.
        visited: set[str] = set()
        fixed_neighbors = fixed_set

        for m in free:
            if m in visited:
                continue
            stack = [m]
            comp: list[str] = []
            visited.add(m)
            connected_to_fixed = False
            while stack:
                x = stack.pop()
                comp.append(x)
                for y in adj.get(x, set()):
                    if y in fixed_neighbors:
                        connected_to_fixed = True
                        continue
                    if y in fixed_set:
                        connected_to_fixed = True
                        continue
                    if y in free and y not in visited:
                        visited.add(y)
                        stack.append(y)
            if not connected_to_fixed and comp:
                mean_s = sum(s[x] for x in comp) / len(comp)
                if mean_s > 0:
                    for x in comp:
                        s[x] = s[x] / mean_s

        ratings: dict[str, float] = {}
        for m in model_list:
            if m in fixed_set:
                ratings[m] = float(fixed_ratings[m])
            else:
                if s[m] > 0:
                    ratings[m] = scale * (math.log(s[m]) / log_base) + init_rating
                else:
                    ratings[m] = init_rating

        return ratings

    # Win matrix W where W[i][j] is "points" of i over j:
    # - win: +2 to winner over loser
    # - tie: +1 to each direction
    W: dict[str, dict[str, float]] = {m: {} for m in model_list}
    for i in model_list:
        for j in model_list:
            if i != j:
                W[i][j] = 0.0

    for model_a, model_b, winner in battles:
        if not model_a or not model_b or model_a == model_b:
            continue
        if winner == "model_a":
            W[model_a][model_b] += 2.0
        elif winner == "model_b":
            W[model_b][model_a] += 2.0
        else:
            # Tie
            W[model_a][model_b] += 1.0
            W[model_b][model_a] += 1.0

    # MM algorithm on probabilities p_i (sum to 1)
    n = len(model_list)
    p: dict[str, float] = {m: 1.0 / n for m in model_list}

    for _ in range(max_iters):
        p_new: dict[str, float] = {}
        for i in model_list:
            num = 0.0
            denom = 0.0
            for j in model_list:
                if i == j:
                    continue
                w_ij = W[i][j]
                w_ji = W[j][i]
                n_ij = w_ij + w_ji
                if n_ij <= 0:
                    continue
                num += w_ij
                denom += n_ij / (p[i] + p[j])

            p_new[i] = (num / denom) if denom > 0 else p[i]

        total = sum(p_new.values())
        if total > 0:
            for k in p_new:
                p_new[k] /= total

        max_diff = max(abs(p_new[m] - p[m]) for m in model_list)
        p = p_new
        if max_diff <= tol:
            break

    mean_p = sum(p.values()) / n if n > 0 else 0.0
    if mean_p <= 0:
        return {m: init_rating for m in model_list}

    log_base = math.log(base)
    ratings: dict[str, float] = {}
    for m in model_list:
        if p[m] > 0:
            beta = math.log(p[m] / mean_p) / log_base
            ratings[m] = scale * beta + init_rating
        else:
            ratings[m] = init_rating

    return ratings


@dataclass
class BootstrapResult:
    """Result of bootstrap ELO computation with confidence intervals.
    
    Attributes:
        ratings: Point estimates of ELO ratings (median of bootstrap samples).
        ci_lower: Lower bound of 95% CI (2.5th percentile).
        ci_upper: Upper bound of 95% CI (97.5th percentile).
        ci_width: Width of 95% CI (ci_upper - ci_lower) for each model.
        std: Standard deviation of bootstrap samples.
        num_battles: Number of battles used for computation.
        num_bootstrap: Number of bootstrap iterations performed.
    """
    ratings: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    ci_width: dict[str, float]
    std: dict[str, float]
    num_battles: int
    num_bootstrap: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ratings": self.ratings,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_width": self.ci_width,
            "std": self.std,
            "num_battles": self.num_battles,
            "num_bootstrap": self.num_bootstrap,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BootstrapResult":
        """Create from dictionary."""
        return cls(
            ratings=data.get("ratings", {}),
            ci_lower=data.get("ci_lower", {}),
            ci_upper=data.get("ci_upper", {}),
            ci_width=data.get("ci_width", {}),
            std=data.get("std", {}),
            num_battles=data.get("num_battles", 0),
            num_bootstrap=data.get("num_bootstrap", 0),
        )
    
    def get_model_ci_width(self, model: str) -> float:
        """Get CI width for a specific model."""
        return self.ci_width.get(model, float("inf"))
    
    def get_max_ci_width(self) -> float:
        """Get the maximum CI width across all models."""
        if not self.ci_width:
            return float("inf")
        return max(self.ci_width.values())
    
    def get_mean_ci_width(self) -> float:
        """Get the mean CI width across all models."""
        if not self.ci_width:
            return float("inf")
        return sum(self.ci_width.values()) / len(self.ci_width)


def compute_bootstrap_bt_elo(
    battles: Sequence[BattleTuple],
    *,
    models: Optional[Iterable[str]] = None,
    num_bootstrap: int = 100,
    scale: float = SCALE,
    base: float = BASE,
    init_rating: float = INIT_RATING,
    fixed_ratings: Optional[dict[str, float]] = None,
    seed: Optional[int] = None,
) -> BootstrapResult:
    """Compute BT Elo ratings with 95% confidence intervals via bootstrap.
    
    Uses multinomial resampling on unique (model_a, model_b, outcome) counts
    for efficiency, following the FastChat approach.
    
    Args:
        battles: List of (model_a, model_b, winner) tuples.
        models: Optional iterable of model names to include.
        num_bootstrap: Number of bootstrap iterations (default 100).
        scale: Elo scale factor (default 400).
        base: Log base for BT parameterization (default 10).
        init_rating: Elo offset / initial rating (default 1000).
        fixed_ratings: Optional mapping of model name -> Elo for fixed anchors.
        seed: Random seed for reproducibility.
    
    Returns:
        BootstrapResult with ratings and confidence intervals.
    """
    if seed is not None:
        random.seed(seed)
    
    battles_list = list(battles)
    n_battles = len(battles_list)
    
    if n_battles == 0:
        return BootstrapResult(
            ratings={},
            ci_lower={},
            ci_upper={},
            ci_width={},
            std={},
            num_battles=0,
            num_bootstrap=num_bootstrap,
        )
    
    # Get point estimate
    point_ratings = compute_bt_elo_ratings(
        battles_list,
        models=models,
        scale=scale,
        base=base,
        init_rating=init_rating,
        fixed_ratings=fixed_ratings,
    )
    
    model_list = sorted(point_ratings.keys())
    if not model_list:
        return BootstrapResult(
            ratings={},
            ci_lower={},
            ci_upper={},
            ci_width={},
            std={},
            num_battles=n_battles,
            num_bootstrap=num_bootstrap,
        )
    
    # Count unique battle outcomes for efficient multinomial resampling
    battle_counts = Counter(battles_list)
    unique_battles = list(battle_counts.keys())
    counts = [battle_counts[b] for b in unique_battles]
    total_count = sum(counts)
    
    # Bootstrap iterations
    bootstrap_ratings: list[dict[str, float]] = []
    
    for _ in range(num_bootstrap):
        # Multinomial resampling of counts
        sampled_counts = _multinomial_sample(total_count, counts)
        
        # Reconstruct battles from sampled counts
        sampled_battles: list[BattleTuple] = []
        for battle, count in zip(unique_battles, sampled_counts):
            sampled_battles.extend([battle] * count)
        
        # Compute ratings for this bootstrap sample
        sample_ratings = compute_bt_elo_ratings(
            sampled_battles,
            models=models,
            scale=scale,
            base=base,
            init_rating=init_rating,
            fixed_ratings=fixed_ratings,
        )
        bootstrap_ratings.append(sample_ratings)
    
    # Compute statistics
    ratings_matrix: dict[str, list[float]] = {m: [] for m in model_list}
    for br in bootstrap_ratings:
        for m in model_list:
            ratings_matrix[m].append(br.get(m, init_rating))
    
    ci_lower: dict[str, float] = {}
    ci_upper: dict[str, float] = {}
    ci_width: dict[str, float] = {}
    std: dict[str, float] = {}
    median_ratings: dict[str, float] = {}
    
    for m in model_list:
        values = sorted(ratings_matrix[m])
        n = len(values)
        
        # Percentiles for 95% CI
        lower_idx = int(n * 0.025)
        upper_idx = int(n * 0.975)
        median_idx = n // 2
        
        ci_lower[m] = values[lower_idx] if n > 0 else init_rating
        ci_upper[m] = values[min(upper_idx, n - 1)] if n > 0 else init_rating
        ci_width[m] = ci_upper[m] - ci_lower[m]
        median_ratings[m] = values[median_idx] if n > 0 else init_rating
        
        # Standard deviation
        if n > 1:
            mean_val = sum(values) / n
            variance = sum((v - mean_val) ** 2 for v in values) / (n - 1)
            std[m] = math.sqrt(variance)
        else:
            std[m] = 0.0
    
    return BootstrapResult(
        ratings=median_ratings,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        ci_width=ci_width,
        std=std,
        num_battles=n_battles,
        num_bootstrap=num_bootstrap,
    )


def _multinomial_sample(n: int, weights: list[int]) -> list[int]:
    """Sample from multinomial distribution.
    
    Args:
        n: Total number of samples to draw.
        weights: Unnormalized weights (counts) for each category.
    
    Returns:
        List of counts for each category summing to n.
    """
    total_weight = sum(weights)
    if total_weight == 0:
        return [0] * len(weights)
    
    # Normalize to probabilities
    probs = [w / total_weight for w in weights]
    
    # Sample n items according to probabilities
    result = [0] * len(weights)
    for _ in range(n):
        r = random.random()
        cumsum = 0.0
        for i, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                result[i] += 1
                break
        else:
            # Edge case: assign to last category
            result[-1] += 1
    
    return result


