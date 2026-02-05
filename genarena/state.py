# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Arena state management module (Bradley-Terry Elo scoring)."""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from genarena.utils import ensure_dir, iso_timestamp


# BT-to-Elo conversion constants (VideoAutoArena-style defaults)
# NOTE: This package intentionally uses *batch* Bradley-Terry scoring rather than
# online ELO with a K-factor, so scores are order-independent and reproducible.
SCALE = 400.0
BASE = 10.0
INIT_RATING = 1000.0

# Backward-compatible alias for existing state.json fields/defaults
DEFAULT_ELO = INIT_RATING

# Default number of bootstrap iterations for CI computation
DEFAULT_NUM_BOOTSTRAP = 100


@dataclass
class ModelStats:
    """Statistics for a single model."""

    elo: float = DEFAULT_ELO
    wins: int = 0
    losses: int = 0
    ties: int = 0
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None

    @property
    def total_battles(self) -> int:
        """Total number of battles."""
        return self.wins + self.losses + self.ties

    @property
    def win_rate(self) -> float:
        """Win rate (wins / total, ties count as 0.5)."""
        if self.total_battles == 0:
            return 0.0
        return (self.wins + 0.5 * self.ties) / self.total_battles

    @property
    def ci_width(self) -> Optional[float]:
        """95% CI width (upper - lower), or None if CI not computed."""
        if self.ci_lower is None or self.ci_upper is None:
            return None
        return self.ci_upper - self.ci_lower

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "elo": self.elo,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties
        }
        if self.ci_lower is not None:
            result["ci_lower"] = self.ci_lower
        if self.ci_upper is not None:
            result["ci_upper"] = self.ci_upper
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelStats":
        """Create from dictionary."""
        return cls(
            elo=data.get("elo", DEFAULT_ELO),
            wins=data.get("wins", 0),
            losses=data.get("losses", 0),
            ties=data.get("ties", 0),
            ci_lower=data.get("ci_lower"),
            ci_upper=data.get("ci_upper"),
        )


@dataclass
class ArenaState:
    """
    Arena state containing ELO ratings and battle statistics.

    Manages model ratings and provides methods for ELO updates.
    """

    # Model name -> ModelStats
    models: dict[str, ModelStats] = field(default_factory=dict)

    # Total battles processed
    total_battles: int = 0

    # Last update timestamp
    last_updated: str = ""

    def get_model_stats(self, model: str) -> ModelStats:
        """
        Get stats for a model, creating if necessary.

        Args:
            model: Model name

        Returns:
            ModelStats for the model
        """
        if model not in self.models:
            self.models[model] = ModelStats()
        return self.models[model]

    def get_elo(self, model: str) -> float:
        """Get ELO rating for a model."""
        return self.get_model_stats(model).elo

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "models": {
                name: stats.to_dict()
                for name, stats in self.models.items()
            },
            "total_battles": self.total_battles,
            "last_updated": self.last_updated
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArenaState":
        """Create from dictionary."""
        state = cls()

        models_data = data.get("models", {})
        for name, stats_data in models_data.items():
            state.models[name] = ModelStats.from_dict(stats_data)

        state.total_battles = data.get("total_battles", 0)
        state.last_updated = data.get("last_updated", "")

        return state


def update_stats(
    state: ArenaState,
    model_a: str,
    model_b: str,
    winner: str
) -> ArenaState:
    """
    Update win/loss/tie statistics based on a battle result.

    This does NOT update Elo ratings directly. Elo ratings are computed via
    Bradley-Terry model fitting from accumulated battle records (see
    `rebuild_state_from_logs`).

    Args:
        state: Current arena state
        model_a: First model name
        model_b: Second model name
        winner: "model_a", "model_b", or "tie" (or the actual model name)

    Returns:
        Updated arena state
    """
    stats_a = state.get_model_stats(model_a)
    stats_b = state.get_model_stats(model_b)

    # Determine actual scores based on winner
    winner_lower = winner.lower()

    if winner_lower == model_a.lower() or winner_lower == "model_a":
        # Model A wins
        stats_a.wins += 1
        stats_b.losses += 1
    elif winner_lower == model_b.lower() or winner_lower == "model_b":
        # Model B wins
        stats_a.losses += 1
        stats_b.wins += 1
    else:
        # Tie
        stats_a.ties += 1
        stats_b.ties += 1

    # Update state metadata
    state.total_battles += 1
    state.last_updated = iso_timestamp()

    return state


def load_state(path: str) -> ArenaState:
    """
    Load arena state from a JSON file.

    Args:
        path: Path to state.json file

    Returns:
        Loaded ArenaState, or empty state if file doesn't exist
    """
    if not os.path.isfile(path):
        return ArenaState()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ArenaState.from_dict(data)
    except (json.JSONDecodeError, IOError):
        return ArenaState()


def save_state(state: ArenaState, path: str) -> None:
    """
    Save arena state to a JSON file.

    Args:
        state: ArenaState to save
        path: Path to save to
    """
    # Ensure directory exists
    ensure_dir(os.path.dirname(path))

    # Update timestamp
    state.last_updated = iso_timestamp()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)


def rebuild_state_from_logs(
    pk_logs_dir: str,
    models: Optional[list[str]] = None
) -> ArenaState:
    """
    Rebuild arena state from battle log files.

    Recomputes:
    - W/L/T statistics
    - Bradley-Terry Elo ratings (VideoAutoArena-style), order-independent

    Args:
        pk_logs_dir: Path to pk_logs directory
        models: Optional list of models to include (includes all if None)

    Returns:
        Rebuilt ArenaState
    """
    from genarena.logs import load_battle_records
    from genarena.bt_elo import compute_bt_elo_ratings, compute_bootstrap_bt_elo
    from genarena.experiments import is_milestone_exp, parse_exp_date_suffix

    state = ArenaState()

    if not os.path.isdir(pk_logs_dir):
        return state

    # Discover experiment directories (enforce `_yyyymmdd` suffix)
    exp_keys: list[tuple[tuple, str]] = []
    # key is (date, name) for deterministic ordering
    for name in os.listdir(pk_logs_dir):
        if name.startswith("."):
            continue
        exp_dir = os.path.join(pk_logs_dir, name)
        if not os.path.isdir(exp_dir):
            continue
        d = parse_exp_date_suffix(name)
        if d is None:
            raise ValueError(
                f"Invalid experiment directory under pk_logs: '{name}'. "
                f"Expected exp_name ending with `_yyyymmdd`."
            )
        exp_keys.append(((d, name), name))
    exp_keys.sort(key=lambda x: x[0])

    milestones = [name for (key, name) in exp_keys if is_milestone_exp(name)]

    def _winner_side(model_a: str, model_b: str, winner: str) -> str:
        """Normalize winner to 'model_a'/'model_b'/'tie' relative to (model_a, model_b)."""
        w = str(winner).lower()
        if w == "tie":
            return "tie"
        if w == str(model_a).lower():
            return "model_a"
        if w == str(model_b).lower():
            return "model_b"
        return "tie"

    def _load_elo_snapshot(path: str) -> Optional[dict[str, float]]:
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if not isinstance(data, dict):
            return None

        # Accept either: {"elo": {...}} or a direct {model: elo} mapping.
        raw = data.get("elo") if isinstance(data.get("elo"), dict) else data
        if not isinstance(raw, dict):
            return None

        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                continue
        return out or None

    def _save_elo_snapshot(
        path: str,
        *,
        exp_name: str,
        elo: dict[str, float],
        model_count: int,
        battle_count: int,
        ci_lower: Optional[dict[str, float]] = None,
        ci_upper: Optional[dict[str, float]] = None,
        ci_width: Optional[dict[str, float]] = None,
        std: Optional[dict[str, float]] = None,
        num_bootstrap: Optional[int] = None,
    ) -> None:
        ensure_dir(os.path.dirname(path))
        payload: dict[str, Any] = {
            "exp_name": exp_name,
            "generated_at": iso_timestamp(),
            "params": {"scale": SCALE, "base": BASE, "init_rating": INIT_RATING},
            "model_count": int(model_count),
            "battle_count": int(battle_count),
            "elo": {k: float(v) for k, v in sorted(elo.items(), key=lambda x: x[0])},
        }
        
        # Include CI information if available
        if ci_lower is not None:
            payload["ci_lower"] = {k: float(v) for k, v in sorted(ci_lower.items(), key=lambda x: x[0])}
        if ci_upper is not None:
            payload["ci_upper"] = {k: float(v) for k, v in sorted(ci_upper.items(), key=lambda x: x[0])}
        if ci_width is not None:
            payload["ci_width"] = {k: float(v) for k, v in sorted(ci_width.items(), key=lambda x: x[0])}
        if std is not None:
            payload["std"] = {k: float(v) for k, v in sorted(std.items(), key=lambda x: x[0])}
        if num_bootstrap is not None:
            payload["num_bootstrap"] = int(num_bootstrap)
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _save_exp_readme(
        exp_dir: str,
        exp_name: str,
        elo: dict[str, float],
        model_count: int,
        battle_count: int,
        ci_lower: Optional[dict[str, float]] = None,
        ci_upper: Optional[dict[str, float]] = None,
    ) -> None:
        """Save README.md with cumulative leaderboard for an experiment directory."""
        from genarena.leaderboard import generate_experiment_readme
        readme_path = os.path.join(exp_dir, "README.md")
        content = generate_experiment_readme(
            exp_name=exp_name,
            elo=elo,
            model_count=model_count,
            battle_count=battle_count,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

    # If no milestones exist, fall back to the legacy full-fit behavior.
    # Also generate README.md for experiments missing them.
    if not milestones:
        # Track battles per experiment for README generation
        battles_cumulative: list[tuple[str, str, str]] = []
        models_seen_cumulative: set[str] = set()

        for (key, name) in exp_keys:
            exp_records = load_battle_records(pk_logs_dir, exp_name=name)

            for record in exp_records:
                model_a = record.get("model_a", "")
                model_b = record.get("model_b", "")
                winner = record.get("final_winner", "tie")

                if models:
                    if model_a not in models or model_b not in models:
                        continue

                if model_a and model_b:
                    update_stats(state, model_a, model_b, winner)
                    battles_cumulative.append((model_a, model_b, _winner_side(model_a, model_b, winner)))
                    models_seen_cumulative.add(model_a)
                    models_seen_cumulative.add(model_b)

            # Check if elo_snapshot.json or README.md is missing for this experiment
            exp_dir = os.path.join(pk_logs_dir, name)
            snapshot_path = os.path.join(exp_dir, "elo_snapshot.json")
            readme_path = os.path.join(exp_dir, "README.md")
            expected_models = sorted(models_seen_cumulative)

            if expected_models:
                existing_snapshot = _load_elo_snapshot(snapshot_path)
                need_snapshot = existing_snapshot is None or any(m not in existing_snapshot for m in expected_models)
                need_readme = not os.path.isfile(readme_path)

                if need_snapshot or need_readme:
                    bootstrap_result = compute_bootstrap_bt_elo(
                        battles_cumulative,
                        models=expected_models,
                        num_bootstrap=DEFAULT_NUM_BOOTSTRAP,
                        scale=SCALE,
                        base=BASE,
                        init_rating=INIT_RATING,
                    )
                    if need_snapshot:
                        _save_elo_snapshot(
                            snapshot_path,
                            exp_name=name,
                            elo=bootstrap_result.ratings,
                            model_count=len(expected_models),
                            battle_count=len(battles_cumulative),
                            ci_lower=bootstrap_result.ci_lower,
                            ci_upper=bootstrap_result.ci_upper,
                            ci_width=bootstrap_result.ci_width,
                            std=bootstrap_result.std,
                            num_bootstrap=bootstrap_result.num_bootstrap,
                        )
                    if need_readme:
                        _save_exp_readme(
                            exp_dir=exp_dir,
                            exp_name=name,
                            elo=bootstrap_result.ratings,
                            model_count=len(expected_models),
                            battle_count=len(battles_cumulative),
                            ci_lower=bootstrap_result.ci_lower,
                            ci_upper=bootstrap_result.ci_upper,
                        )

        include_models = models if models is not None else list(state.models.keys())
        
        # Compute bootstrap CI for final ratings
        bootstrap_result = compute_bootstrap_bt_elo(
            battles_cumulative,
            models=include_models,
            num_bootstrap=DEFAULT_NUM_BOOTSTRAP,
            scale=SCALE,
            base=BASE,
            init_rating=INIT_RATING,
        )

        for m in bootstrap_result.ratings:
            stats = state.get_model_stats(m)
            stats.elo = float(bootstrap_result.ratings[m])
            stats.ci_lower = bootstrap_result.ci_lower.get(m)
            stats.ci_upper = bootstrap_result.ci_upper.get(m)

        state.last_updated = iso_timestamp()
        return state

    # === Milestone mode ===
    # Ensure every milestone has an elo_snapshot.json (auto-generate if missing/incomplete),
    # then use the latest milestone snapshot as fixed anchors to insert newer models.
    milestone_set = set(milestones)
    latest_milestone_name = milestones[-1]
    latest_milestone_key = next(k for (k, name) in exp_keys if name == latest_milestone_name)

    models_filter = set(models) if models else None

    battles_all: list[tuple[str, str, str]] = []
    battles_after_latest: list[tuple[str, str, str]] = []
    models_seen_upto: set[str] = set()
    models_seen_all: set[str] = set()

    # Iterate experiments in order, accumulate battles, and generate snapshots at milestones.
    for (key, name) in exp_keys:
        exp_records = load_battle_records(pk_logs_dir, exp_name=name)

        for record in exp_records:
            model_a = record.get("model_a", "")
            model_b = record.get("model_b", "")
            winner = record.get("final_winner", "tie")

            if models_filter is not None:
                if model_a not in models_filter or model_b not in models_filter:
                    continue

            if not model_a or not model_b:
                continue

            update_stats(state, model_a, model_b, winner)

            side = _winner_side(model_a, model_b, winner)
            battles_all.append((model_a, model_b, side))
            models_seen_all.add(model_a)
            models_seen_all.add(model_b)

            models_seen_upto.add(model_a)
            models_seen_upto.add(model_b)

            if key > latest_milestone_key:
                battles_after_latest.append((model_a, model_b, side))

        if name in milestone_set:
            snapshot_path = os.path.join(pk_logs_dir, name, "elo_snapshot.json")
            expected_models = sorted(models_seen_upto)

            # If there are no models yet, don't generate an empty snapshot.
            if not expected_models:
                continue

            existing = _load_elo_snapshot(snapshot_path)
            if existing is None or any(m not in existing for m in expected_models):
                # Use bootstrap to compute ELO with CI for milestone snapshots
                bootstrap_result = compute_bootstrap_bt_elo(
                    battles_all,
                    models=expected_models,
                    num_bootstrap=DEFAULT_NUM_BOOTSTRAP,
                    scale=SCALE,
                    base=BASE,
                    init_rating=INIT_RATING,
                )
                _save_elo_snapshot(
                    snapshot_path,
                    exp_name=name,
                    elo=bootstrap_result.ratings,
                    model_count=len(expected_models),
                    battle_count=len(battles_all),
                    ci_lower=bootstrap_result.ci_lower,
                    ci_upper=bootstrap_result.ci_upper,
                    ci_width=bootstrap_result.ci_width,
                    std=bootstrap_result.std,
                    num_bootstrap=bootstrap_result.num_bootstrap,
                )
                # Also generate README.md for the milestone
                _save_exp_readme(
                    exp_dir=os.path.join(pk_logs_dir, name),
                    exp_name=name,
                    elo=bootstrap_result.ratings,
                    model_count=len(expected_models),
                    battle_count=len(battles_all),
                    ci_lower=bootstrap_result.ci_lower,
                    ci_upper=bootstrap_result.ci_upper,
                )
            else:
                # Snapshot exists, but check if README.md is missing
                readme_path = os.path.join(pk_logs_dir, name, "README.md")
                if not os.path.isfile(readme_path):
                    # Load CI info from snapshot if available
                    snapshot_ci_lower: Optional[dict[str, float]] = None
                    snapshot_ci_upper: Optional[dict[str, float]] = None
                    try:
                        with open(snapshot_path, "r", encoding="utf-8") as f:
                            snapshot_data = json.load(f)
                        snapshot_ci_lower = snapshot_data.get("ci_lower")
                        snapshot_ci_upper = snapshot_data.get("ci_upper")
                    except Exception:
                        pass
                    _save_exp_readme(
                        exp_dir=os.path.join(pk_logs_dir, name),
                        exp_name=name,
                        elo=existing,
                        model_count=len(expected_models),
                        battle_count=len(battles_all),
                        ci_lower=snapshot_ci_lower,
                        ci_upper=snapshot_ci_upper,
                    )
        else:
            # Non-milestone experiment: check if elo_snapshot.json or README.md is missing
            exp_dir = os.path.join(pk_logs_dir, name)
            snapshot_path = os.path.join(exp_dir, "elo_snapshot.json")
            readme_path = os.path.join(exp_dir, "README.md")
            expected_models = sorted(models_seen_upto)

            if expected_models:
                existing_snapshot = _load_elo_snapshot(snapshot_path)
                need_snapshot = existing_snapshot is None or any(m not in existing_snapshot for m in expected_models)
                need_readme = not os.path.isfile(readme_path)

                if need_snapshot or need_readme:
                    bootstrap_result = compute_bootstrap_bt_elo(
                        battles_all,
                        models=expected_models,
                        num_bootstrap=DEFAULT_NUM_BOOTSTRAP,
                        scale=SCALE,
                        base=BASE,
                        init_rating=INIT_RATING,
                    )
                    if need_snapshot:
                        _save_elo_snapshot(
                            snapshot_path,
                            exp_name=name,
                            elo=bootstrap_result.ratings,
                            model_count=len(expected_models),
                            battle_count=len(battles_all),
                            ci_lower=bootstrap_result.ci_lower,
                            ci_upper=bootstrap_result.ci_upper,
                            ci_width=bootstrap_result.ci_width,
                            std=bootstrap_result.std,
                            num_bootstrap=bootstrap_result.num_bootstrap,
                        )
                    if need_readme:
                        _save_exp_readme(
                            exp_dir=exp_dir,
                            exp_name=name,
                            elo=bootstrap_result.ratings,
                            model_count=len(expected_models),
                            battle_count=len(battles_all),
                            ci_lower=bootstrap_result.ci_lower,
                            ci_upper=bootstrap_result.ci_upper,
                        )

    # Load anchors from the latest milestone snapshot (it should exist now, if milestone had any models).
    latest_snapshot_path = os.path.join(pk_logs_dir, latest_milestone_name, "elo_snapshot.json")
    anchor_elo = _load_elo_snapshot(latest_snapshot_path) or {}

    include_models = list(models) if models is not None else sorted(models_seen_all)
    anchor_elo = {m: float(v) for m, v in anchor_elo.items() if m in set(include_models)}

    # Final ratings: anchored insertion from latest milestone snapshot.
    # Compute bootstrap CI for final ratings
    if anchor_elo:
        bootstrap_result = compute_bootstrap_bt_elo(
            battles_after_latest,
            models=include_models,
            fixed_ratings=anchor_elo,
            num_bootstrap=DEFAULT_NUM_BOOTSTRAP,
            scale=SCALE,
            base=BASE,
            init_rating=INIT_RATING,
        )
    else:
        bootstrap_result = compute_bootstrap_bt_elo(
            battles_all,
            models=include_models,
            num_bootstrap=DEFAULT_NUM_BOOTSTRAP,
            scale=SCALE,
            base=BASE,
            init_rating=INIT_RATING,
        )

    for m in bootstrap_result.ratings:
        stats = state.get_model_stats(m)
        stats.elo = float(bootstrap_result.ratings[m])
        stats.ci_lower = bootstrap_result.ci_lower.get(m)
        stats.ci_upper = bootstrap_result.ci_upper.get(m)

    state.last_updated = iso_timestamp()
    return state
