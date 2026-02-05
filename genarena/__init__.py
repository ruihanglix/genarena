"""GenArena Arena Evaluation - VLM-based pairwise image generation evaluation."""

__version__ = "0.1.0"
__author__ = "GenArena Team"

"""
Keep package import lightweight.

This subpackage may be split out and installed independently. Some modules
(e.g., dataset IO) require optional heavy dependencies (pyarrow, pandas, etc.).
To avoid import-time failures, we expose symbols via lazy imports in __getattr__.
"""

__all__ = [
    "__version__",
    "__author__",
    "Arena",
    "ParquetDataset",
    "DataSample",
    "ModelOutputManager",
    "VLMJudge",
    "ArenaState",
    "generate_leaderboard",
]


def __getattr__(name: str):
    """Lazy attribute loader to keep import side-effects minimal."""
    if name == "Arena":
        from .arena import Arena
        return Arena
    if name == "ParquetDataset":
        from .data import ParquetDataset
        return ParquetDataset
    if name == "DataSample":
        from .data import DataSample
        return DataSample
    if name == "ModelOutputManager":
        from .models import ModelOutputManager
        return ModelOutputManager
    if name == "VLMJudge":
        from .vlm import VLMJudge
        return VLMJudge
    if name == "ArenaState":
        from .state import ArenaState
        return ArenaState
    if name == "generate_leaderboard":
        from .leaderboard import generate_leaderboard
        return generate_leaderboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
