# GenArena Documentation

GenArena is a VLM-based pairwise evaluation system for image generation models using Bradley-Terry ELO ranking.

## Documentation Structure

| Document | Description |
|----------|-------------|
| [Quick Start](./quickstart.md) | Installation and basic usage guide |
| [Architecture](./architecture.md) | System design and key concepts |
| [CLI Reference](./cli-reference.md) | Complete command-line interface documentation |
| [Experiment Management](./experiments.md) | How to organize and manage experiments |
| [FAQ](./faq.md) | Frequently asked questions |

## Key Features

- **Pairwise Evaluation**: Compare image generation models head-to-head using VLM judges
- **Position Debiasing**: Double-call swap method to eliminate position bias
- **Bradley-Terry ELO**: Order-independent batch scoring (not online K-factor)
- **Incremental Evaluation**: Add new models without re-running historical battles
- **Milestone Anchoring**: Stable ELO scores across experiments via anchored fitting
- **Adaptive Sampling**: CI-based sampling to achieve target confidence efficiently
- **Multi-endpoint Support**: Load balancing across multiple VLM API endpoints
- **Git/HuggingFace Sync**: Built-in version control and dataset sharing

## Quick Links

- **Installation**: `uv pip install -e .`
- **Run battles**: `genarena run --arena_dir <path> --data_dir <path>`
- **View leaderboard**: `genarena leaderboard --arena_dir <path> --subset basic`
- **Start visualization**: `genarena serve --arena_dir <path> --data_dir <path>`

## System Requirements

- Python 3.10+
- VLM API access (OpenAI-compatible endpoint)
- Parquet dataset with prompts and model outputs
