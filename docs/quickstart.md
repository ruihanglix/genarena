# Quick Start Guide

This guide will help you get GenArena up and running quickly.

## Installation

```bash
# Clone the repository
git clone https://github.com/ruihanglix/genarena.git
cd genarena

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

## One-Click Setup with `genarena init`

The fastest way to get started is using the `init` command, which downloads all necessary data from official repositories:

```bash
# Download benchmark data + official arena data
genarena init --arena_dir ./arena --data_dir ./data

# Download only specific subsets
genarena init --arena_dir ./arena --data_dir ./data --subsets basic

# Also initialize Git repository
genarena init --arena_dir ./arena --data_dir ./data --git
```

This will:
1. Download benchmark Parquet data from `rhli/genarena` (HuggingFace)
2. Download official arena data (model outputs + battle logs) from `rhli/genarena-battlefield`
3. Optionally initialize a Git repository for version control

After initialization, you can immediately run evaluations or view the leaderboard.

> **Note**: For private HuggingFace repositories, set the `HF_TOKEN` environment variable first.

## Environment Setup

Set your VLM API credentials:

```bash
# Single endpoint
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.example.com/v1"

# Multiple endpoints (load balancing)
export OPENAI_API_KEYS="key1,key2,key3"
export OPENAI_BASE_URLS="https://api1.example.com/v1,https://api2.example.com/v1"
```

## Directory Structure

GenArena requires two directories:

### 1. Data Directory (`data_dir`)

Contains Parquet files with prompts and model outputs:

```
data_dir/
├── basic/                    # Subset name
│   ├── data.parquet         # Or multiple shards
│   └── shard_0.parquet
├── advanced/
│   └── data.parquet
```

Each Parquet file should contain columns like:
- `prompt`: Text prompt
- `<model_name>`: Path to generated image for each model

### 2. Arena Directory (`arena_dir`)

Stores model outputs, battle logs, and state:

```
arena_dir/
└── <subset>/
    ├── models/<exp_name>_yyyymmdd/<model>/   # Model outputs
    ├── pk_logs/<exp_name>_yyyymmdd/          # Battle logs
    │   ├── *.jsonl                           # Battle results
    │   ├── raw_outputs/*.jsonl               # Audit logs
    │   └── elo_snapshot.json                 # Milestone snapshot
    ├── arena/state.json                      # Current state
    └── README.md                             # Leaderboard
```

## Basic Usage

### 1. Run Evaluation Battles

```bash
# Run on all subsets
genarena run --arena_dir /path/to/arena --data_dir /path/to/data

# Run on specific subset
genarena run --arena_dir /path/to/arena --data_dir /path/to/data --subset basic

# Run with specific experiment name
genarena run --arena_dir /path/to/arena --data_dir /path/to/data \
    --exp_name MyExperiment_20260128
```

### 2. Check Status

```bash
genarena status --arena_dir /path/to/arena --data_dir /path/to/data
```

Output example:
```
=== Arena Status ===

Arena Directory: /path/to/arena
Data Directory: /path/to/data

Subset: basic
  Models: 5 (model_a, model_b, model_c...)
  Total Battles: 1250
  Last Updated: 2026-01-28 10:30:00
```

### 3. View Leaderboard

```bash
genarena leaderboard --arena_dir /path/to/arena --subset basic
```

Output example:
```
╔════════════════════════════════════════════════════════════════════╗
║                        Basic Leaderboard                           ║
╠══════╤═══════════════╤══════╤══════════════╤═══════╤═══════╤═══════╣
║ Rank │ Model         │ ELO  │ 95% CI       │ Wins  │ Losses│ Ties  ║
╠══════╪═══════════════╪══════╪══════════════╪═══════╪═══════╪═══════╣
║ 1    │ model_a       │ 1085 │ [1070, 1100] │ 320   │ 180   │ 100   ║
║ 2    │ model_b       │ 1020 │ [1005, 1035] │ 250   │ 230   │ 120   ║
║ 3    │ model_c       │ 985  │ [970, 1000]  │ 200   │ 280   │ 120   ║
╚══════╧═══════════════╧══════╧══════════════╧═══════╧═══════╧═══════╝
```

### 4. Start Visualization Server

```bash
genarena serve --arena_dir /path/to/arena --data_dir /path/to/data --port 8080
```

Open `http://localhost:8080` in your browser to explore battles interactively.

## Sampling Modes

### Full Mode

Run all possible sample battles:

```bash
genarena run --arena_dir ... --data_dir ... \
    --sampling_mode full \
    --sample_size 500  # Optional: limit samples per pair
```

### Adaptive Mode (Default)

Dynamically sample until CI converges:

```bash
genarena run --arena_dir ... --data_dir ... \
    --sampling_mode adaptive \
    --target_ci_width 15.0 \  # Target ±7.5 ELO
    --min_samples 100 \       # Minimum before checking CI
    --max_samples 1500        # Hard cap
```

## Adding New Models

1. Add model outputs to the data directory
2. Run arena - only new model pairs will be evaluated:

```bash
# GenArena automatically detects new models and schedules their battles
genarena run --arena_dir ... --data_dir ... --exp_name NewModels_20260128
```

Historical battles are preserved and reused in ELO calculation.

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--num_threads` | Parallel API threads | 8 |
| `--judge_model` | VLM model name | Qwen/Qwen3-VL-32B-Instruct-FP8 |
| `--temperature` | VLM temperature | 0.0 (greedy) |
| `--timeout` | API timeout (seconds) | 120 |
| `--verbose` | Enable debug logging | False |
| `--enable_progress_bar` | Show progress bar | False |

## Next Steps

- Read [Architecture](./architecture.md) to understand system design
- See [CLI Reference](./cli-reference.md) for all commands and options
