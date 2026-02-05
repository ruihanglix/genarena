# CLI Reference

Complete reference for all GenArena command-line commands and options.

## Global Options

```bash
genarena --version  # Show version
genarena --help     # Show help
```

---

## `genarena init`

Initialize arena and download data from official HuggingFace repositories.

### Synopsis

```bash
genarena init [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arena_dir` | `./arena` | Arena directory path |
| `--data_dir` | `./data` | Benchmark data directory path |
| `--subsets` | (all) | Comma-separated list of subsets to download |
| `--git` | `false` | Initialize Git repository in arena_dir |
| `--data-only` | `false` | Only download benchmark Parquet data |
| `--arena-only` | `false` | Only download arena data (model outputs + logs) |
| `--benchmark-repo` | `rhli/genarena` | HuggingFace repo for benchmark data |
| `--arena-repo` | `rhli/genarena-battlefield` | HuggingFace repo for arena data |
| `--revision` | `main` | HuggingFace revision/branch |
| `--overwrite` | `false` | Overwrite existing files |
| `--verbose` | `false` | Enable verbose output |

### Examples

```bash
# Full initialization (benchmark + arena data)
genarena init --arena_dir ./arena --data_dir ./data

# Download only benchmark Parquet data
genarena init --data_dir ./data --data-only

# Download only arena data (model outputs + battle logs)
genarena init --arena_dir ./arena --arena-only

# Download specific subsets only
genarena init --arena_dir ./arena --data_dir ./data --subsets basic,reasoning

# Initialize with Git repository
genarena init --arena_dir ./arena --data_dir ./data --git

# Use custom HuggingFace repositories
genarena init \
    --benchmark-repo myorg/my-benchmark \
    --arena-repo myorg/my-arena

# Overwrite existing files
genarena init --arena_dir ./arena --data_dir ./data --overwrite
```

### Output

```
=== GenArena Init ===

[Step 1/2] Downloading benchmark data from rhli/genarena...
  Target directory: ./data

  Downloading basic/data-00000-of-00002.parquet... 156.3 MB ✓
  Downloading basic/data-00001-of-00002.parquet... 142.7 MB ✓

  Benchmark data download complete:
    Downloaded: 2 files (299.0 MB)
    Skipped: 0 files (already exist)
    Failed: 0 files

[Step 2/2] Downloading arena data from rhli/genarena-battlefield...
  Target directory: ./arena

  Downloaded: 8, Skipped: 0, Failed: 0

=== Summary ===
Data directory:  ./data
Arena directory: ./arena
Subsets:         basic
  Benchmark data: 2 files (299.0 MB)
  Arena data: downloaded to ./arena

Next steps:
  # View current status
  genarena status --arena_dir ./arena --data_dir ./data

  # Run evaluation battles
  genarena run --arena_dir ./arena --data_dir ./data

  # View leaderboard
  genarena leaderboard --arena_dir ./arena --subset basic
```

---

## `genarena run`

Run pairwise evaluation battles.

### Synopsis

```bash
genarena run --arena_dir <path> --data_dir <path> [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--arena_dir` | Arena directory for storing outputs and logs |
| `--data_dir` | Parquet dataset directory |

### Optional Arguments

#### Scope Options

| Option | Default | Description |
|--------|---------|-------------|
| `--subset` | (all) | Process specific subset only |
| `--models` | (all) | Comma-separated list of models to include |
| `--exp_name` | (auto) | Experiment name (must end with `_yyyymmdd`) |
| `--sample_size` | (all) | Limit samples per model pair |

#### Sampling Mode Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sampling_mode` | `adaptive` | `adaptive` (CI-based) or `full` (all samples) |
| `--target_ci_width` | `15.0` | Target 95% CI width for adaptive mode |
| `--min_samples` | `100` | Minimum samples before checking CI |
| `--max_samples` | `1500` | Maximum samples per pair (hard cap) |
| `--milestone_min_samples` | `1000` | Minimum samples for milestone experiments |

#### VLM Options

| Option | Default | Description |
|--------|---------|-------------|
| `--judge_model` | `Qwen/Qwen3-VL-32B-Instruct-FP8` | VLM model name |
| `--temperature` | `0.0` | VLM temperature (0 = greedy) |
| `--prompt` | `mmrb2` | Prompt module name |
| `--timeout` | `120` | API timeout in seconds |
| `--max_retries` | `3` | Maximum retry attempts |
| `--base_urls` | (env) | Comma-separated VLM API base URLs |
| `--api_keys` | (env) | Comma-separated API keys |

#### Parallelization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num_threads` | `8` | Number of parallel API threads |
| `--num_processes` | `1` | Number of processes (shard by parquet file) |
| `--parallel_swap_calls` | `false` | Run original+swapped calls in parallel |

#### Other Options

| Option | Default | Description |
|--------|---------|-------------|
| `--enable_progress_bar` | `false` | Show progress bar |
| `--no-audit-log` | `false` | Disable audit logging |
| `--no-clean-orphaned-logs` | `false` | Keep logs for removed models |
| `--verbose` | `false` | Enable debug logging |

### Examples

```bash
# Basic run
genarena run --arena_dir ./arena --data_dir ./data

# Run specific subset with custom experiment name
genarena run --arena_dir ./arena --data_dir ./data \
    --subset basic \
    --exp_name MyExp_20260128

# Full mode with limited samples
genarena run --arena_dir ./arena --data_dir ./data \
    --sampling_mode full \
    --sample_size 500

# Adaptive mode with tight CI target
genarena run --arena_dir ./arena --data_dir ./data \
    --sampling_mode adaptive \
    --target_ci_width 10.0 \
    --min_samples 200

# High parallelism with multiple endpoints
genarena run --arena_dir ./arena --data_dir ./data \
    --num_threads 16 \
    --base_urls "https://api1.example.com/v1,https://api2.example.com/v1" \
    --api_keys "key1,key2"

# Run with progress bar
genarena run --arena_dir ./arena --data_dir ./data \
    --enable_progress_bar \
    --verbose
```

---

## `genarena status`

Show arena status summary.

### Synopsis

```bash
genarena status --arena_dir <path> --data_dir <path>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--arena_dir` | Arena directory |
| `--data_dir` | Parquet dataset directory |
| `--verbose` | Enable verbose output |

### Example

```bash
genarena status --arena_dir ./arena --data_dir ./data
```

Output:
```
=== Arena Status ===

Arena Directory: ./arena
Data Directory: ./data

Subset: basic
  Models: 5 (flux1_kontext, qwen_image_edit, gpt_image_1...)
  Total Battles: 1250
  Last Updated: 2026-01-28 10:30:00

Subset: reasoning
  Models: 5 (flux1_kontext, qwen_image_edit, gpt_image_1...)
  Total Battles: 800
  Last Updated: 2026-01-28 09:15:00
```

---

## `genarena leaderboard`

Display ELO leaderboard for a subset.

### Synopsis

```bash
genarena leaderboard --arena_dir <path> --subset <name>
```

### Arguments

| Argument | Description |
|----------|-------------|
| `--arena_dir` | Arena directory |
| `--subset` | Subset name to display |
| `--verbose` | Enable verbose output |

### Example

```bash
genarena leaderboard --arena_dir ./arena --subset basic
```

---

## `genarena serve`

Start the battle visualization web server.

### Synopsis

```bash
genarena serve --arena_dir <path> --data_dir <path> [options]
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--arena_dir` | (required) | Arena directory |
| `--data_dir` | (required) | Parquet dataset directory |
| `--host` | `0.0.0.0` | Host to bind server |
| `--port` | `8080` | Port to listen on |
| `--debug` | `false` | Enable Flask debug mode |

### Example

```bash
# Start server on default port
genarena serve --arena_dir ./arena --data_dir ./data

# Custom host and port
genarena serve --arena_dir ./arena --data_dir ./data \
    --host 127.0.0.1 --port 3000

# Debug mode
genarena serve --arena_dir ./arena --data_dir ./data --debug
```

---

## Git Commands

Git version control for arena directories.

### `genarena git init`

Initialize Git repository.

```bash
genarena git init --arena_dir <path>
```

### `genarena git commit`

Commit changes.

```bash
genarena git commit --arena_dir <path> [-m "message"]
```

| Option | Description |
|--------|-------------|
| `-m, --message` | Custom commit message (default: auto-generated) |

### `genarena git remote`

Configure remote repository.

```bash
# Show current remote
genarena git remote --arena_dir <path>

# Set remote URL
genarena git remote --arena_dir <path> --url <url>

# Force overwrite existing remote
genarena git remote --arena_dir <path> --url <url> --force
```

### `genarena git push`

Push commits to remote.

```bash
genarena git push --arena_dir <path>
```

### `genarena git sync`

Commit and push in one command.

```bash
genarena git sync --arena_dir <path>
```

---

## Hugging Face Commands

Sync arena data with Hugging Face Dataset repositories.

### `genarena hf upload`

Upload arena data to Hugging Face.

```bash
genarena hf upload --arena_dir <path> --repo_id <username/repo> [options]
```

| Option | Description |
|--------|-------------|
| `--subsets` | Comma-separated list of subsets (default: all) |
| `--models` | Comma-separated list of models (default: all) |
| `--experiments` | Comma-separated list of experiments (default: all) |
| `--overwrite` | Overwrite existing files |
| `--max-retries` | Max retries per file (default: 3) |
| `--verbose` | Enable verbose output |

### Example

```bash
# Upload all data
genarena hf upload --arena_dir ./arena --repo_id myuser/my-arena

# Upload specific subset
genarena hf upload --arena_dir ./arena --repo_id myuser/my-arena \
    --subsets basic --overwrite
```

### `genarena hf pull`

Download arena data from Hugging Face.

```bash
genarena hf pull --arena_dir <path> --repo_id <username/repo> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--subsets` | (all) | Comma-separated list of subsets |
| `--models` | (all) | Comma-separated list of models |
| `--experiments` | (all) | Comma-separated list of experiments |
| `--revision` | `main` | Repository revision/branch |
| `--overwrite` | `false` | Overwrite existing local files |
| `--verbose` | `false` | Enable verbose output |

### Example

```bash
# Pull all data
genarena hf pull --arena_dir ./arena --repo_id myuser/my-arena

# Pull specific revision
genarena hf pull --arena_dir ./arena --repo_id myuser/my-arena \
    --revision v1.0 --overwrite
```

### `genarena hf list`

List contents of a Hugging Face repository.

```bash
genarena hf list --repo_id <username/repo> [--revision <branch>]
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Single API key |
| `OPENAI_API_KEYS` | Comma-separated API keys (multi-endpoint) |
| `OPENAI_BASE_URL` | Single API base URL |
| `OPENAI_BASE_URLS` | Comma-separated base URLs (multi-endpoint) |

### Multi-endpoint Example

```bash
export OPENAI_API_KEYS="key1,key2,key3"
export OPENAI_BASE_URLS="https://api1.com/v1,https://api2.com/v1,https://api3.com/v1"

genarena run --arena_dir ./arena --data_dir ./data --num_threads 24
```

Requests are distributed across endpoints in round-robin fashion.

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (check stderr for details) |
