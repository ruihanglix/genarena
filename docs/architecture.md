# System Architecture

This document explains GenArena's architecture, key components, and data flow.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              GenArena                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ Data     │───▶│ Arena    │───▶│ VLM      │───▶│ Battle Logger    │   │
│  │ Loader   │    │ Scheduler│    │ Judge    │    │ (JSONL)          │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘   │
│       │                                                   │              │
│       ▼                                                   ▼              │
│  ┌──────────┐                                      ┌──────────────────┐  │
│  │ Parquet  │                                      │ State Rebuilder  │  │
│  │ Dataset  │                                      │ (BT-ELO)         │  │
│  └──────────┘                                      └──────────────────┘  │
│                                                           │              │
│                                                           ▼              │
│                                                    ┌──────────────────┐  │
│                                                    │ Leaderboard      │  │
│                                                    │ (README.md)      │  │
│                                                    └──────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Modules

### `arena.py` - Main Orchestration

The `Arena` class coordinates the entire evaluation process:

```python
class Arena:
    def __init__(self, config: ArenaConfig):
        # Initialize VLM judge, data loader, battle scheduler
        
    def run(self) -> ArenaState:
        # 1. Discover models from arena_dir
        # 2. Generate battle pairs (model_a, model_b, sample_index)
        # 3. Filter out completed battles (checkpoint/resume)
        # 4. Execute battles in parallel
        # 5. Rebuild state from all logs
        # 6. Return updated state
```

Key features:
- **Parallel execution**: `ThreadPoolExecutor` for API calls
- **Multi-process sharding**: `ProcessPoolExecutor` for large datasets
- **Checkpoint/resume**: Skip already-completed battles
- **Adaptive sampling**: CI-based stopping criterion

### `battle.py` - Single Battle Execution

Executes one pairwise comparison with position debiasing:

```python
def execute_battle(judge, sample, img_a, img_b):
    # Original call: [img_a, img_b]
    result_1 = judge.evaluate(sample, img_a, img_b)
    
    # Swapped call: [img_b, img_a]
    result_2 = judge.evaluate(sample, img_b, img_a)
    
    # Combine results
    if results_agree(result_1, result_2):
        return winner, is_consistent=True
    else:
        return "tie", is_consistent=False
```

### `vlm.py` - VLM API Client

Handles communication with VLM endpoints:

```python
class VLMJudge:
    def __init__(self, model, base_urls, api_keys):
        # Support multiple endpoints for load balancing
        
    def evaluate(self, sample, img_a, img_b) -> dict:
        # Build prompt messages
        # Call VLM API with retry logic
        # Parse structured response
```

Features:
- **Multi-endpoint**: Round-robin across API endpoints
- **Retry logic**: Exponential backoff on failures
- **Rate limiting**: Handle 429 errors gracefully

### `bt_elo.py` - Rating Computation

Standalone module for Bradley-Terry ELO calculation:

```python
# No internal dependencies - can be used independently

def compute_bt_elo_ratings(battles, fixed_ratings=None):
    # Build win matrix
    # MM algorithm for BT fitting
    # Convert to ELO scale
    
def compute_bootstrap_bt_elo(battles, num_bootstrap=100):
    # Bootstrap resampling for confidence intervals
```

See [ELO Algorithm](./elo-algorithm.md) for details.

### `state.py` - State Management

Manages arena state and ELO rebuilding:

```python
@dataclass
class ArenaState:
    models: dict[str, ModelStats]
    elo: dict[str, float]
    ci_lower: dict[str, float]
    ci_upper: dict[str, float]
    total_battles: int
    last_updated: str

def rebuild_state_from_logs(pk_logs_dir, models):
    # Load all battle records
    # Detect milestones for anchored fitting
    # Compute BT-ELO with bootstrap CI
    # Return complete ArenaState
```

### `logs.py` - Battle Logging

Thread-safe logging of battle results:

```python
class BattleLogger:
    def log_battle(self, record: BattleRecord):
        # Slim log: essential fields only
        # Audit log: full VLM responses
        # File locking for concurrent writes

def load_battle_records(pk_logs_dir) -> list[BattleRecord]:
    # Load all JSONL files
    # Parse into structured records
```

### `data.py` - Dataset Loading

Handles Parquet dataset access:

```python
class ParquetDataLoader:
    def __init__(self, data_dir, subset):
        # Discover parquet files
        # Build sample index
        
    def get_sample(self, index) -> dict:
        # Load specific sample
        # Return prompt + model outputs
```

## Directory Layout

### Data Directory

```
data_dir/
├── basic/                     # Subset name
│   ├── data.parquet          # Single file
│   └── shard_*.parquet       # Or multiple shards
├── advanced/
│   └── ...
```

### Arena Directory

```
arena_dir/
└── <subset>/
    ├── models/
    │   └── <exp_name>_yyyymmdd/
    │       └── <model_name>/
    │           └── <sample_index>.png
    │
    ├── pk_logs/
    │   └── <exp_name>_yyyymmdd/
    │       ├── <model_a>_vs_<model_b>.jsonl  # Battle logs
    │       ├── raw_outputs/
    │       │   └── <model_a>_vs_<model_b>.jsonl  # Audit logs
    │       ├── elo_snapshot.json             # Milestone snapshot
    │       └── config.json                   # Experiment config
    │
    ├── arena/
    │   └── state.json                        # Current aggregated state
    │
    └── README.md                             # Leaderboard markdown
```

## Data Flow

### 1. Battle Execution

```
Parquet Dataset
      │
      ▼
┌─────────────┐
│ Load Sample │ (prompt, model outputs)
└─────────────┘
      │
      ▼
┌─────────────┐     ┌─────────────┐
│ VLM Call 1  │────▶│ VLM Call 2  │  (position swap)
└─────────────┘     └─────────────┘
      │                   │
      └─────────┬─────────┘
                ▼
        ┌─────────────┐
        │ Combine     │ (agree -> winner, disagree -> tie)
        └─────────────┘
                │
                ▼
        ┌─────────────┐
        │ Log Result  │ (JSONL + audit)
        └─────────────┘
```

### 2. State Rebuilding

```
pk_logs/*.jsonl
      │
      ▼
┌─────────────────┐
│ Load All Logs   │
└─────────────────┘
      │
      ▼
┌─────────────────┐     ┌─────────────────┐
│ Check Milestones│────▶│ Load Snapshot   │ (if exists)
└─────────────────┘     └─────────────────┘
      │                         │
      │    ┌────────────────────┘
      ▼    ▼
┌─────────────────┐
│ BT-ELO Compute  │ (with/without anchors)
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Bootstrap CI    │
└─────────────────┘
      │
      ▼
┌─────────────────┐
│ Save State      │ (state.json)
└─────────────────┘
```

## Log Formats

### Battle Log (Slim)

```json
{
    "model_a": "flux_1",
    "model_b": "sdxl",
    "sample_index": 42,
    "final_winner": "flux_1",
    "is_consistent": true,
    "timestamp": "2026-01-28T10:30:00Z"
}
```

### Audit Log (Full)

```json
{
    "model_a": "flux_1",
    "model_b": "sdxl",
    "sample_index": 42,
    "timestamp": "2026-01-28T10:30:00Z",
    "original_call": {
        "raw_response": "Based on the comparison...",
        "parsed_result": {"winner": "A", "score_a": 8, "score_b": 6},
        "parse_success": true
    },
    "swapped_call": {
        "raw_response": "Comparing the two images...",
        "parsed_result": {"winner": "B", "score_a": 6, "score_b": 8},
        "parse_success": true
    },
    "final_winner": "flux_1",
    "is_consistent": true
}
```

## Thread Safety

GenArena ensures thread-safe operation:

### File Locking

```python
import fcntl

with open(log_path, "a") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    try:
        f.write(json.dumps(record) + "\n")
    finally:
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

### Atomic State Updates

- State is rebuilt from logs (source of truth)
- No in-memory state mutations during battles
- Final state write is atomic (temp file + rename)

## Extensibility

### Custom Prompts

Add new prompt modules in `genarena/prompts/`:

```python
# genarena/prompts/my_prompt.py

def build_messages(sample, img_a, img_b) -> list[dict]:
    """Build VLM messages for comparison."""
    return [
        {"role": "system", "content": "..."},
        {"role": "user", "content": [...]}
    ]

def parse_response(response: str) -> dict:
    """Parse VLM response to extract winner."""
    return {"winner": "A" | "B" | "tie", ...}
```

Use with: `--prompt my_prompt`

### Custom Judges

Implement the `VLMJudge` interface:

```python
class MyJudge:
    def evaluate(self, sample, img_a, img_b) -> dict:
        # Custom evaluation logic
        return {"winner": ..., "raw_response": ...}
```

## Performance Considerations

### Parallelization

| Level | Mechanism | Use Case |
|-------|-----------|----------|
| API calls | `ThreadPoolExecutor` | I/O-bound VLM requests |
| Shards | `ProcessPoolExecutor` | Large parquet datasets |
| Swap calls | Optional parallel | Trade latency vs rate limits |

### Memory Management

- Parquet files loaded lazily by shard
- Images loaded on-demand per battle
- Battle logs appended (no full reload)

### Checkpointing

Battles are logged immediately after completion. On restart:
1. Load completed battle history
2. Filter out completed (model_a, model_b, sample_index) tuples
3. Only run remaining battles
