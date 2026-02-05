# Experiment Management

This document explains how GenArena organizes experiments, handles incremental model additions, and manages historical battle data.

## Experiment Naming Convention

All experiments must follow the naming pattern:

```
<name>_YYYYMMDD
```

Examples:
- `Baseline_20260128`
- `NewModels_20260201`
- `GenArena_Official_20260315`

### Milestone Experiments

Experiments starting with `GenArena_` are **milestones**:

```
GenArena_Baseline_20260128
GenArena_Official_20260315
```

Milestones trigger:
1. ELO snapshot generation (`elo_snapshot.json`)
2. Higher minimum sample requirements (configurable)
3. Anchored fitting for subsequent experiments

## Directory Structure

```
arena_dir/
└── <subset>/
    ├── models/
    │   ├── Baseline_20260128/        # First experiment
    │   │   ├── model_a/
    │   │   ├── model_b/
    │   │   └── model_c/
    │   └── NewModels_20260201/       # Second experiment
    │       └── model_d/              # Only new models here
    │
    ├── pk_logs/
    │   ├── Baseline_20260128/        # Logs for first experiment
    │   │   ├── model_a_vs_model_b.jsonl
    │   │   ├── model_a_vs_model_c.jsonl
    │   │   └── model_b_vs_model_c.jsonl
    │   └── NewModels_20260201/       # Logs for second experiment
    │       ├── model_d_vs_model_a.jsonl
    │       ├── model_d_vs_model_b.jsonl
    │       └── model_d_vs_model_c.jsonl
    │
    └── arena/
        └── state.json               # Aggregated state from ALL logs
```

## Adding New Models

### Workflow

1. **Add model outputs** to a new experiment folder:

```bash
# Create new experiment directory
mkdir -p arena_dir/basic/models/NewModels_20260201/model_d/

# Copy model outputs
cp /path/to/model_d_outputs/* arena_dir/basic/models/NewModels_20260201/model_d/
```

2. **Run evaluation**:

```bash
genarena run --arena_dir ./arena --data_dir ./data \
    --exp_name NewModels_20260201
```

3. GenArena automatically:
   - Detects model_d as new
   - Schedules battles: model_d vs model_a, model_d vs model_b, model_d vs model_c
   - Skips historical battles (model_a vs model_b, etc.) - already completed
   - Computes ELO from ALL accumulated logs

### What Happens to Historical Logs?

**Historical logs are preserved and reused.** When computing ELO:

1. All logs from all experiments are loaded
2. Battle records are accumulated
3. BT-ELO is computed from the complete battle history

This means:
- Old battle logs contribute to ELO calculation
- No need to re-run historical battles
- Adding models is incremental, not from scratch

## Log Accumulation

### Battle Record Format

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

Key design decisions:
- **Models stored alphabetically**: `model_a < model_b` ensures consistent file naming
- **Winner stored by name**: Not "model_a"/"model_b" but actual model name
- **One file per pair**: All samples for a pair in one file

### Checkpoint/Resume

GenArena tracks completed battles via:

```python
completed = set()  # (model_a, model_b, sample_index)

for record in all_logs:
    completed.add((record.model_a, record.model_b, record.sample_index))

# Only run battles not in completed set
pending = all_pairs - completed
```

This enables:
- **Interrupt/resume**: Stop and continue without data loss
- **Incremental runs**: Only new model pairs are evaluated
- **Parallel workers**: Multiple processes can run without conflicts

## Milestone Snapshots

### Purpose

Milestones provide stable ELO anchors for comparing across time periods.

Without milestones:
```
Time 1: A=1050, B=950, C=1000
Time 2 (add D): A=1030, B=970, C=980, D=1020  # All ratings shifted!
```

With milestones:
```
Milestone: A=1050, B=950, C=1000 (fixed)
Time 2: A=1050, B=950, C=1000, D=1020  # Anchors preserved
```

### Snapshot Format

`pk_logs/GenArena_Baseline_20260128/elo_snapshot.json`:

```json
{
    "exp_name": "GenArena_Baseline_20260128",
    "generated_at": "2026-01-28T10:30:00Z",
    "params": {
        "scale": 400.0,
        "base": 10.0,
        "init_rating": 1000.0
    },
    "model_count": 5,
    "battle_count": 2500,
    "elo": {
        "model_a": 1085.5,
        "model_b": 1020.3,
        "model_c": 985.2,
        "model_d": 950.8,
        "model_e": 958.2
    },
    "ci_lower": {...},
    "ci_upper": {...},
    "ci_width": {...},
    "std": {...},
    "num_bootstrap": 100
}
```

### Anchored Fitting

When a milestone exists, new models are fit relative to anchors:

```python
# Load milestone snapshot
anchor_elo = load_snapshot("GenArena_Baseline_20260128")

# Fit only new models, keep anchors fixed
new_ratings = compute_bt_elo(
    battles_after_milestone,
    fixed_ratings=anchor_elo  # A, B, C fixed
)

# Result: A, B, C unchanged; D fitted relative to them
```

## Handling Model Removal

When models are removed from the data directory:

1. **Detection**: State sync detects models in logs but not in models directory
2. **Archival**: Orphaned logs moved to `.pk_logs_rm/` (not deleted)
3. **Rebuild**: ELO recomputed excluding removed models

```bash
# Logs moved to archive (recoverable)
arena_dir/<subset>/.pk_logs_rm/<exp_name>/model_removed_vs_*.jsonl
```

To disable auto-cleanup:
```bash
genarena run ... --no-clean-orphaned-logs
```

## Sampling Modes

### Full Mode

Run all possible sample combinations:

```bash
genarena run ... --sampling_mode full --sample_size 500
```

- Deterministic sample selection
- Useful for reproducible benchmarks
- Historical logs from full mode are fully reusable

### Adaptive Mode

Sample until CI converges:

```bash
genarena run ... \
    --sampling_mode adaptive \
    --target_ci_width 15.0 \
    --min_samples 100 \
    --max_samples 1500
```

Process:
1. Run initial batch (min_samples)
2. Compute bootstrap CI
3. For pairs with CI width > target: add more samples
4. Repeat until all pairs converge or hit max_samples

Historical logs work seamlessly:
- Existing samples count toward min_samples
- CI is computed from all available battles

## Multi-Experiment Workflow

### Example: Quarterly Evaluations

```
Q1: GenArena_Q1_20260331 (milestone)
    - Models: A, B, C, D, E
    - Full pairwise evaluation
    - Creates elo_snapshot.json

Q2: GenArena_Q2_20260630 (milestone)
    - New models: F, G
    - Only run: F vs *, G vs *
    - Anchored to Q1 snapshot
    - Creates new elo_snapshot.json

Q3: GenArena_Q3_20260930 (milestone)
    - New models: H
    - Only run: H vs *
    - Anchored to Q2 snapshot
```

Each milestone:
1. Includes all previous battle history
2. Anchors to the previous milestone
3. Creates its own snapshot for future anchoring

### Ad-hoc Experiments

Non-milestone experiments for quick testing:

```bash
# Quick comparison (not a milestone)
genarena run ... --exp_name QuickTest_20260715

# Results computed but no snapshot created
# Does not affect future milestone anchoring
```

## Best Practices

### 1. Use Milestones for Official Releases

```bash
# Official evaluation
genarena run ... --exp_name GenArena_Official_20260128
```

### 2. Use Descriptive Experiment Names

```
# Good
GenArena_Baseline_20260128
NewFluxModels_20260201
SDXLVariants_20260215

# Bad
test_20260128
run1_20260201
```

### 3. Keep Experiment Dates Chronological

The system sorts experiments by date suffix. Ensure dates are accurate for proper milestone ordering.

### 4. Don't Modify Historical Logs

Logs are append-only. Never manually edit JSONL files. If corrections are needed:
1. Delete the incorrect log file
2. Re-run affected battles

### 5. Regular Backups

Use Git or HuggingFace sync for versioning:

```bash
# Git sync
genarena git sync --arena_dir ./arena

# HuggingFace upload
genarena hf upload --arena_dir ./arena --repo_id user/my-arena
```

## Troubleshooting

### "Model not found in any experiment"

Ensure model outputs are in the correct location:
```
arena_dir/<subset>/models/<exp_name>/<model_name>/
```

### "Experiment name must end with _yyyymmdd"

Fix the experiment name:
```bash
# Wrong
--exp_name MyExperiment

# Correct
--exp_name MyExperiment_20260128
```

### "No battles to run"

All battles already completed. Check:
```bash
genarena status --arena_dir ./arena --data_dir ./data
```

To force re-evaluation, delete relevant log files and re-run.
