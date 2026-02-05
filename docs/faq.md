# Frequently Asked Questions (FAQ)

## General

### Q: What is GenArena?

GenArena is a VLM-based pairwise evaluation system for image generation models. It uses Vision-Language Models (VLMs) to compare images head-to-head and computes Bradley-Terry ELO ratings for ranking models.

### Q: How is GenArena different from other evaluation methods?

| Approach | GenArena | Human Evaluation | Automatic Metrics |
|----------|----------|------------------|-------------------|
| Scalability | High | Low | High |
| Cost | Medium | High | Low |
| Consistency | High | Medium | High |
| Human alignment | Medium-High | High | Low-Medium |
| Explainability | Medium | High | Low |

GenArena offers a balance: more scalable than human evaluation, more aligned with human preference than automatic metrics like FID/CLIP.

---

## Historical Data & New Models

### Q: Can I reuse old battle logs when adding new models?

**Yes!** This is a core feature of GenArena. Historical battle logs are automatically accumulated and reused:

1. When you add a new model, GenArena only runs battles involving the new model
2. Historical battles (between existing models) are NOT re-run
3. ELO is computed from ALL accumulated battle logs (historical + new)

Example:
```
# Historical state: models A, B, C with 1000 battles
# Add model D

# GenArena will:
# - Run: D vs A, D vs B, D vs C (new battles only)
# - Skip: A vs B, A vs C, B vs C (already completed)
# - Compute ELO from: all 1000 historical + new D battles
```

---

## ELO Ratings

### Q: Why do ELO scores change when I add a new model (without milestones)?

Without milestone anchoring, ELO is recomputed from scratch using all battles. Adding new battles changes the overall distribution, which can shift all ratings.

**Solution**: Use milestone experiments to anchor ratings:
```bash
genarena run ... --exp_name GenArena_xxxx
```

Future experiments will anchor to this snapshot, keeping baseline models' ratings stable.

### Q: What does the 95% CI mean?

The 95% Confidence Interval indicates the uncertainty in the ELO estimate. For example:
```
Model A: ELO = 1050 [1035, 1065]
```

This means we're 95% confident the "true" ELO is between 1035 and 1065.

**Interpretation**:
- Narrow CI (e.g., ±10): High confidence, many battles
- Wide CI (e.g., ±50): Low confidence, need more battles

### Q: How many battles do I need for reliable rankings?

Rule of thumb for adaptive mode defaults:
- **100 samples/pair**: Minimum for CI calculation
- **200-500 samples/pair**: Reasonable confidence (CI ~20-30)
- **500+ samples/pair**: High confidence (CI <15)

Use `--target_ci_width` to specify your target precision:
```bash
# High precision (more battles)
genarena run ... --target_ci_width 10.0

# Lower precision (fewer battles)
genarena run ... --target_ci_width 30.0
```

### Q: Why does GenArena use Bradley-Terry instead of traditional ELO?

Traditional (K-factor) ELO has problems:
1. **Order-dependent**: Different battle orders = different final ratings
2. **Sensitive to K-factor**: Hard to choose the right value
3. **Non-reproducible**: Can't verify results

Bradley-Terry (batch) ELO:
1. **Order-independent**: Same battles always = same ratings
2. **No hyperparameters**: Theoretically principled
3. **Reproducible**: Deterministic from battle logs

---

## Position Bias

### Q: What is position bias and how does GenArena handle it?

VLMs tend to favor images based on their position (first or second) regardless of quality. GenArena uses **double-call swap** to mitigate this:

```
Call 1: VLM(img_A, img_B) -> "A wins"
Call 2: VLM(img_B, img_A) -> "B wins" (same as "A wins" after swap)

If both agree -> A wins
If they disagree -> Tie
```

This doubles the API calls but eliminates position bias.

---

## Performance

### Q: How can I speed up evaluation?

1. **Increase threads**:
   ```bash
   genarena run ... --num_threads 16
   ```

2. **Use multiple API endpoints**:
   ```bash
   export OPENAI_BASE_URLS="url1,url2,url3"
   export OPENAI_API_KEYS="key1,key2,key3"
   genarena run ... --num_threads 24
   ```

3. **Enable parallel swap calls** (caution: may increase 429 errors):
   ```bash
   genarena run ... --parallel_swap_calls
   ```

4. **Use multi-process for large datasets**:
   ```bash
   genarena run ... --num_processes 4
   ```

### Q: I'm getting 429 (rate limit) errors. What should I do?

1. **Reduce threads**: `--num_threads 4`
2. **Add more API endpoints**: Load balance across multiple keys
3. **Disable parallel swap calls**: Remove `--parallel_swap_calls`
4. **Wait and retry**: GenArena has built-in retry with backoff

### Q: How much does evaluation cost?

Cost depends on:
- VLM model pricing
- Number of model pairs
- Samples per pair
- Image sizes

Rough estimate for 5 models, 500 samples/pair:
- Pairs: 5 × 4 / 2 = 10 pairs
- Battles: 10 × 500 = 5,000 battles
- API calls: 5,000 × 2 (swap) = 10,000 calls

At typical VLM pricing ($0.01-0.05 per call with images), total: $100-500.

---

## Troubleshooting

### Q: "ModuleNotFoundError: No module named 'genarena'"

Install the package:
```bash
uv pip install -e .
# or
pip install -e .
```

### Q: "OPENAI_API_KEY not set"

Set environment variables:
```bash
export OPENAI_API_KEY="your-key"
export OPENAI_BASE_URL="https://api.example.com/v1"
```

Or pass directly:
```bash
genarena run ... --api_keys "your-key" --base_urls "https://api.example.com/v1"
```

### Q: "No subsets found"

Check your data directory structure:
```
data_dir/
├── subset1/
│   └── *.parquet
├── subset2/
│   └── *.parquet
```

Subset names are directory names containing parquet files.

### Q: "Experiment name must end with _yyyymmdd"

Fix the experiment name format:
```bash
# Wrong
--exp_name MyExperiment

# Correct
--exp_name MyExperiment_20260128
```

### Q: "State file is corrupted"

State is rebuilt from logs, so you can safely delete it:
```bash
rm arena_dir/<subset>/arena/state.json
genarena run ...  # State will be rebuilt
```

### Q: How do I completely restart an evaluation?

Delete the pk_logs directory for the experiment:
```bash
rm -rf arena_dir/<subset>/pk_logs/<exp_name>/
genarena run ... --exp_name <exp_name>
```

**Warning**: This deletes all battle history for that experiment.
