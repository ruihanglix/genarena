# Contribution Guidelines

Thank you for contributing to the GenArena leaderboard! This document outlines how to submit your evaluation results.

## Prerequisites

1. **GitHub CLI (`gh`)**: Install from https://cli.github.com
   ```bash
   # Authenticate with GitHub
   gh auth login
   ```

2. **HuggingFace Account**: Create a Dataset repository to host your evaluation data
   ```bash
   # Login to HuggingFace
   huggingface-cli login
   ```

3. **genarena**: Install the genarena package
   ```bash
   pip install genarena
   ```

## Submission Requirements

### Model Requirements

- **New Models**: Your submission must include at least one model that is not already on the official leaderboard
- **Model Outputs**: All generated images must be included in the submission
- **Consistent Naming**: Model names should be descriptive and consistent

### Evaluation Requirements

- **Complete Battles**: Include all battle logs for your evaluation
- **Position Debiasing**: Use the standard double-call swap method
- **Judge Model**: Use the same VLM judge model as official evaluations (recommended: `Qwen/Qwen3-VL-32B-Instruct-FP8`)

### Experiment Naming

- Experiment names must end with `_yyyymmdd` format
- Example: `MyNewModel_20260130`

## How to Submit

### 1. Run Your Evaluation

```bash
genarena run \
  --arena_dir /path/to/arena \
  --data_dir /path/to/data \
  --subset basic \
  --exp_name MyNewModel_20260130 \
  --models MyNewModel,BaselineModel
```

### 2. Submit Results

```bash
genarena submit \
  --arena_dir /path/to/arena \
  --subset basic \
  --exp_name MyNewModel_20260130 \
  --hf_repo your-username/your-genarena-results
```

The command will:
1. Validate your local data
2. Upload to your HuggingFace repository
3. Create a Pull Request to this repository

### 3. Wait for Validation

- The GitHub Actions bot will automatically validate your submission
- Check the PR comments for the validation report
- Fix any issues and update your submission if needed

### 4. Maintainer Review

- A maintainer will review your submission
- If approved, your models will be added to the official leaderboard

## Validation Checks

The automated validation checks:

1. **Schema Validation**: Submission JSON follows the required format
2. **Data Accessibility**: Files are accessible on HuggingFace
3. **Checksum Verification**: SHA256 checksums match
4. **Battle Count**: Reported battle count matches actual data
5. **ELO Verification**: Reported ELO scores match recalculated values
6. **New Model Check**: At least one model is new to the leaderboard

## Troubleshooting

### "No new models found"

Your submission must include at least one model that is not already on the official leaderboard. Check `official_models.json` for the current list.

### "SHA256 checksum mismatch"

Ensure you haven't modified the data after upload. Re-run `genarena submit` to re-upload.

### "ELO mismatch"

The ELO scores in your submission should match what genarena calculates from the battle logs. This might indicate corrupted or modified battle logs.

## Questions?

If you have questions or issues, please open an issue in this repository.
