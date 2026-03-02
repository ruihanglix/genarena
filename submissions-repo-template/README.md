# GenArena Submissions

This repository receives community submissions of evaluation results for the GenArena leaderboard.

## How to Submit

Use the `genarena submit` command to submit your evaluation results:

```bash
genarena submit \
  --arena_dir /path/to/your/arena \
  --subset basic \
  --exp_name YourModel_20260130 \
  --hf_repo your-username/your-hf-repo
```

### Requirements

1. **GitHub CLI (`gh`)** - Install from https://cli.github.com and authenticate with `gh auth login`
2. **HuggingFace Account** - Create a Dataset repository to store your evaluation data
3. **genarena** - Install with `pip install genarena`

### Submission Guidelines

1. **New Models Only**: Submissions must include at least one model that is not already on the official leaderboard
2. **Complete Evaluation**: Include all battle logs and model outputs
3. **Valid Experiment Name**: Must end with `_yyyymmdd` format (e.g., `MyModel_20260130`)
4. **Consistent Configuration**: Use the same evaluation settings as official evaluations

## Submission Process

1. Run `genarena submit` - validates data, uploads to HF, creates PR
2. GitHub Actions bot automatically validates your submission
3. Maintainers review and approve/reject
4. Approved submissions are integrated into the official leaderboard

## Directory Structure

```
submissions/
├── pending/     # Submissions awaiting review
├── approved/    # Accepted submissions
└── rejected/    # Rejected submissions
```

## For Maintainers

After merging a submission PR, the **integration workflow** runs automatically:

1. Downloads data from the submitter's HuggingFace repo
2. Uploads to `rhli/genarena-battlefield` (archive format) and `genarena/leaderboard-data` (CDN format)
3. Rebuilds ELO state from all battle logs (Bradley-Terry, no API calls)
4. Updates `official_models.json` with new models
5. Moves the submission from `pending/` to `approved/`
6. Commits changes back to this repository

**Required secrets:** Set `HF_TOKEN` in the repository settings with write access to both HuggingFace repos.

### Manual Override

If the integration workflow fails, you can manually integrate:

```bash
# 1. Download submission data
genarena hf pull \
  --arena_dir /path/to/official-arena \
  --repo_id submitter/their-hf-repo \
  --experiments ExpName_20260130

# 2. Rebuild leaderboard and upload
genarena deploy upload \
  --arena_dir /path/to/official-arena \
  --arena_repo rhli/genarena-battlefield \
  --space_repo genarena/leaderboard

# 3. Move submission to approved/
mv submissions/pending/sub_xxx.json submissions/approved/

# 4. Update official_models.json
genarena export-models --arena_dir /path/to/official-arena -o official_models.json
```
