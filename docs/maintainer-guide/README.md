# GenArena Maintainer Guide

This guide is for maintainers of the official GenArena leaderboard. It covers how to manage community submissions, update the leaderboard, and maintain the related infrastructure.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Daily Maintenance](#daily-maintenance)
4. [Processing Community Submissions](#processing-community-submissions)
5. [Updating the Official Leaderboard](#updating-the-official-leaderboard)
6. [Troubleshooting](#troubleshooting)
7. [Deployment](./deploy.md) - Deploy GenArena Explorer to HuggingFace Spaces

---

## Architecture Overview

The GenArena community submission system consists of three components:

```
┌─────────────────────────┐
│   genarena/submissions  │  GitHub repo - receives community PRs
│   (submissions repo)    │  stores submission metadata
└───────────┬─────────────┘
            │
            │ Validation Bot (GitHub Actions)
            │ downloads data, validates, replies to PR
            v
┌─────────────────────────┐
│   User's HuggingFace    │  User's own HF Dataset repo
│   Dataset Repository    │  stores actual battle logs and images
└───────────┬─────────────┘
            │
            │ Maintainer manually pulls
            v
┌─────────────────────────┐
│   Official Arena Data   │  Official evaluation data storage
│   (HF + local)          │  contains all approved data
└─────────────────────────┘
```

### Key Repositories

| Repository | Purpose | URL |
|------------|---------|-----|
| `genarena/submissions` | Receives community submission PRs | https://github.com/genarena/submissions |
| `rhli/genarena-battlefield` | Official evaluation data (HF) | https://huggingface.co/datasets/rhli/genarena-battlefield |
| `genarena/genarena` | Code repository | https://github.com/genarena/genarena |

---

## Infrastructure Setup

### 1. Initialize the Submissions Repository

If you need to create the submissions repo from scratch:

```bash
# Use the template from the project
cp -r submissions-repo-template/* /path/to/submissions-repo/

# Or create from GitHub
gh repo create genarena/submissions --public
cd submissions-repo
git init
# Copy template files...
git add .
git commit -m "Initial setup"
git push -u origin main
```

### 2. Configure GitHub Actions Secrets

In the `genarena/submissions` repo's Settings > Secrets, configure:

| Secret | Purpose | How to Get |
|--------|---------|------------|
| `HF_TOKEN` | Download user HF data for validation | https://huggingface.co/settings/tokens (needs read permission) |

### 3. Maintainer Local Environment

```bash
# Install genarena
pip install genarena

# Or install from source
git clone https://github.com/genarena/genarena.git
cd genarena
pip install -e .

# Login to HuggingFace (for uploading official data)
huggingface-cli login

# Login to GitHub CLI (for managing PRs)
gh auth login
```

### 4. Official Arena Directory Structure

Maintainers need to maintain a local copy of the official arena data:

```
/path/to/official-arena/
├── basic/                      # subset
│   ├── models/
│   │   └── <exp_name>/        # model outputs per experiment
│   │       └── <model>/
│   │           └── *.png
│   ├── pk_logs/
│   │   └── <exp_name>/        # battle logs per experiment
│   │       ├── *.jsonl
│   │       └── config.json
│   └── arena/
│       └── state.json         # current ELO state
│
└── advanced/                   # another subset
    └── ...
```

---

## Daily Maintenance

### Check Pending Submissions

```bash
# List all open PRs
gh pr list --repo genarena/submissions

# View details of a specific PR
gh pr view 42 --repo genarena/submissions
```

### Review Validation Bot Results

Each PR automatically triggers GitHub Actions validation. The bot replies with a validation report in the PR comments.

The validation report includes:
- Schema validation results
- Data download and checksum verification
- Battle count verification
- ELO recalculation comparison
- New model check

**Example Validation Report:**

```markdown
## Submission Validation Report

**Status:** ✅ All checks passed

### Summary
| Item | Value |
|------|-------|
| Submission ID | `sub_20260130T100000_a1b2c3d4` |
| Experiment | `NewModel_20260130` |
| Subset | `basic` |
| New Models | `NewModel` |
| Total Battles | 500 |

### Validation Checks
- ✅ JSON parse
- ✅ Schema validation
- ✅ Model 'NewModel' is new
- ✅ Download pk_logs
- ✅ SHA256 checksum
- ✅ Extract ZIP
- ✅ Parse battle logs
- ✅ Battle count
- ✅ ELO verification
```

---

## Processing Community Submissions

### Review Checklist

Before approving a PR, verify the following:

1. **Validation Bot Passed** - All automated checks are green
2. **Model Name Validity** - Model names are clear and represent real models
3. **Sufficient Battles** - Enough battles for meaningful ELO scores
4. **Consistent Evaluation Config** - Judge model and prompt match official settings
5. **Trustworthy Source** - Submitter is trustworthy, data looks reasonable

### Approve and Merge a Submission

```bash
# 1. Review the PR
gh pr view 42 --repo genarena/submissions

# 2. If everything looks good, approve the PR
gh pr review 42 --approve --repo genarena/submissions

# 3. Merge the PR
gh pr merge 42 --merge --repo genarena/submissions
```

### Pull Submission Data to Official Arena

```bash
# Get info from the submission JSON
cat submissions/pending/sub_20260130T100000_a1b2c3d4.json

# Pull the data
genarena hf pull \
  --arena_dir /path/to/official-arena \
  --repo_id submitter-username/their-hf-repo \
  --subsets basic \
  --experiments NewModel_20260130

# Verify data was pulled correctly
ls /path/to/official-arena/basic/models/NewModel_20260130/
ls /path/to/official-arena/basic/pk_logs/NewModel_20260130/
```

### Rebuild ELO State

```bash
# Method 1: Use genarena run to rebuild (recommended)
# --sample_size 0 means don't run new battles, just rebuild state
genarena run \
  --arena_dir /path/to/official-arena \
  --data_dir /path/to/prompt-data \
  --subset basic \
  --sample_size 0

# Method 2: Just view the state
genarena leaderboard \
  --arena_dir /path/to/official-arena \
  --subset basic
```

### Update official_models.json

```bash
# Generate new official_models.json
genarena export-models \
  --arena_dir /path/to/official-arena \
  --output /path/to/submissions-repo/official_models.json

# Review the update
cat /path/to/submissions-repo/official_models.json
```

### Complete the Submission Process

```bash
cd /path/to/submissions-repo

# Move submission file to approved
mv submissions/pending/sub_20260130T100000_a1b2c3d4.json submissions/approved/

# Commit changes
git add .
git commit -m "Approved: NewModel_20260130 - added to official leaderboard"
git push
```

### Sync to Official HuggingFace

```bash
# Upload updated official data
genarena hf upload \
  --arena_dir /path/to/official-arena \
  --repo_id rhli/genarena-battlefield \
  --subsets basic \
  --experiments NewModel_20260130
```

---

## Updating the Official Leaderboard

### Complete Workflow Script

Here's a complete script for processing a submission:

```bash
#!/bin/bash
# process_submission.sh

SUBMISSION_ID="sub_20260130T100000_a1b2c3d4"
SUBMITTER_HF_REPO="submitter/their-repo"
EXP_NAME="NewModel_20260130"
SUBSET="basic"

OFFICIAL_ARENA="/path/to/official-arena"
SUBMISSIONS_REPO="/path/to/submissions-repo"
PROMPT_DATA="/path/to/prompt-data"

echo "=== Processing Submission: $SUBMISSION_ID ==="

# 1. Pull data
echo "Step 1: Pulling data from HuggingFace..."
genarena hf pull \
  --arena_dir "$OFFICIAL_ARENA" \
  --repo_id "$SUBMITTER_HF_REPO" \
  --subsets "$SUBSET" \
  --experiments "$EXP_NAME"

# 2. Rebuild ELO
echo "Step 2: Rebuilding ELO state..."
genarena run \
  --arena_dir "$OFFICIAL_ARENA" \
  --data_dir "$PROMPT_DATA" \
  --subset "$SUBSET" \
  --sample_size 0

# 3. Display updated leaderboard
echo "Step 3: Updated leaderboard:"
genarena leaderboard \
  --arena_dir "$OFFICIAL_ARENA" \
  --subset "$SUBSET"

# 4. Update official_models.json
echo "Step 4: Updating official_models.json..."
genarena export-models \
  --arena_dir "$OFFICIAL_ARENA" \
  --output "$SUBMISSIONS_REPO/official_models.json"

# 5. Move submission file
echo "Step 5: Moving submission to approved..."
mv "$SUBMISSIONS_REPO/submissions/pending/${SUBMISSION_ID}.json" \
   "$SUBMISSIONS_REPO/submissions/approved/"

# 6. Commit changes
echo "Step 6: Committing changes..."
cd "$SUBMISSIONS_REPO"
git add .
git commit -m "Approved: $EXP_NAME"
git push

# 7. Upload to official HF
echo "Step 7: Uploading to official HuggingFace..."
genarena hf upload \
  --arena_dir "$OFFICIAL_ARENA" \
  --repo_id rhli/genarena-battlefield \
  --subsets "$SUBSET" \
  --experiments "$EXP_NAME"

echo "=== Done! ==="
```

### Batch Processing Multiple Submissions

```bash
# List all pending submissions
ls submissions/pending/

# Process each submission
for f in submissions/pending/sub_*.json; do
  echo "Processing: $f"
  # Extract info and process...
done
```

---

## Troubleshooting

### Common Issues

#### 1. Validation Bot reports "SHA256 checksum mismatch"

**Cause:** Submitter modified the HF repo data after upload

**Solution:** Ask submitter to re-run `genarena submit`

#### 2. Validation Bot reports "Model already exists in official leaderboard"

**Cause:** Submitter is trying to submit an existing model

**Solution:**
- If it's the exact same model, reject the submission
- If it's a different version with the same name, ask submitter to use a different model name

#### 3. ELO Recalculation Mismatch

**Cause:** Submitter may have used a different genarena version or modified battle logs

**Solution:** Check submitter's genarena version, ask them to resubmit with the latest version

#### 4. Data Pull Failure

```bash
# Check if HF repo is accessible
huggingface-cli repo info submitter/their-repo --repo-type dataset

# Manual download to verify
huggingface-cli download submitter/their-repo basic/pk_logs/ExpName.zip --repo-type dataset
```

#### 5. GitHub Actions Failure

Check Actions logs:
```bash
gh run list --repo genarena/submissions
gh run view <run-id> --repo genarena/submissions --log
```

### Rejecting a Submission

If you need to reject a submission:

```bash
# 1. Comment on the PR explaining the rejection reason
gh pr comment 42 --repo genarena/submissions --body "Rejecting this submission because..."

# 2. Close the PR
gh pr close 42 --repo genarena/submissions

# 3. Move submission file to rejected (if already merged)
mv submissions/pending/sub_xxx.json submissions/rejected/
git add . && git commit -m "Rejected: reason" && git push
```

---

## Security Considerations

1. **Don't leak HF_TOKEN** - This token is used by GitHub Actions to download validation data
2. **Verify submitter identity** - For suspicious submissions, ask for more information
3. **Check data volume** - Unusually large or small data may indicate issues
4. **Backup official data** - Ensure you have backups before merging new data

---

## Contact

If you encounter issues you can't resolve:
- GitHub Issues: https://github.com/genarena/genarena/issues
- Other maintainers: @maintainer-list

---

## Appendix

### A. Command Quick Reference

| Command | Purpose |
|---------|---------|
| `genarena submit` | User submits evaluation results |
| `genarena export-models` | Generate official_models.json |
| `genarena hf pull` | Pull data from HF |
| `genarena hf upload` | Upload data to HF |
| `genarena leaderboard` | View leaderboard |
| `genarena status` | View arena status |
| `genarena run --sample_size 0` | Rebuild ELO state |
| `genarena deploy upload` | Deploy to HuggingFace Spaces |
| `genarena deploy info` | Show deployment instructions |

### B. File Path Reference

| Path | Description |
|------|-------------|
| `submissions/pending/*.json` | Pending submissions |
| `submissions/approved/*.json` | Approved submissions |
| `submissions/rejected/*.json` | Rejected submissions |
| `official_models.json` | Official models list |
| `<arena>/*/arena/state.json` | ELO state file |
| `<arena>/*/pk_logs/*/config.json` | Experiment config file |
