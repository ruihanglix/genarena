# GenArena Deploy Guide

This guide covers how to deploy GenArena Explorer to HuggingFace Spaces for public access.

## Overview

The `genarena deploy` command group handles deployment to HuggingFace Spaces. Data is organized across three repositories:

- **Space repo**: Contains Dockerfile, app code, and genarena package
- **Arena repo (Dataset)**: Contains battle logs and model output images (served via CDN)
- **Data repo (`rhli/genarena`)**: Parquet benchmark data (downloaded during Docker build)

```
┌─────────────────────────────────────────────────────────────┐
│                    HuggingFace Spaces                       │
├─────────────────────────────────────────────────────────────┤
│  Space Repo (your-org/genarena-explorer)                    │
│  ├── Dockerfile         # Downloads rhli/genarena at build  │
│  ├── README.md                                              │
│  ├── pyproject.toml                                         │
│  └── genarena/          # Python package                    │
│      └── deploy/app.py  # Startup script                    │
└─────────────────────────────────────────────────────────────┘
          │                              │
          │ HF_ARENA_REPO env var        │ Downloaded at build
          v                              v
┌─────────────────────────┐    ┌─────────────────────────────┐
│  Arena Repo (Dataset)   │    │  Data Repo (rhli/genarena)  │
│  your-org/genarena-arena│    │  (public benchmark data)    │
│  └── <subset>/          │    │  └── <subset>/              │
│      ├── pk_logs/       │    │      └── data-*.parquet     │
│      ├── models/  (CDN) │    └─────────────────────────────┘
│      └── arena/         │
└─────────────────────────┘
```

## Prerequisites

### 1. HuggingFace Account and Token

```bash
# Get a token from https://huggingface.co/settings/tokens
# Token needs write permission for both repos

export HF_TOKEN='your_token_here'
```

### 2. Create HuggingFace Repositories

Create two repositories on HuggingFace:

1. **Dataset repository** for arena data:
   - Go to https://huggingface.co/new-dataset
   - Name: e.g., `your-org/genarena-arena`

2. **Space repository** for deployment:
   - Go to https://huggingface.co/new-space
   - Name: e.g., `your-org/genarena-explorer`
   - **SDK: Select "Docker"** (important!)

## Commands

### `genarena deploy upload`

Upload arena data and deploy files to HuggingFace Spaces.

```bash
genarena deploy upload \
  --arena_dir ./arena \
  --arena_repo your-org/genarena-arena \
  --space_repo your-org/genarena-explorer
```

#### Required Arguments

| Argument | Description |
|----------|-------------|
| `--arena_dir` | Local arena directory containing battle logs and model outputs |
| `--arena_repo` | HuggingFace Dataset repo for arena data (e.g., `your-org/genarena-arena`) |
| `--space_repo` | HuggingFace Space repo for deployment (e.g., `your-org/genarena-explorer`) |

#### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--subsets` | all | Comma-separated list of subsets to upload |
| `--overwrite` | false | Overwrite existing files (default: incremental upload) |
| `--max-retries` | 3 | Maximum retry attempts per file on connection failure |
| `--verbose` | false | Enable verbose output |

#### What Gets Uploaded

1. **Arena data** (to Dataset repo):
   - `pk_logs/` - Battle log files (`.jsonl`)
   - `models/` - Model output images as **individual files** (`.png`, `.jpg`, etc.)
   - `arena/state.json` - ELO state

2. **Deploy files** (to Space repo):
   - `Dockerfile` - Docker build configuration (includes `rhli/genarena` download)
   - `README.md` - Space configuration (title, emoji, SDK settings)
   - `genarena/` - Python package files
   - `pyproject.toml` - Package configuration

**Note**: Parquet benchmark data is NOT uploaded. It is automatically downloaded from `rhli/genarena` during the Docker build process.

> **Important**: Unlike `genarena hf upload` which packs images into ZIP files, `genarena deploy upload` uploads images as individual files. This is required for HuggingFace CDN to serve images directly to the web interface.

### `genarena deploy info`

Show deployment configuration and instructions.

```bash
genarena deploy info
```

This displays a comprehensive guide including:
- Prerequisites
- Usage examples
- Directory structure
- Environment variables

## Deployment Workflow

### Step 1: Prepare Local Data

Ensure your local arena directory is ready:

```bash
# Check arena directory structure
ls -la ./arena/basic/
# Should contain: models/, pk_logs/, arena/
```

### Step 2: Upload Data

```bash
genarena deploy upload \
  --arena_dir ./arena \
  --arena_repo your-org/genarena-arena \
  --space_repo your-org/genarena-explorer
```

Example output:

```
============================================================
  GenArena HuggingFace Spaces Deployment
============================================================
  Arena dir:   ./arena
  Arena repo:  your-org/genarena-arena
  Space repo:  your-org/genarena-explorer
  Mode:        incremental
  Max retries: 3

  Note: Parquet data is downloaded from rhli/genarena during build
============================================================

============================================================
  Deployment upload completed successfully!
============================================================

Arena data: Uploaded 150, skipped 0, failed 0 files
Deploy files: Uploaded 3, skipped 0, failed 0 deploy files. Uploaded 45 package files

Next steps:
  1. Go to https://huggingface.co/spaces/your-org/genarena-explorer
  2. Set environment variable: HF_ARENA_REPO=your-org/genarena-arena
  3. The Space should automatically rebuild and deploy
```

### Step 3: Configure Space Environment

1. Go to your Space: `https://huggingface.co/spaces/your-org/genarena-explorer`
2. Click **Settings**
3. Under **Repository secrets**, add:
   - Name: `HF_ARENA_REPO`
   - Value: `your-org/genarena-arena`

### Step 4: Verify Deployment

The Space will automatically rebuild after configuration. Once ready:

1. Visit your Space URL
2. Verify the leaderboard loads correctly
3. Check that battle records display with images

## Incremental Updates

By default, uploads are incremental - existing files are skipped. This allows you to:

- Resume interrupted uploads
- Add new battles without re-uploading everything
- Update specific subsets

```bash
# Upload only new files (default behavior)
genarena deploy upload \
  --arena_dir ./arena \
  --arena_repo your-org/genarena-arena \
  --space_repo your-org/genarena-explorer

# Force re-upload all files
genarena deploy upload \
  --arena_dir ./arena \
  --arena_repo your-org/genarena-arena \
  --space_repo your-org/genarena-explorer \
  --overwrite
```

## Uploading Specific Subsets

To upload only specific subsets:

```bash
genarena deploy upload \
  --arena_dir ./arena \
  --arena_repo your-org/genarena-arena \
  --space_repo your-org/genarena-explorer \
  --subsets basic,advanced
```

## Environment Variables

### For the CLI

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | HuggingFace API token (required for upload) |

### For the Deployed Space

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_ARENA_REPO` | `genarena/leaderboard-data` | HuggingFace Dataset repo for arena data |
| `HF_DATA_REPO` | `rhli/genarena` | HuggingFace Dataset repo for parquet benchmark data |

## How Parquet Data is Loaded

The parquet benchmark data (`rhli/genarena`) is downloaded automatically:

1. **During Docker build**: The Dockerfile runs `huggingface_hub.snapshot_download()` to download the dataset to `/app/data/`
2. **At runtime (fallback)**: If data is not present, `app.py` downloads it on startup

This approach:
- Keeps the Space repo small (no large parquet files)
- Uses the official benchmark dataset
- Allows easy updates by rebuilding the Space

## Troubleshooting

### Upload Fails with "Token not found"

```bash
# Ensure HF_TOKEN is set
export HF_TOKEN='your_token_here'

# Or login via CLI
huggingface-cli login
```

### Space Shows "Application Error"

1. Check Space logs in the HuggingFace UI
2. Verify `HF_ARENA_REPO` environment variable is set correctly
3. Ensure the Arena repo is accessible (public or token has read access)

### Images Not Loading

Images are served via HuggingFace CDN. Ensure:
- Images were uploaded to the Arena repo (not the Space repo)
- The Arena repo is public or the Space has access

### Parquet Data Download Fails

If the Space fails to download parquet data:

1. Check if `rhli/genarena` is accessible
2. Verify network connectivity in Space logs
3. Optionally set `HF_DATA_REPO` to use a different dataset

### Connection Errors During Upload

Use `--max-retries` to handle transient network issues:

```bash
genarena deploy upload \
  --arena_dir ./arena \
  --arena_repo your-org/genarena-arena \
  --space_repo your-org/genarena-explorer \
  --max-retries 5
```

## Local Development

For local testing before deployment:

```bash
# Download benchmark data locally (if not already present)
huggingface-cli download rhli/genarena --repo-type dataset --local-dir ./data

# Run local server
genarena serve --arena_dir ./arena --data_dir ./data --port 8080

# Open in browser
open http://localhost:8080
```

## See Also

- [CLI Reference](../cli-reference.md) - Full CLI documentation
- [Architecture](../architecture.md) - System architecture overview
- [Maintainer Guide](./README.md) - Guide for leaderboard maintainers
