# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""HuggingFace Spaces startup script for GenArena Explorer."""

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
HF_ARENA_REPO = os.environ.get("HF_ARENA_REPO", "genarena/leaderboard-data")
HF_DATA_REPO = os.environ.get("HF_DATA_REPO", "rhli/genarena")


def main():
    """Main entry point for HuggingFace Spaces."""
    from huggingface_hub import list_repo_files, snapshot_download

    logger.info("=" * 60)
    logger.info("  GenArena Explorer - HuggingFace Spaces")
    logger.info("=" * 60)
    logger.info(f"  Arena Repo: {HF_ARENA_REPO}")
    logger.info(f"  Data Repo:  {HF_DATA_REPO}")
    logger.info("=" * 60)

    # 1. Parquet data - check if already downloaded during Docker build
    data_dir = "/app/data"
    if not os.path.isdir(data_dir) or not os.listdir(data_dir):
        # Download parquet data at runtime if not present
        logger.info(f"Downloading parquet data from {HF_DATA_REPO}...")
        data_dir = snapshot_download(
            HF_DATA_REPO,
            repo_type="dataset",
            local_dir="/app/data",
        )
        logger.info(f"Parquet data downloaded to: {data_dir}")
    else:
        logger.info(f"Using pre-downloaded parquet data: {data_dir}")

    # 2. Download arena metadata (excluding images)
    logger.info(f"Downloading arena metadata from {HF_ARENA_REPO}...")
    arena_dir = snapshot_download(
        HF_ARENA_REPO,
        repo_type="dataset",
        local_dir="/app/arena",
        ignore_patterns=["*.png", "*.jpg", "*.jpeg", "*.webp"],
    )
    logger.info(f"Arena metadata downloaded to: {arena_dir}")

    # 3. Get image file list for CDN URL mapping
    logger.info("Scanning image files for CDN URL mapping...")
    all_files = list_repo_files(HF_ARENA_REPO, repo_type="dataset")
    image_files = [
        f for f in all_files if f.endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    logger.info(f"Found {len(image_files)} image files")

    # 4. Start Flask server
    logger.info("Starting Flask server...")
    from genarena.visualize.app import create_hf_app

    app = create_hf_app(
        arena_dir=arena_dir,
        data_dir=data_dir,
        hf_repo=HF_ARENA_REPO,
        image_files=image_files,
    )

    logger.info("=" * 60)
    logger.info("  Server ready: http://0.0.0.0:7860")
    logger.info("=" * 60)

    app.run(host="0.0.0.0", port=7860, threaded=True)


if __name__ == "__main__":
    main()
