---
title: GenArena Leaderboard
emoji: ⚔️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# GenArena Leaderboard

Interactive visualization for GenArena image generation model evaluations.

## Features

- Browse battle records across multiple subsets
- View model outputs side-by-side with input images
- ELO leaderboard and win rate matrix
- Search and filter battles by model, result, consistency
- Detailed VLM judge reasoning for each battle
- Head-to-head comparison between models

## Data Sources

- **Arena Data**: Loaded from `HF_ARENA_REPO` environment variable
- **Benchmark Data**: Stored in this Space under `data/`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_ARENA_REPO` | `genarena/leaderboard-data` | HuggingFace Dataset repo for arena data (battle logs, model outputs) |

## Deployment

This Space is deployed using Docker. The parquet benchmark data is stored directly in this repository under `data/`, while arena data (battle logs and model output images) is fetched from the configured `HF_ARENA_REPO`.

Model output images are served via HuggingFace CDN URLs for efficient delivery.

## Local Development

```bash
# Install genarena
pip install -e .

# Run local server
genarena serve --arena_dir ./arena --data_dir ./data --port 8080
```

## Links

- [GenArena GitHub](https://github.com/genarena/genarena)
- [Official Arena Data](https://huggingface.co/datasets/rhli/genarena-battlefield)
- [Benchmark Data](https://huggingface.co/datasets/rhli/genarena)
