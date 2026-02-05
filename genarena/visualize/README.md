# GenArena Arena Visualizer

A web-based visualization tool for browsing and analyzing battle records from GenArena Arena evaluations.

## Features

- **Multi-subset Support**: Select and switch between different subsets directly in the web interface
- **Paginated Browsing**: Efficiently browse large numbers of battle records with pagination
- **Flexible Filtering**:
  - Filter by model (view all battles involving a specific model)
  - Filter by result (wins/losses/ties for a selected model)
  - Filter by consistency (consistent vs inconsistent VLM judgments)
- **Detailed Battle View**: Click any battle card to see:
  - Full instruction text
  - Input image and both model outputs side-by-side
  - Complete VLM judge reasoning (original and swapped calls)
- **Dark Theme**: Modern dark UI designed for extended analysis sessions

## Installation

The visualizer requires Flask:

```bash
pip install flask
```

## Usage

### Command Line

Start the visualization server using the `genarena serve` command:

```bash
genarena serve \
  --arena_dir /path/to/arena \
  --data_dir /path/to/data \
  --port 8080 \
  --host 0.0.0.0
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--arena_dir` | Yes | - | Path to the arena directory containing subset folders with battle logs |
| `--data_dir` | Yes | - | Path to the data directory containing parquet files |
| `--host` | No | `0.0.0.0` | Host address to bind the server |
| `--port` | No | `8080` | Port number to listen on |
| `--debug` | No | `False` | Enable Flask debug mode |

### Example

```bash
genarena serve \
  --arena_dir /projects/genarena/arena \
  --data_dir /datasets/genarena/data \
  --port 8080
```

Then open `http://localhost:8080` in your browser.

## Web Interface

### Navigation

1. **Select Subset**: Use the dropdown in the header to choose a subset
2. **Select Experiment**: Choose an experiment from the dropdown (populated after subset selection)
3. **Browse Battles**: Scroll through the paginated battle cards

### Filtering

Use the sidebar filters to narrow down results:

- **Model Filter**: Show only battles involving a specific model
- **Result Filter**: When a model is selected, filter by wins/losses/ties
- **Consistency Filter**: Show only consistent or inconsistent judgments

### Battle Cards

Each card displays:
- Model names (winner highlighted in green, loser in red)
- Instruction text (truncated)
- Thumbnail images: input, model A output, model B output
- Result badges (Win/Loss/Tie, Consistent/Inconsistent)

Click a card to open the detail modal.

### Detail Modal

The detail view shows:
- Full instruction text
- Large images for comparison
- Complete VLM judge outputs for both calls (original order and swapped order)
- Parse results and winner determination

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `j` / `↓` | Next page |
| `k` / `↑` | Previous page |
| `Esc` | Close detail modal |

## API Endpoints

The visualizer exposes a REST API that can be used programmatically:

| Endpoint | Description |
|----------|-------------|
| `GET /api/subsets` | List available subsets |
| `GET /api/subsets/<subset>/info` | Get subset info (models, experiments) |
| `GET /api/subsets/<subset>/experiments/<exp>/battles` | Get paginated battles |
| `GET /api/subsets/<subset>/experiments/<exp>/battles/<id>` | Get battle detail |
| `GET /api/subsets/<subset>/stats` | Get statistics |
| `GET /images/<subset>/<model>/<index>` | Serve model output image |
| `GET /images/<subset>/input/<index>` | Serve input image |

### Query Parameters for `/battles`

| Parameter | Type | Description |
|-----------|------|-------------|
| `page` | int | Page number (1-indexed) |
| `page_size` | int | Records per page (default: 20) |
| `model` | string | Filter by model name |
| `result` | string | Filter by result: `wins`, `losses`, `ties` |
| `consistent` | string | Filter by consistency: `true`, `false` |

## Directory Structure

```
visualize/
├── __init__.py       # Package exports
├── app.py            # Flask application and routes
├── data_loader.py    # Data loading and querying logic
├── templates/
│   └── index.html    # Main page template
└── static/
    ├── style.css     # Dark theme styles
    └── app.js        # Frontend JavaScript
```

## Requirements

- Python 3.8+
- Flask
- GenArena arena with battle logs (pk_logs directory)
- Parquet dataset with evaluation data

