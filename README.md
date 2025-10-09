# LLM-based Video Anomaly Detection

A video anomaly detection system that uses Google Gemini API to analyze video frames and detect anomalous or violent content in the XD-Violence dataset.

## Project Structure

```
LLMVAD/
├── main.py                      # Entry point with evaluation modes
├── eval.py                      # Video processing and evaluation logic
├── evaluation_stats.py          # Accuracy metrics and PR curves
├── llm_handler.py              # LLM API interaction
├── anomaly_judge.py            # Anomaly detection logic
├── utils.py                    # Shared utilities and logging
├── prompts.py                  # Gemini prompts for video analysis
├── pyproject.toml              # Project configuration and dependencies
├── uv.lock                     # Locked dependencies (uv)
├── .env                        # Environment variables (Kaggle credentials)
├── scripts/                    # Dataset download scripts
│   ├── download_data.py        # Python wrapper for dataset downloads
│   ├── download_dataset.ps1    # PowerShell download script
│   ├── datasets_config.yaml    # Dataset configuration
│   └── README.md               # Scripts documentation
├── cache/                      # Model response and results cache
│   ├── {eval_mode}/            # Cached model responses per eval mode
│   └── results/                # Cached metrics and plots
│       ├── *_raw.joblib        # Raw predictions cache
│       ├── *_raw.yaml          # Raw predictions metadata
│       ├── *_threshold_*.joblib # Accuracy reports cache
│       ├── *_threshold_*.yaml   # Accuracy reports metadata
│       └── *_pr_curves.png      # Precision-Recall curve plots
└── datasets/                   # Downloaded datasets
    └── XD_Violence_1-1004/     # XD-Violence dataset
```

## Features

- **Google Gemini API**: Uses Gemini 2.0 Flash for video understanding
- **Multiple Evaluation Modes**: Video-based and ontological category analysis
- **Comprehensive Metrics**: Accuracy, precision, recall, and Precision-Recall curves
- **Intelligent Caching**: Multi-layer caching with YAML metadata sidecars
- **Dataset Management**: Automated dataset download via Kaggle API
- **Modern Python**: Uses uv, pathlib, type hints, and modern Python idioms
- **XD-Violence Dataset**: Specialized for violence detection in videos

## Usage

### Download Dataset

```bash
# Download XD-Violence dataset from Kaggle
python scripts/download_data.py xd-violence

# Force re-download
python scripts/download_data.py xd-violence --force
```

### Run Evaluation

```bash
# Run video processing with specific evaluation mode
python main.py --eval_mode ontological_categories

# Generate accuracy reports and PR curves
python evaluation_stats.py
```

### Evaluation Modes

- **`video_simple`**: Basic video-level anomaly detection
- **`ontological_categories`**: Category-specific anomaly classification

## Setup

### Prerequisites

```bash
# Install uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Configuration

1. **Kaggle API Credentials**: Create `.env` file with:

   ```bash
   KAGGLE_USERNAME=your_username
   KAGGLE_KEY=your_api_key
   ```

2. **Google Gemini API**: Configure API key in your environment or code

## Metrics and Evaluation

### Available Metrics

The system provides comprehensive evaluation metrics:

- **Accuracy, Precision, Recall**: Per-class and overall metrics
- **Precision-Recall Curves**: Visualized with Average Precision (AP) scores
- **Mean Average Precision (mAP)**: Overall model performance metric
- **YAML Metadata**: Detailed statistics about predictions and ground truth

### Event Types

- **B1**: Fighting
- **B2**: Shooting
- **B4**: Riot
- **B5**: Abuse
- **B6**: Car accident
- **G**: Explosion
- **A**: Not Anomalous (Normal content)

## Workflow

1. **Dataset Download**: Automated download from Kaggle using aria2c or fallback
2. **Video Processing**: Process videos from XD-Violence dataset
3. **LLM Analysis**: Send video data to Google Gemini API
4. **Response Caching**: Cache model responses with metadata
5. **Metrics Calculation**: Generate accuracy reports and PR curves
6. **Results Caching**: Cache computed metrics with YAML sidecars

## Cache Management

The system uses a sophisticated multi-layer caching system:

- **Model Responses**: Cached per evaluation mode in `cache/{eval_mode}/`
- **Raw Predictions**: Cached with statistics in `cache/results/*_raw.joblib`
- **Accuracy Reports**: Cached per threshold in `cache/results/*_threshold_*.joblib`
- **YAML Metadata**: Human-readable sidecars with dataset statistics
- **PR Curve Plots**: Saved as PNG files per evaluation mode
- **Git Ignored**: All cache files are excluded from version control

## Code Quality

- **Linting**: Uses `ruff` for code quality and formatting
- **Type Checking**: Uses `pyright` for static type analysis
- **Type Hints**: Comprehensive Python type annotations
- **Modern Python**: Uses latest Python features and idioms
- **Documented**: Clear docstrings and inline comments
- **Package Management**: Uses `uv` for fast, reliable dependency management

## Output Files

### Cache Directory Structure

- `cache/{eval_mode}/*.joblib`: Cached model responses per video
- `cache/results/*_raw.joblib`: Raw prediction scores
- `cache/results/*_raw.yaml`: Raw prediction metadata
- `cache/results/*_threshold_*.joblib`: Accuracy reports
- `cache/results/*_threshold_*.yaml`: Accuracy report metadata
- `cache/results/*_pr_curves.png`: Precision-Recall curve visualizations

### YAML Metadata

Each cached result includes a YAML sidecar with:

- Timestamp of creation
- Evaluation mode used
- Total number of samples
- Ground truth label distribution (count and percentage)
- Prediction statistics (for threshold-based caches)
- Binary prediction counts (for accuracy reports)

## Development

```bash
# Format and lint code
uv run ruff check --fix
uv run ruff format

# Type checking
uv run pyright

# Install development dependencies
uv sync --dev

# Download dataset
python scripts/download_data.py xd-violence
```

## Model Configuration

The system uses Google Gemini with specialized prompts for anomaly detection:

- **Model**: Google Gemini (via `google-genai` library)
- **Scoring System**: 0-1 scale with detailed reasoning
- **Output Format**: Structured JSON with tag_id, score, and reasoning
- **Default Threshold**: 0.7 for confident anomaly detection
- **Evaluation Modes**: Support for different analysis strategies

## Dataset

Uses the XD-Violence dataset with video files named in format:
`MovieName__timestamp_label_EventType.mp4`

Example: `A.Beautiful.Mind.2001__00-01-45_00-02-50_label_A.mp4`
