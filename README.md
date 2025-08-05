# LLM-based Video Anomaly Detection

A video anomaly detection system that uses Google Gemini API to analyze video frames and detect anomalous or violent content in the XD-Violence dataset.

## Project Structure

```
LLMVAD/
├── main.py                 # Optimized entry point with accuracy reporting
├── utils.py                # Shared utilities and logging
├── prompts.py              # Gemini prompts for video analysis
├── pyproject.toml          # Project configuration and linting rules
├── .gitignore              # Git ignore rules (excludes cache/)
├── cache/                  # Model response cache (git-ignored)
├── datasets/               # XD-Violence dataset
└── requirements.txt        # Dependencies
```

## Features

- **Google Gemini API**: Uses Gemini 2.5 Flash for video understanding
- **Optimized Code**: Concise, linted, and efficient implementation
- **Accuracy Reporting**: Per-class accuracy, precision, and recall metrics
- **Caching System**: Automatic caching of model responses for faster re-runs
- **Modern Python**: Uses pathlib, type hints, and modern Python idioms
- **XD-Violence Dataset**: Specialized for violence detection in videos

## Usage

### Basic Usage

```bash
# Run video processing and accuracy analysis
python main.py

# Run accuracy report only (uses cached results)
python main.py --accuracy_report
```

### Video Processing

The system processes videos from the XD-Violence dataset and:
1. Extracts video frames
2. Sends to Gemini API for analysis
3. Caches responses for efficiency
4. Calculates accuracy metrics

## Setup

### Prerequisites
```bash
# Install uv (recommended package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Google Cloud Configuration
1. Set up Google Cloud credentials in `GeminiAPI/gcp_credentials.json`
2. Configure project settings in `main.py`:
   ```python
   project="devtest-autopilot"
   location="us-central1"
   ```

## Accuracy Reporting

The system provides detailed accuracy metrics for the XD-Violence dataset:

```
Class   Accuracy        Precision       Recall
----------------------------------------
B1      0.880           0.450           0.900
B2      0.920           0.590           1.000
B4      0.995           0.750           1.000
B5      0.940           0.143           1.000
B6      0.990           0.833           0.833
G       0.955           0.591           1.000
A       0.860           1.000           0.810

Overall Accuracy: 0.934
```

### Event Types
- **B1**: Fighting
- **B2**: Shooting  
- **B4**: Riot
- **B5**: Abuse
- **B6**: Car accident
- **G**: Explosion
- **A**: Not Anomalous

## Workflow

1. **Video Processing**: Extract frames from XD-Violence dataset videos
2. **Gemini Analysis**: Send video data to Gemini 2.5 Flash API
3. **JSON Response**: Parse structured JSON responses with anomaly scores
4. **Caching**: Store results for faster subsequent runs
5. **Accuracy Analysis**: Calculate per-class metrics from cached results

## Cache Management

- **Automatic Caching**: Model responses are cached in `cache/` directory
- **Git Ignored**: Cache files are excluded from version control
- **Regeneratable**: Cache can be cleared and regenerated as needed
- **Hash-based**: Cache keys include prompt hash for version control

## Code Quality

- **Linted**: Uses `ruff` for code quality and formatting
- **Type Hints**: Modern Python type annotations
- **Optimized**: Concise, efficient implementation
- **Documented**: Clear docstrings and comments

## Output Files

- `cache/*.joblib`: Cached Gemini API responses
- Accuracy reports printed to console
- Structured JSON responses with anomaly scores

## Development

```bash
# Format and lint code
uv run ruff check --fix
uv run ruff format

# Run tests
uv run python -m pytest

# Install development dependencies
uv sync --dev
```

## Model Configuration

The system uses Gemini 2.5 Flash with a specialized prompt for anomaly detection:

- **Model**: `gemini-2.5-flash`
- **Scoring System**: 0-1 scale with detailed reasoning
- **Output Format**: Structured JSON with tag_id, score, and reasoning
- **Threshold**: >0.7 for confident anomaly detection

## Dataset

Uses the XD-Violence dataset with video files named in format:
`MovieName__timestamp_label_EventType.mp4`

Example: `A.Beautiful.Mind.2001__00-01-45_00-02-50_label_A.mp4`





