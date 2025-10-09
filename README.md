# LLM-based Video Anomaly Detection

A video anomaly detection system that uses Google Gemini API to analyze video frames and detect anomalous or violent content in the XD-Violence dataset.

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Dataset](#dataset)
- [Development](#development)

## About

This project implements a video anomaly detection system specifically designed for the XD-Violence dataset. It leverages Google Gemini's vision capabilities to analyze video content and classify violent/anomalous events across multiple categories.

**Key Features:**

- Multiple evaluation modes (video-level and category-specific)
- Comprehensive metrics (Accuracy, Precision, Recall, PR curves, mAP)
- Intelligent multi-layer caching with YAML metadata
- Automated dataset download via Kaggle API
- Modern Python with type hints and fast package management (uv)

## Prerequisites

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

## Dataset

Download the XD-Violence dataset:

```bash
python scripts/download_data.py xd-violence
```

For more information about the dataset, visit: [XD-Violence Documentation](https://roc-ng.github.io/XD-Violence/)

## Development

```bash
# Run evaluation
uv run python main.py

# Download dataset
python scripts/download_data.py xd-violence

# Code quality
uv run ruff check --fix
uv run pyright
```
