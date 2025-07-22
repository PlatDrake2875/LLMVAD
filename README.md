# LLM-based Video Anomaly Detection

A video anomaly detection system that uses Large Language Models (LLMs) to analyze video frames and detect anomalous or violent content. Supports both HuggingFace Gemma and Google Gemini APIs.

## Project Structure

```
LLMVAD/
├── main.py                 # Unified entry point for both APIs
├── HuggingFaceAPI/         # HuggingFace Gemma implementation
│   ├── gemma_client.py     # HuggingFace Gemma client
│   ├── video_processor.py  # Video processing with Gemma
│   ├── anomaly_detector.py # Anomaly detection with Gemma
│   ├── hf_auth.py         # HuggingFace authentication
│   ├── config.py          # HuggingFace configuration
│   └── logs/              # HuggingFace logs
├── GeminiAPI/             # Google Gemini implementation
│   ├── gemini_client.py   # Google Gemini client
│   ├── video_processor.py # Video processing with Gemini
│   ├── anomaly_detector.py # Anomaly detection with Gemini
│   ├── config.py          # Gemini configuration
│   ├── gcp_credentials.json # Google Cloud credentials
│   └── logs/              # Gemini logs
├── datasets/              # Shared video datasets
├── utils.py               # Shared utilities
├── plotting.py            # Plotting utilities
└── requirements.txt       # Dependencies
```

## Usage

### HuggingFace API (Gemma)

```bash
python main.py --api huggingface \
    --video_dir "path/to/videos" \
    --K 10 \
    --summarization_chunk_size 3 \
    --model "google/gemma-3-4b-it" \
    --device "auto"
```

### Gemini API

```bash
python main.py --api gemini \
    --video_dir "path/to/videos" \
    --K 10 \
    --summarization_chunk_size 3 \
    --model "gemini-2.5-flash" \
    --project_id "your-project-id"
```

## Arguments

- `--api`: Choose between "huggingface" or "gemini" (default: "huggingface")
- `--video_dir`: Directory containing video files
- `-K`: Process every Kth frame (default: 10)
- `--summarization_chunk_size`: Number of frames to group for summarization (default: 3)
- `--model`: Model name (varies by API)
- `--device`: Device for HuggingFace ("auto", "cuda", "cpu")
- `--project_id`: Google Cloud project ID for Gemini

## Setup

### HuggingFace API
1. Install dependencies: `pip install transformers torch opencv-python pillow huggingface-hub`
2. Set up HuggingFace authentication (see HuggingFaceAPI/hf_auth.py)

### Gemini API
1. Install dependencies: `pip install google-generativeai google-auth opencv-python pillow`
2. Ensure `GeminiAPI/gcp_credentials.json` contains valid Google Cloud credentials
3. Set correct `project_id` in `GeminiAPI/config.py`

## Workflow

1. **Video Processing**: Extract frames at specified intervals
2. **Frame Analysis**: Use LLM to describe each frame
3. **Chunk Summarization**: Group frame descriptions and create higher-level summaries
4. **Anomaly Detection**: Analyze summaries to detect anomalous content
5. **Output**: Save anomaly scores and generate plots (HuggingFace only)

## Output Files

- `*_chunked_summaries.pkl`: Frame summaries
- `*_anomaly_scores.pkl`: Anomaly detection scores
- `*_anomaly_plot.png`: Anomaly score visualization (HuggingFace only)
- Log files in respective `logs/` directories





