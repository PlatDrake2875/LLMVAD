# Gemini API

This folder contains the Google Gemini-based implementation of the video anomaly detection system.

## Files

- `gemini_client.py` - Google Gemini client for text generation
- `video_processor.py` - Video processing and frame analysis using Gemini
- `anomaly_detector.py` - Anomaly detection using Gemini
- `config.py` - Configuration settings
- `gcp_credentials.json` - Google Cloud service account credentials
- `logs/` - Log files

Note: Video datasets are stored in the parent directory's `datasets/` folder.

## Usage

Run from the parent directory:

```bash
python main.py --api gemini --video_dir "path/to/videos" --K 10 --summarization_chunk_size 3
```

## Dependencies

- google-generativeai
- google-auth
- opencv-python
- pillow

## Setup

1. Ensure `gcp_credentials.json` contains valid Google Cloud service account credentials
2. Set the correct `project_id` in `config.py`
3. Install dependencies: `pip install google-generativeai google-auth opencv-python pillow` 