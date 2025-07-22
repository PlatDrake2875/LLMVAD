# HuggingFace API

This folder contains the HuggingFace-based implementation of the video anomaly detection system.

## Files

- `gemma_client.py` - HuggingFace Gemma client for text generation
- `video_processor.py` - Video processing and frame analysis
- `anomaly_detector.py` - Anomaly detection using Gemma
- `hf_auth.py` - HuggingFace authentication setup
- `config.py` - Configuration settings
- `logs/` - Log files
- `.hf_cache/` - HuggingFace model cache

Note: Video datasets are stored in the parent directory's `datasets/` folder.

## Usage

Run from the parent directory:

```bash
python main.py --api huggingface --video_dir "path/to/videos" --K 10 --summarization_chunk_size 3
```

## Dependencies

- transformers
- torch
- opencv-python
- pillow
- huggingface-hub 