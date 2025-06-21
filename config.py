# config.py

# Default configuration for video processing and anomaly detection
DEFAULT_CONFIG = {
    "video_dir": "datasets/XD_Violence_1-1004",
    "frame_interval": 10,
    "summarization_chunk_size": 3, # Add this for video_processor
    "ollama_url": "http://localhost:11434/api/chat",
    "ollama_model": "gemma3:4b-it-q4_K_M",
    "timeout": 30,
}

