import logging
import os
import pickle
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler


def setup_logging(log_folder: str = "logs", log_level: str = "INFO"):
    """
    Configures a global logger to print timestamped messages to both console and file.

    Args:
        log_folder: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    os.makedirs(log_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_folder, f"anomaly_detection_{timestamp}.log")

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            ),
        ],
    )
    logging.info("Logging configured.")


def _get_summaries(video_dir: str, summaries_filename: str) -> list:
    """
    Load pickled summaries from a file.

    Args:
        video_dir: Directory containing the summaries file
        summaries_filename: Name of the summaries file

    Returns:
        List of summaries or empty list if file not found
    """
    summaries_path = os.path.join(video_dir, summaries_filename)
    if os.path.exists(summaries_path):
        try:
            with open(summaries_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading summaries from {summaries_path}: {e}")
    return []
