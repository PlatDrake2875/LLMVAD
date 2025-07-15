import os
import logging
import pickle
import sys


def setup_logging():
    """
    Configures a global logger to print timestamped messages to the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("Logging configured.")


def _get_summaries(
    video_dir: str, video_path: str
) -> list:
    """
    Loads pickled summaries from a specified file.

    Args:
        video_dir (str): The directory where the pickled file is located.
        video_path (str): The filename of the pickled summaries (e.g., "video_name_chunked_summaries.pkl").

    Returns:
        list: A list of loaded summaries.
    """
    full_path = os.path.join(video_dir, video_path)
    summaries = []

    if not os.path.exists(full_path):
        logging.error(f"Pickle file not found at: {full_path}")
        return []

    try:
        with open(full_path, "rb") as openfile:
            summaries = pickle.load(openfile)
        logging.info(f"Successfully loaded summaries from: {full_path}")
    except Exception as e:
        logging.error(f"Error loading summaries from {full_path}: {e}")
        summaries = []

    return summaries
