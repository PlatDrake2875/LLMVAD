import argparse
import logging
import os
import pickle # Needed for loading and saving pickled files

# Import classes and utility functions from our modules
from utils import setup_logging, _get_summaries
from ollama_client import OllamaClient
from video_processor import VideoProcessor
from anomaly_detector import AnomalyDetector
from plotting import plot_anomaly_scores # Import the new plotting function
from config import DEFAULT_CONFIG

def main():
    """
    Main function to parse arguments, initialize classes, and start video processing.
    """
    # Setup logging first
    setup_logging()
    logging.info("Application started.")

    parser = argparse.ArgumentParser(description="Process video frames with Ollama for summarization.")
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_CONFIG["video_dir"],
        help="Directory containing the video file to process."
    )
    parser.add_argument(
        "-K",
        type=int,
        default=DEFAULT_CONFIG["frame_interval"],
        help="Process every Kth frame (e.g., K=10 means every 10th frame)."
    )
    parser.add_argument(
        "--summarization_chunk_size",
        type=int,
        default=DEFAULT_CONFIG["summarization_chunk_size"],
        help="The number of frame descriptions to group into one chunk for higher-level summarization."
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default=DEFAULT_CONFIG["ollama_url"],
        help="URL for the Ollama chat API endpoint."
    )
    parser.add_argument(
        "--ollama_model",
        type=str,
        default=DEFAULT_CONFIG["ollama_model"],
        help="Name of the Ollama model to use for summarization."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONFIG["timeout"],
        help="Timeout for Ollama API requests in seconds."
    )

    args = parser.parse_args()

    logging.info("Parsed command-line arguments.")
    
    ollama_client = OllamaClient(
        api_url=args.ollama_url,
        model_name=args.ollama_model,
        timeout=args.timeout
    )
    
    video_processor = VideoProcessor(
        video_directory=args.video_dir,
        ollama_client=ollama_client,
        frame_interval=args.K,
        summarization_chunk_size=args.summarization_chunk_size
    )
    
    video_processor.process_video()
    
    logging.info("Application finished.")

def anomaly_detection(args: argparse.Namespace):
    """
    Performs anomaly detection using the AnomalyDetector, plots the scores,
    and saves the raw anomaly scores locally.
    """
    setup_logging() # Ensure logging is set up for anomaly_detection if called directly
    logging.info("Starting anomaly detection.")
    
    anomaly_detector = AnomalyDetector(
        api_url=args.ollama_url,
        model_name=args.ollama_model,
        timeout=args.timeout
    )

    video_files = sorted([f for f in os.listdir(args.video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    if not video_files:
        logging.error(f"No video files found in {args.video_dir} for anomaly detection.")
        return

    # Process only the first video file as requested
    video_file = video_files[0] 
    file_name_without_ext = os.path.splitext(video_file)[0]
    summaries_filename = f"{file_name_without_ext}_chunked_summaries.pkl"
    
    logging.info(f"Loading summaries for anomaly detection from: {summaries_filename}")
    video_descriptions = _get_summaries(args.video_dir, summaries_filename)

    if not video_descriptions:
        logging.warning(f"No summaries loaded for '{video_file}', skipping anomaly detection for this video.")
        return # Break here since we are only processing the first video

    logging.info(f"Running anomaly judgment for video: {video_file}")
    anomaly_scores = anomaly_detector.judge_video(video_descriptions)
    
    if anomaly_scores:
        # --- Save anomaly scores locally ---
        scores_pickle_filename = f"{file_name_without_ext}_anomaly_scores.pkl"
        scores_pickle_path = os.path.join(args.video_dir, scores_pickle_filename)
        try:
            with open(scores_pickle_path, 'wb') as f:
                pickle.dump(anomaly_scores, f)
            logging.info(f"Anomaly scores for '{video_file}' pickled to: {scores_pickle_path}")
        except Exception as e:
            logging.error(f"Error pickling anomaly scores for '{video_file}': {e}")
        # --- End save anomaly scores locally ---

        logging.info(f"Plotting anomaly scores for {video_file}...")
        plot_anomaly_scores(anomaly_scores, video_filename=video_file, output_dir=args.video_dir)
    else:
        logging.warning(f"No anomaly scores generated for {video_file}, skipping plot and saving.")

    logging.info("Anomaly detection, plotting, and saving finished for the first relevant video.")


if __name__ == "__main__":
    parser_anomaly = argparse.ArgumentParser(description="Run anomaly detection and plot scores.")
    parser_anomaly.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_CONFIG["video_dir"],
        help="Directory containing the video files and their pickled summaries."
    )
    parser_anomaly.add_argument(
        "--ollama_url",
        type=str,
        default=DEFAULT_CONFIG["ollama_url"],
        help="URL for the Ollama chat API endpoint (used by AnomalyDetector)."
    )
    parser_anomaly.add_argument(
        "--ollama_model",
        type=str,
        default=DEFAULT_CONFIG["ollama_model"],
        help="Name of the Ollama model to use (used by AnomalyDetector)."
    )
    parser_anomaly.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONFIG["timeout"],
        help="Timeout for Ollama API requests in seconds (used by AnomalyDetector)."
    )

    args_anomaly = parser_anomaly.parse_args()

    # Uncomment one of the following to run:
    # main() # Runs video processing, summarization, and saves pickles
    anomaly_detection(args_anomaly) # Loads pickles and runs anomaly detection + plotting + saving scores
