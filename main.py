import argparse
import logging
import os
import pickle

from utils import setup_logging, _get_summaries
from gemma_client import HuggingFaceGemmaClient
from video_processor import VideoProcessor
from anomaly_detector import AnomalyDetector
from plotting import plot_anomaly_scores
from config import DEFAULT_CONFIG
from hf_auth import setup_huggingface_auth


def main():
    """
    Main function to parse arguments, initialize classes, and start video processing.
    """
    setup_logging()
    logging.info("Application started.")
    
    setup_huggingface_auth()

    parser = argparse.ArgumentParser(
        description="Process video frames with Gemma 3 for multimodal summarization."
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_CONFIG["video_dir"],
        help="Directory containing the video file to process.",
    )
    parser.add_argument(
        "-K",
        type=int,
        default=DEFAULT_CONFIG["frame_interval"],
        help="Process every Kth frame (e.g., K=10 means every 10th frame).",
    )
    parser.add_argument(
        "--summarization_chunk_size",
        type=int,
        default=DEFAULT_CONFIG["summarization_chunk_size"],
        help="The number of frame descriptions to group into one chunk for higher-level summarization.",
    )
    parser.add_argument(
        "--gemma_model",
        type=str,
        default=DEFAULT_CONFIG["gemma_model"],
        help="Name of the HuggingFace Gemma 3 model to use for summarization.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Device to run the model on ('auto', 'cuda', 'cpu').",
    )

    args = parser.parse_args()

    logging.info("Parsed command-line arguments.")

    gemma_client = HuggingFaceGemmaClient(
        model_name=args.gemma_model, device=args.device
    )

    video_processor = VideoProcessor(
        video_directory=args.video_dir,
        model_client=gemma_client,
        frame_interval=args.K,
        summarization_chunk_size=args.summarization_chunk_size,
    )

    video_processor.process_video()

    logging.info("Application finished.")


def anomaly_detection(args: argparse.Namespace):
    """
    Performs anomaly detection using the AnomalyDetector, plots the scores,
    and saves the raw anomaly scores locally.
    """
    setup_logging()
    logging.info("Starting anomaly detection.")
    
    setup_huggingface_auth()

    anomaly_detector = AnomalyDetector(model_name=args.gemma_model, device=args.device)

    video_files = sorted(
        [f for f in os.listdir(args.video_dir) if f.lower().endswith(".mp4")]
    )
    if not video_files:
        logging.error(
            f"No video files found in {args.video_dir} for anomaly detection."
        )
        return

    video_file = video_files[0]
    file_name_without_ext = os.path.splitext(video_file)[0]
    summaries_filename = f"{file_name_without_ext}_chunked_summaries.pkl"

    logging.info(f"Loading summaries for anomaly detection from: {summaries_filename}")
    video_descriptions = _get_summaries(args.video_dir, summaries_filename)

    if not video_descriptions:
        logging.warning(
            f"No summaries loaded for '{video_file}', skipping anomaly detection for this video."
        )
        return

    logging.info(f"Running anomaly judgment for video: {video_file}")
    anomaly_scores = anomaly_detector.judge_video(video_descriptions)

    if anomaly_scores:
        scores_pickle_filename = f"{file_name_without_ext}_anomaly_scores.pkl"
        scores_pickle_path = os.path.join(args.video_dir, scores_pickle_filename)
        try:
            with open(scores_pickle_path, "wb") as f:
                pickle.dump(anomaly_scores, f)
            logging.info(
                f"Anomaly scores for '{video_file}' pickled to: {scores_pickle_path}"
            )
        except Exception as e:
            logging.error(f"Error pickling anomaly scores for '{video_file}': {e}")

        logging.info(f"Plotting anomaly scores for {video_file}...")
        plot_anomaly_scores(
            anomaly_scores, video_filename=video_file, output_dir=args.video_dir
        )
    else:
        logging.warning(
            f"No anomaly scores generated for {video_file}, skipping plot and saving."
        )

    logging.info(
        "Anomaly detection, plotting, and saving finished for the first relevant video."
    )


if __name__ == "__main__":
    parser_anomaly = argparse.ArgumentParser(
        description="Run anomaly detection and plot scores."
    )
    parser_anomaly.add_argument(
        "--video_dir",
        type=str,
        default=DEFAULT_CONFIG["video_dir"],
        help="Directory containing the video files and their pickled summaries.",
    )
    parser_anomaly.add_argument(
        "--gemma_model",
        type=str,
        default=DEFAULT_CONFIG["gemma_model"],
        help="Name of the HuggingFace Gemma 3 model to use (used by AnomalyDetector).",
    )
    parser_anomaly.add_argument(
        "--device",
        type=str,
        default=DEFAULT_CONFIG["device"],
        help="Device to run the model on ('auto', 'cuda', 'cpu') (used by AnomalyDetector).",
    )

    args_anomaly = parser_anomaly.parse_args()

    anomaly_detection(
        args_anomaly
    )
