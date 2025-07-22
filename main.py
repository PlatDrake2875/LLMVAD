import argparse
import logging
import os
import pickle

from GeminiAPI.anomaly_detector import AnomalyDetector as GeminiAnomalyDetector
from GeminiAPI.config import DEFAULT_CONFIG as GEMINI_DEFAULT_CONFIG
from GeminiAPI.gemini_client import GeminiClient
from GeminiAPI.video_processor import VideoProcessor as GeminiVideoProcessor
from HuggingFaceAPI.anomaly_detector import AnomalyDetector as HFAnomalyDetector
from HuggingFaceAPI.config import DEFAULT_CONFIG as HF_DEFAULT_CONFIG
from HuggingFaceAPI.gemma_client import HuggingFaceGemmaClient
from HuggingFaceAPI.hf_auth import setup_huggingface_auth
from HuggingFaceAPI.video_processor import VideoProcessor as HFVideoProcessor
from plotting import plot_anomaly_scores
from utils import _get_summaries, setup_logging


def main():
    """
    Main function to parse arguments, initialize classes, and start video processing.
    """
    parser = argparse.ArgumentParser(
        description="Process video frames with either HuggingFace Gemma or Google Gemini for multimodal summarization."
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["huggingface", "gemini"],
        default="huggingface",
        help="Choose API: 'huggingface' for Gemma or 'gemini' for Google Gemini.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        help="Directory containing the video file to process.",
    )
    parser.add_argument(
        "-K",
        type=int,
        help="Process every Kth frame (e.g., K=10 means every 10th frame).",
    )
    parser.add_argument(
        "--summarization_chunk_size",
        type=int,
        help="The number of frame descriptions to group into one chunk for higher-level summarization.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (Gemma model for HuggingFace, Gemini model for Gemini API).",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run the model on ('auto', 'cuda', 'cpu') - only for HuggingFace.",
    )
    parser.add_argument(
        "--project_id",
        type=str,
        help="Google Cloud project ID - only for Gemini API.",
    )

    args = parser.parse_args()

    if args.api == "huggingface":
        run_huggingface_workflow(args)
    else:
        run_gemini_workflow(args)


def run_huggingface_workflow(args):
    setup_logging(log_folder="HuggingFaceAPI/logs", log_level="INFO")
    logging.info("HuggingFace API Application started.")

    setup_huggingface_auth()

    video_dir = args.video_dir or "datasets/XD_Violence_1-1004"
    frame_interval = args.K or HF_DEFAULT_CONFIG["frame_interval"]
    summarization_chunk_size = (
        args.summarization_chunk_size or HF_DEFAULT_CONFIG["summarization_chunk_size"]
    )
    model = args.model or HF_DEFAULT_CONFIG["gemma_model"]
    device = args.device or HF_DEFAULT_CONFIG["device"]

    logging.info("Parsed HuggingFace command-line arguments.")

    gemma_client = HuggingFaceGemmaClient(model_name=model, device=device)

    video_processor = HFVideoProcessor(
        video_directory=video_dir,
        model_client=gemma_client,
        frame_interval=frame_interval,
        summarization_chunk_size=summarization_chunk_size,
    )

    video_processor.process_video()

    logging.info("HuggingFace Application finished.")


def run_gemini_workflow(args):
    setup_logging(log_folder="GeminiAPI/logs", log_level="INFO")
    logging.info("Gemini API Application started.")

    video_dir = args.video_dir or "datasets/XD_Violence_1-1004"
    frame_interval = args.K or GEMINI_DEFAULT_CONFIG["frame_interval"]
    summarization_chunk_size = (
        args.summarization_chunk_size
        or GEMINI_DEFAULT_CONFIG["summarization_chunk_size"]
    )
    model = args.model or GEMINI_DEFAULT_CONFIG["gemini_model"]
    project_id = args.project_id or GEMINI_DEFAULT_CONFIG["project_id"]

    logging.info("Parsed Gemini command-line arguments.")

    gemini_client = GeminiClient(model_name=model, project_id=project_id)

    video_processor = GeminiVideoProcessor(
        video_directory=video_dir,
        model_client=gemini_client,
        frame_interval=frame_interval,
        summarization_chunk_size=summarization_chunk_size,
    )

    video_processor.process_video()

    logging.info("Gemini Application finished.")


def anomaly_detection(args: argparse.Namespace):
    """
    Performs anomaly detection using the appropriate AnomalyDetector based on API choice.
    """
    if args.api == "huggingface":
        run_huggingface_anomaly_detection(args)
    else:
        run_gemini_anomaly_detection(args)


def run_huggingface_anomaly_detection(args):
    setup_logging(log_folder="HuggingFaceAPI/logs", log_level="INFO")
    logging.info("Starting HuggingFace anomaly detection.")

    setup_huggingface_auth()

    model = args.model or HF_DEFAULT_CONFIG["gemma_model"]
    device = args.device or HF_DEFAULT_CONFIG["device"]
    video_dir = args.video_dir or "datasets/XD_Violence_1-1004"

    anomaly_detector = HFAnomalyDetector(model_name=model, device=device)

    video_files = sorted(
        [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    )
    if not video_files:
        logging.error(f"No video files found in {video_dir} for anomaly detection.")
        return

    video_file = video_files[0]
    file_name_without_ext = os.path.splitext(video_file)[0]
    summaries_filename = f"{file_name_without_ext}_chunked_summaries.pkl"

    logging.info(f"Loading summaries for anomaly detection from: {summaries_filename}")
    video_descriptions = _get_summaries(video_dir, summaries_filename)

    if not video_descriptions:
        logging.warning(
            f"No summaries loaded for '{video_file}', skipping anomaly detection for this video."
        )
        return

    logging.info(f"Running anomaly judgment for video: {video_file}")
    anomaly_scores = anomaly_detector.judge_video(video_descriptions)

    if anomaly_scores:
        scores_pickle_filename = f"{file_name_without_ext}_anomaly_scores.pkl"
        scores_pickle_path = os.path.join(video_dir, scores_pickle_filename)
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
            anomaly_scores, video_filename=video_file, output_dir=video_dir
        )
    else:
        logging.warning(
            f"No anomaly scores generated for {video_file}, skipping plot and saving."
        )

    logging.info("HuggingFace anomaly detection finished.")


def run_gemini_anomaly_detection(args):
    setup_logging(log_folder="GeminiAPI/logs", log_level="INFO")
    logging.info("Starting Gemini anomaly detection.")

    model = args.model or GEMINI_DEFAULT_CONFIG["gemini_model"]
    project_id = args.project_id or GEMINI_DEFAULT_CONFIG["project_id"]
    video_dir = args.video_dir or "datasets/XD_Violence_1-1004"

    anomaly_detector = GeminiAnomalyDetector(model_name=model, project_id=project_id)

    video_files = sorted(
        [f for f in os.listdir(video_dir) if f.lower().endswith(".mp4")]
    )
    if not video_files:
        logging.error(f"No video files found in {video_dir} for anomaly detection.")
        return

    video_file = video_files[0]
    file_name_without_ext = os.path.splitext(video_file)[0]
    summaries_filename = f"{file_name_without_ext}_chunked_summaries.pkl"

    logging.info(f"Loading summaries for anomaly detection from: {summaries_filename}")
    video_descriptions = _get_summaries(video_dir, summaries_filename)

    if not video_descriptions:
        logging.warning(
            f"No summaries loaded for '{video_file}', skipping anomaly detection for this video."
        )
        return

    logging.info(f"Running anomaly judgment for video: {video_file}")
    anomaly_scores = anomaly_detector.judge_video(video_descriptions)

    if anomaly_scores:
        scores_pickle_filename = f"{file_name_without_ext}_anomaly_scores.pkl"
        scores_pickle_path = os.path.join(video_dir, scores_pickle_filename)
        try:
            with open(scores_pickle_path, "wb") as f:
                pickle.dump(anomaly_scores, f)
            logging.info(
                f"Anomaly scores for '{video_file}' pickled to: {scores_pickle_path}"
            )
        except Exception as e:
            logging.error(f"Error pickling anomaly scores for '{video_file}': {e}")

    logging.info("Gemini anomaly detection finished.")


if __name__ == "__main__":
    main()
