import concurrent.futures
import logging
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
from tqdm import tqdm

from anomaly_judge import AnomalyJudge
from enums import EvalModes
from llm_handler import LLMHandler
from prompts import Ontological_Detectives_Prompts, Ontological_Prompts, Simple_Prompts
from utils import read_dataset


def setup_inference_logger(judge_mode: EvalModes | str) -> logging.Logger:
    """Set up a logger for inference with both file and console handlers."""
    judge_mode_value = (
        judge_mode.value if isinstance(judge_mode, EvalModes) else judge_mode
    )

    project_root = Path(__file__).parent.absolute()
    log_dir = project_root / "logs" / judge_mode_value
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"inference_{timestamp}.log"

    logger = logging.getLogger(f"inference_{judge_mode_value}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Inference logging started for mode: %s", judge_mode_value)
    logger.info("Log file: %s", log_file)

    return logger


def create_datapoint_log(
    judge_mode: EvalModes | str,
    video_name: str,
    response_data: dict,
    error_trace: str | None = None,
) -> None:
    """Create a detailed log file for each processed data point."""
    judge_mode_value = (
        judge_mode.value if isinstance(judge_mode, EvalModes) else judge_mode
    )

    project_root = Path(__file__).parent.absolute()
    datapoint_log_dir = project_root / "logs" / judge_mode_value / "datapoints"
    datapoint_log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = datapoint_log_dir / f"{video_name}_{timestamp}.log"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Data Point Evaluation Log\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Video Name: {response_data.get('video_name', 'N/A')}\n")
        f.write(f"Video Path: {response_data.get('video_path', 'N/A')}\n")
        f.write(f"Timestamp: {response_data.get('timestamp', 'N/A')}\n")
        f.write(f"Model: {response_data.get('model', 'N/A')}\n")
        f.write(f"Inference Time: {response_data.get('inference_time', 0):.2f}s\n")
        f.write(f"Retries: {response_data.get('retries', 0)}\n")
        f.write(f"Judge Mode: {judge_mode_value}\n")
        f.write("\n" + "-" * 80 + "\n")
        f.write("System Instruction:\n")
        f.write("-" * 80 + "\n")
        f.write(response_data.get("system_instruction", "N/A") + "\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("Model Response:\n")
        f.write("-" * 80 + "\n")
        f.write(response_data.get("response_text", "N/A") + "\n")

        if error_trace:
            f.write("\n" + "-" * 80 + "\n")
            f.write("Error Trace:\n")
            f.write("-" * 80 + "\n")
            f.write(error_trace + "\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("End of Log\n")
        f.write("=" * 80 + "\n")


def process_single_video(args):
    (
        handler,
        video_path,
        video_name,
        cache_filepath,
        video_data,
        system_prompt,
        judge_mode,
        user_prompt,
        semaphore,
        max_retries,
        logger,
    ) = args

    with semaphore:
        logger.info("Started processing: %s", video_name)
        start_time = time.time()
        retries = 0

        while retries <= max_retries:
            try:
                response = judge_video(
                    handler=handler,
                    system_prompt=system_prompt,
                    judge_mode=judge_mode,
                    user_prompt=user_prompt,
                    video_data=video_data,
                )

                elapsed_time = time.time() - start_time
                logger.info(
                    "Successfully processed: %s | Time: %.2fs | Retries: %d",
                    video_name,
                    elapsed_time,
                    retries,
                )

                response_data = {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "system_instruction": system_prompt,
                    "response_text": response.text if response.text is not None else "",
                    "model": handler.model_name,
                    "inference_time": elapsed_time,
                    "retries": retries,
                    "timestamp": datetime.now().isoformat(),
                }

                joblib.dump(response_data, cache_filepath)
                logger.info("Cached result to: %s", cache_filepath)

                create_datapoint_log(
                    judge_mode=judge_mode,
                    video_name=video_name,
                    response_data=response_data,
                    error_trace=None,
                )

                return True, video_path
            except Exception as e:
                retries += 1
                elapsed_time = time.time() - start_time
                error_trace = traceback.format_exc()

                if retries <= max_retries:
                    logger.warning(
                        "Error processing %s: %s | Retry %d/%d | Time elapsed: %.2fs",
                        video_name,
                        str(e),
                        retries,
                        max_retries,
                        elapsed_time,
                    )
                    time.sleep(2**retries)
                else:
                    logger.error(
                        "Failed processing %s after %d retries | Error: %s | Total time: %.2fs",
                        video_name,
                        max_retries,
                        str(e),
                        elapsed_time,
                    )

                    error_response_data = {
                        "video_name": video_name,
                        "video_path": str(video_path),
                        "system_instruction": system_prompt,
                        "response_text": f"ERROR: {str(e)}",
                        "model": handler.model_name,
                        "inference_time": elapsed_time,
                        "retries": retries,
                        "timestamp": datetime.now().isoformat(),
                    }

                    create_datapoint_log(
                        judge_mode=judge_mode,
                        video_name=video_name,
                        response_data=error_response_data,
                        error_trace=error_trace,
                    )

                    return False, video_path


def invoke_video_understanding_llm(
    handler: LLMHandler,
    write_mode: Literal["fill", "overwrite"] = "fill",
    max_size: int = 200,
    system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
    judge_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    user_prompt: str = "Follow the system prompt.",
    max_concurrent: int = 3,
    max_retries: int = 5,
    sampling_strategy: Literal["sequential", "balanced", "focused"] = "sequential",
) -> None:
    logger = setup_inference_logger(judge_mode)

    logger.info("Starting batch inference with %d videos", max_size)
    logger.info(
        "Configuration: write_mode=%s, max_concurrent=%d, max_retries=%d",
        write_mode,
        max_concurrent,
        max_retries,
    )
    logger.info("Sampling strategy: %s", sampling_strategy)
    logger.info("Model: %s", handler.model_name)

    videos = read_dataset(
        max_size=max_size,
        judge_mode=judge_mode,
        write_mode=write_mode,
        system_prompt=system_prompt,
        sampling_strategy=sampling_strategy,
    )

    logger.info("Loaded %d videos for processing", len(videos))

    semaphore = threading.Semaphore(max_concurrent)

    tasks = [
        (
            handler,
            video_path,
            video_name,
            cache_filepath,
            video_data,
            system_prompt,
            judge_mode,
            user_prompt,
            semaphore,
            max_retries,
            logger,
        )
        for video_path, (video_name, cache_filepath, video_data) in videos.items()
    ]

    batch_start_time = time.time()
    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_video, task) for task in tasks]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Processing videos (max {max_concurrent} concurrent)",
        ):
            results.append(future.result())

    batch_elapsed_time = time.time() - batch_start_time
    successful = sum(1 for success, _ in results if success)
    failed = len(videos) - successful

    logger.info("=" * 80)
    logger.info("Batch inference completed")
    logger.info("Total videos: %d", len(videos))
    logger.info("Successful: %d", successful)
    logger.info("Failed: %d", failed)
    logger.info("Success rate: %.2f%%", successful / len(videos) * 100)
    logger.info("Total time: %.2fs", batch_elapsed_time)
    logger.info("Average time per video: %.2fs", batch_elapsed_time / len(videos))
    logger.info("=" * 80)

    print(f"\nProcessed {successful}/{len(videos)} videos successfully")
    print(f"Total time: {batch_elapsed_time:.2f}s")


def judge_video(
    handler: LLMHandler,
    system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
    judge_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    user_prompt: str = "Follow the system prompt.",
    video_data: bytes | None = None,
):
    if video_data is None:
        raise ValueError("video_data cannot be None")

    judge = AnomalyJudge(handler)

    if judge_mode == EvalModes.VIDEO_SIMPLE:
        response = judge.judge_video_simple(
            video_data=video_data,
            system_prompt=system_prompt,
            user_prompt="Follow the system prompt and return valid JSON only.",
        )
    elif judge_mode == EvalModes.ONTOLOGICAL_DETECTIVES:
        response = judge.judge_video_ontological_detectives(
            video_data=video_data,
            system_prompt=Ontological_Detectives_Prompts.SYSTEM_PROMPT_ONTOLOGICAL,
            user_prompt=user_prompt,
        )
    elif judge_mode == EvalModes.ONTOLOGICAL_CATEGORIES:
        response = judge.judge_video_ontological_categories(
            video_data=video_data,
            system_prompt=Ontological_Prompts.SYSTEM_PROMPT_SYNTHESIZER,
            user_prompt=user_prompt,
        )
    else:
        raise ValueError(f"Unknown judge_mode: {judge_mode}")

    return response
