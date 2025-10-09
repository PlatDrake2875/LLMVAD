import concurrent.futures
import threading
import time
from typing import Literal

import joblib
from tqdm import tqdm

from anomaly_judge import AnomalyJudge
from enums import EvalModes
from llm_handler import LLMHandler
from prompts import Ontological_Detectives_Prompts, Ontological_Prompts, Simple_Prompts
from utils import read_dataset


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
    ) = args

    with semaphore:
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

                print(f"\nProcessed: {video_name}")

                response_data = {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "system_instruction": system_prompt,
                    "response_text": response.text if response.text is not None else "",
                    "model": handler.model_name,
                }

                joblib.dump(response_data, cache_filepath)
                return True, video_path
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    print(
                        f"\nError processing {video_name}: {str(e)}. Retry {retries}/{max_retries}"
                    )
                    # Sleep with exponential backoff before retrying
                    time.sleep(2**retries)
                else:
                    print(
                        f"\nFailed processing {video_name} after {max_retries} retries: {str(e)}"
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
    videos = read_dataset(
        max_size=max_size,
        judge_mode=judge_mode,
        write_mode=write_mode,
        system_prompt=system_prompt,
        sampling_strategy=sampling_strategy,
    )

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
        )
        for video_path, (video_name, cache_filepath, video_data) in videos.items()
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_video, task) for task in tasks]

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Processing videos (max {max_concurrent} concurrent)",
        ):
            results.append(future.result())

    successful = sum(1 for success, _ in results if success)
    print(f"\nProcessed {successful}/{len(videos)} videos successfully")


def judge_video(
    handler: LLMHandler,
    system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
    judge_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    user_prompt: str = "Follow the system prompt.",
    video_data: bytes | None = None,
):
    if video_data is None:
        raise ValueError("video_data cannot be None")

    # Create an AnomalyJudge instance with the LLM handler
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
