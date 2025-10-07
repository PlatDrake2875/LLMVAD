import concurrent.futures
import hashlib
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import joblib
from tqdm import tqdm

from anomaly_judge import AnomalyJudge
from llm_handler import LLMHandler, LLMHandlerFactory
from prompts import Ontological_Detectives_Prompts, Ontological_Prompts, Simple_Prompts


class EvalModes(Enum):
    VIDEO_SIMPLE = "video_simple"
    ONTOLOGICAL_DETECTIVES = "ontological_detectives"
    ONTOLOGICAL_CATEGORIES = "ontological_categories"


def read_dataset(
    max_size: int,
    judge_mode: EvalModes | str,
    write_mode: Literal["fill", "overwrite"],
    system_prompt: str,
) -> dict[str, Any]:
    project_root = Path(__file__).parent.absolute()
    DATASET_PATH = project_root / "datasets" / "XD_Violence_1-1004"
    judge_mode_value = (
        judge_mode.value if isinstance(judge_mode, EvalModes) else judge_mode
    )
    CACHE_PATH = project_root / "cache" / judge_mode_value

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")

    if not DATASET_PATH.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {DATASET_PATH}")

    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    videos_data = {}
    for video_name in sorted(DATASET_PATH.iterdir())[:max_size]:
        if not video_name.name.endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = DATASET_PATH / video_name.name
        video_base_name = video_name.stem

        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
        cache_filename = f"{video_base_name}_{prompt_hash}.joblib"
        cache_filepath = CACHE_PATH / cache_filename

        if write_mode == "fill" and cache_filepath.exists():
            continue

        with video_path.open("rb") as video_file:
            video_data = video_file.read()

        videos_data[video_path] = [video_base_name, cache_filepath, video_data]

    return videos_data


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
                    "response_text": response.text,
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
) -> None:
    videos = read_dataset(
        max_size=max_size,
        judge_mode=judge_mode,
        write_mode=write_mode,
        system_prompt=system_prompt,
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


if __name__ == "__main__":
    handler = LLMHandlerFactory.create_handler(
        provider="gemini", model_name="gemini-2.5-flash"
    )

    invoke_video_understanding_llm(
        handler=handler,
        judge_mode=EvalModes.ONTOLOGICAL_CATEGORIES,
        max_concurrent=8,
        max_size=10,
        max_retries=5,
    )
