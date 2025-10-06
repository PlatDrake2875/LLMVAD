import concurrent.futures
import hashlib
import json
import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import joblib
from google.genai import Client, types
from tqdm import tqdm

from gemini_handler import get_client
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
        client,
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
                    client=client,
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
                    "model": "gemini-2.5-flash",
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
    client: Client,
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
            client,
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
    client: Client,
    system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
    judge_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    user_prompt: str = "Follow the system prompt.",
    video_data: bytes | None = None,
):
    if video_data is None:
        raise ValueError("video_data cannot be None")

    if judge_mode == EvalModes.VIDEO_SIMPLE:
        user_prompt = "Follow the system prompt and return valid JSON only."
        response = judge_video_simple(
            client=client, video_data=video_data, system_prompt=system_prompt
        )
    elif judge_mode == EvalModes.ONTOLOGICAL_DETECTIVES:
        response = judge_video_ontological_detectives(
            client=client,
            video_data=video_data,
            system_prompt=Ontological_Detectives_Prompts.SYSTEM_PROMPT_ONTOLOGICAL,
            user_prompt=user_prompt,
        )
    elif judge_mode == EvalModes.ONTOLOGICAL_CATEGORIES:
        response = judge_video_ontological_categories(
            client=client,
            video_data=video_data,
            system_prompt=Ontological_Prompts.SYSTEM_PROMPT_SYNTHESIZER,
            user_prompt=user_prompt,
        )
    else:
        raise ValueError(f"Unknown judge_mode: {judge_mode}")

    return response


def judge_video_simple(
    client: Client,
    video_data: bytes,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = Simple_Prompts.SYSTEM_PROMPT_VIDEO_SIMPLE,
    user_prompt: str = "Follow the system prompt and return valid JSON only.",
):
    response = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
        contents=[
            types.Part.from_bytes(
                data=video_data,
                mime_type="video/mp4",
            ),
            user_prompt,
        ],
    )

    return response


def judge_video_ontological_detectives(
    client: Client,
    video_data: bytes,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = Ontological_Detectives_Prompts.SYSTEM_PROMPT_ONTOLOGICAL,
    user_prompt: str = "Follow the system prompt.",
):
    detectives = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
        contents=[
            types.Part.from_bytes(
                data=video_data,
                mime_type="video/mp4",
            ),
            user_prompt,
        ],
    )

    if detectives.text is None:
        raise ValueError("No response text from detectives generation")
    detectives_list = json.loads(detectives.text)
    print(f"Detectives list: {detectives_list}")

    detectives_scores = []
    for detective in detectives_list:
        detective_title = detective.split("<>")[0]
        detective_target = detective.split("<>")[1]
        SYSTEM_PROMPT_DETECTIVE = (
            Ontological_Detectives_Prompts.get_system_prompt_detective(
                detective_title, detective_target
            )
        )

        detective_scores = client.models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_DETECTIVE,
                response_mime_type="application/json",
            ),
            contents=[
                types.Part.from_bytes(
                    data=video_data,
                    mime_type="video/mp4",
                ),
                user_prompt,
            ],
        )
        print(detective_scores.text)
        detectives_scores.append(detective_scores)

    user_prompt = ""
    for detective_scores in detectives_scores:
        user_prompt += detective_scores.text + "\n\n"

    master_scores = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=Ontological_Detectives_Prompts.SYSTEM_PROMPT_MASTER,
            response_mime_type="application/json",
        ),
        contents=[
            types.Part.from_bytes(
                data=video_data,
                mime_type="video/mp4",
            ),
            user_prompt,
        ],
    )

    return master_scores


def judge_video_ontological_categories(
    client: Client,
    video_data: bytes,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = Ontological_Prompts.SYSTEM_PROMPT_SYNTHESIZER,
    user_prompt: str = "Follow the system prompt.",
    subjects: list[str] | None = None,
):
    if subjects is None:
        subjects = [
            "violence",
            "nature",
            "sports",
            "urban",
            "vehicles",
            "crowds",
            "weapons",
            "emergency",
            "normal activity",
            "religious ritual",
            "culture",
        ]
    subject_scores = []
    for subject in subjects:
        SYSTEM_PROMPT_SYSTEM = Ontological_Prompts.get_system_prompt_base(subject)
        subject_score = client.models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT_SYSTEM,
                response_mime_type="application/json",
            ),
            contents=[
                types.Part.from_bytes(
                    data=video_data,
                    mime_type="video/mp4",
                ),
                user_prompt,
            ],
        )
        print(subject_score.text)
        subject_scores.append(subject_score)

    user_prompt = ""
    for subject_score in subject_scores:
        user_prompt += subject_score.text + "\n\n"

    master_scores = client.models.generate_content(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
        ),
        contents=[
            types.Part.from_bytes(
                data=video_data,
                mime_type="video/mp4",
            ),
            user_prompt,
        ],
    )

    return master_scores


if __name__ == "__main__":
    client = get_client()
    invoke_video_understanding_llm(
        client=client,
        judge_mode=EvalModes.ONTOLOGICAL_CATEGORIES,
        max_concurrent=8,
        max_size=10,
        max_retries=5,
    )
