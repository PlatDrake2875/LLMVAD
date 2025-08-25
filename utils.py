import hashlib
from pathlib import Path
from typing import Any, Literal

from anomaly_detection import EvalModes


def read_dataset(
    max_size: int,
    judge_mode: EvalModes,
    write_mode: Literal["fill", "overwrite"],
    system_prompt: str,
) -> dict[str, Any]:
    project_root = Path(__file__).parent.absolute()
    DATASET_PATH = project_root / "datasets" / "XD_Violence_1-1004"
    CACHE_PATH = project_root / "cache" / judge_mode.value

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
