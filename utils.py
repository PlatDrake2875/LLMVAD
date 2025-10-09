import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from enums import EvalModes


def _extract_labels_from_filename(filename: str) -> list[str]:
    """Extract label(s) from video filename format: ...label_X.mp4 or ...label_X-Y-Z.mp4."""
    try:
        label_part = filename.split("label_")[1].split(".mp4")[0]
        return label_part.split("-")
    except (IndexError, AttributeError):
        return []


def _group_videos_by_label(video_paths: list[Path]) -> dict[str, list[Path]]:
    """Group video paths by their label(s)."""
    label_groups: dict[str, list[Path]] = defaultdict(list)
    for video_path in video_paths:
        labels = _extract_labels_from_filename(video_path.name)
        for label in labels:
            label_groups[label].append(video_path)
    return label_groups


def _sample_sequential(video_paths: list[Path], max_size: int) -> list[Path]:
    """Sample videos sequentially (sorted by filename)."""
    return sorted(video_paths)[:max_size]


def _sample_balanced(video_paths: list[Path], max_size: int) -> list[Path]:
    """Sample videos trying to balance each category as much as possible."""
    label_groups = _group_videos_by_label(video_paths)

    if not label_groups:
        return sorted(video_paths)[:max_size]

    num_labels = len(label_groups)
    samples_per_label = max(1, max_size // num_labels)

    sampled_videos: set[Path] = set()

    for _, videos in label_groups.items():
        available = [v for v in videos if v not in sampled_videos]
        sampled_videos.update(available[:samples_per_label])

    if len(sampled_videos) < max_size:
        remaining_videos = [v for v in video_paths if v not in sampled_videos]
        additional_needed = max_size - len(sampled_videos)
        sampled_videos.update(remaining_videos[:additional_needed])

    return list(sampled_videos)[:max_size]


def _sample_focused(
    video_paths: list[Path], max_size: int, focus_label: str
) -> list[Path]:
    """Sample videos only from a specific label category."""
    label_groups = _group_videos_by_label(video_paths)

    if focus_label not in label_groups:
        raise ValueError(
            f"Focus label '{focus_label}' not found in dataset. "
            f"Available labels: {sorted(label_groups.keys())}"
        )

    focused_videos = label_groups[focus_label]
    return focused_videos[:max_size]


def read_dataset(
    max_size: int,
    judge_mode: EvalModes | str,
    write_mode: Literal["fill", "overwrite"],
    system_prompt: str,
    dataset_name: str = "XD_Violence_1-1004",
    sampling_strategy: Literal["sequential", "balanced", "focused"] = "sequential",
    focus_label: str | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).parent.absolute()
    DATASET_PATH = project_root / "datasets" / dataset_name
    judge_mode_value = (
        judge_mode.value if isinstance(judge_mode, EvalModes) else judge_mode
    )
    CACHE_PATH = project_root / "cache" / judge_mode_value

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {DATASET_PATH}")

    if not DATASET_PATH.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {DATASET_PATH}")

    if sampling_strategy == "focused" and focus_label is None:
        raise ValueError(
            "focus_label must be provided when sampling_strategy is 'focused'"
        )

    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    all_video_paths = [
        video_path
        for video_path in DATASET_PATH.iterdir()
        if video_path.name.endswith((".mp4", ".avi", ".mov", ".mkv"))
    ]

    if sampling_strategy == "sequential":
        selected_videos = _sample_sequential(all_video_paths, max_size)
    elif sampling_strategy == "balanced":
        selected_videos = _sample_balanced(all_video_paths, max_size)
    elif sampling_strategy == "focused":
        selected_videos = _sample_focused(all_video_paths, max_size, focus_label)  # type: ignore
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    videos_data = {}
    for video_path in selected_videos:
        video_base_name = video_path.stem

        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
        cache_filename = f"{video_base_name}_{prompt_hash}.joblib"
        cache_filepath = CACHE_PATH / cache_filename

        if write_mode == "fill" and cache_filepath.exists():
            continue

        with video_path.open("rb") as video_file:
            video_data = video_file.read()

        videos_data[video_path] = [video_base_name, cache_filepath, video_data]

    return videos_data
