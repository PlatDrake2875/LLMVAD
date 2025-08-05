"""Main module for video anomaly detection using Google Gemini API."""

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Literal

import joblib
from google.genai import Client, types

from gemini_handler import get_client
from prompts import (
    SYSTEM_PROMPT_MASTER,
    SYSTEM_PROMPT_ONTOLOGICAL,
    SYSTEM_PROMPT_VIDEO_SIMPLE,
    get_system_prompt_detective,
)


class EvalModes(Enum):
    VIDEO_SIMPLE: str = "video_simple"
    ONTOLOGICAL: str = "ontological"


def invoke_video_understanding_llm(
    client: Client,
    mode: Literal["fill", "overwrite"] = "fill",
    max_size: int = 200,
    system_prompt: str = SYSTEM_PROMPT_VIDEO_SIMPLE,
    judge_mode: EvalModes = EvalModes.VIDEO_SIMPLE,
    user_prompt: str = "Follow the system prompt.",
) -> None:
    """Process videos for anomaly detection using Google Gemini API."""
    DATASET_PATH = Path("datasets") / "XD_Violence_1-1004"
    CACHE_PATH = Path("cache") / judge_mode.value

    CACHE_PATH.mkdir(exist_ok=True)

    for video_name in sorted(DATASET_PATH.iterdir())[:max_size]:
        if not video_name.name.endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = DATASET_PATH / video_name.name
        print(f"Processing video {video_path}")
        video_base_name = video_name.stem

        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
        cache_filename = f"{video_base_name}_{prompt_hash}.joblib"
        cache_filepath = CACHE_PATH / cache_filename

        if cache_filepath.exists():
            print("Video already cached!")
            continue

        with video_path.open("rb") as video_file:
            video_data = video_file.read()

        response = None
        if judge_mode == EvalModes.VIDEO_SIMPLE:
            user_prompt = "Follow the system prompt and return valid JSON only."
            response = judge_video_simple(
                client=client, video_data=video_data, system_prompt=system_prompt
            )
        elif judge_mode == EvalModes.ONTOLOGICAL:
            response = judge_video_ontological(
                client=client,
                video_data=video_data,
                system_prompt=SYSTEM_PROMPT_ONTOLOGICAL,
                user_prompt=user_prompt,
            )

        print(response.text)

        response_data = {
            "video_name": video_name.name,
            "video_path": str(video_path),
            "system_instruction": system_prompt,
            "response_text": response.text,
            "model": "gemini-2.5-flash",
        }

        joblib.dump(response_data, cache_filepath)


def judge_video_simple(
    client: Client,
    video_data: bytes,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = SYSTEM_PROMPT_VIDEO_SIMPLE,
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


def judge_video_ontological(
    client: Client,
    video_data: bytes,
    model_name: str = "gemini-2.5-flash",
    system_prompt: str = SYSTEM_PROMPT_ONTOLOGICAL,
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

    # Parse the JSON string into a list
    detectives_list = json.loads(detectives.text)
    print(f"Detectives list: {detectives_list}")

    detectives_scores = []
    for detective in detectives_list:
        detective_title = detective.split("<>")[0]
        detective_target = detective.split("<>")[1]
        SYSTEM_PROMPT_DETECTIVE = get_system_prompt_detective(
            detective_title, detective_target
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
            system_instruction=SYSTEM_PROMPT_MASTER,
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


def get_accuracy_report() -> dict[str, dict[str, float]]:
    CACHE_PATH = Path("cache") / EvalModes.VIDEO_SIMPLE.value
    labels = ["B1", "B2", "B4", "B5", "B6", "G", "A"]

    predictions, expected = [], []
    for cache_file in sorted(CACHE_PATH.glob("*.joblib")):
        data = joblib.load(cache_file)
        pred_data = json.loads(data["response_text"])
        video_name = data["video_name"]

        pred = {
            label: 1
            if any(
                item.get("tag_id") == label and item.get("score", 0) > 0.7
                for item in pred_data
            )
            else 0
            for label in labels
        }
        predictions.append(pred)

        exp_labels = video_name.split("label_")[1].split(".mp4")[0].split("-")
        exp = {label: 1 if label in exp_labels else 0 for label in labels}
        expected.append(exp)

    results = {}
    for label in labels:
        tp = tn = fp = fn = 0
        for pred, exp in zip(predictions, expected):
            if pred[label] == 1 and exp[label] == 1:
                tp += 1
            elif pred[label] == 0 and exp[label] == 0:
                tn += 1
            elif pred[label] == 1 and exp[label] == 0:
                fp += 1
            else:
                fn += 1

        total = tp + tn + fp + fn
        results[label] = {
            "accuracy": (tp + tn) / total if total > 0 else 0.0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    return results


def print_accuracy_report() -> None:
    results = get_accuracy_report()
    print("Class\tAccuracy\tPrecision\tRecall")
    print("-" * 40)

    for label in ["B1", "B2", "B4", "B5", "B6", "G", "A"]:
        r = results[label]
        print(
            f"{label}\t{r['accuracy']:.3f}\t\t{r['precision']:.3f}\t\t{r['recall']:.3f}"
        )

    overall = sum(r["accuracy"] for r in results.values()) / len(results)
    print(f"\nOverall Accuracy: {overall:.3f}")


if __name__ == "__main__":
    client = get_client()
    invoke_video_understanding_llm(client=client, judge_mode=EvalModes.ONTOLOGICAL)
    # print_accuracy_report()
    # invoke_image_llm(client, get_frame())
