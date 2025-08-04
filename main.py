"""Main module for video anomaly detection using Google Gemini API."""

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import joblib
from google import genai
from google.genai import Client, types

from prompts import SYSTEM_PROMPT_VIDEO_SIMPLE


def get_client() -> Client:
    """Initialize and return a Google Gemini client."""
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "GeminiAPI/gcp_credentials.json"

    client = genai.Client(
        vertexai=True,
        project="devtest-autopilot",
        location="us-central1",
    )

    return client


def get_frame() -> bytes:
    """Extract a single frame from a video file and return as bytes."""
    import cv2

    video_path = "datasets/XD_Violence_1-1004/A.Beautiful.Mind.2001__00-01-45_00-02-50_label_A.mp4"
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    cap.release()

    success, buffer = cv2.imencode(".jpg", frame)
    binary_image = buffer.tobytes()
    return binary_image


def invoke_text_llm(client: Client, content: str) -> str:
    """Invoke text-based LLM with the given content."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=content,
    )
    return response.text


def invoke_image_llm(client: Client, image: bytes) -> str:
    """Invoke image-based LLM with the given image."""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part.from_bytes(
                data=image,
                mime_type="image/jpeg",
            ),
            "Caption this image. Be thorough",
        ],
    )

    return response.text


def invoke_video_understanding_llm(
    client: Client,
    mode: Literal["fill", "overwrite"] = "fill",
    system_prompt: str = SYSTEM_PROMPT_VIDEO_SIMPLE,
) -> None:
    """Process videos for anomaly detection using Google Gemini API."""
    DATASET_PATH = Path("datasets") / "XD_Violence_1-1004"
    CACHE_PATH = Path("cache")

    CACHE_PATH.mkdir(exist_ok=True)

    for video_name in sorted(DATASET_PATH.iterdir())[:200]:
        if not video_name.name.endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue
        video_path = DATASET_PATH / video_name.name
        print(f"Processing video {video_path}")
        video_base_name = video_name.stem

        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()[:8]
        cache_filename = f"{video_base_name}_{prompt_hash}.joblib"
        cache_filepath = CACHE_PATH / cache_filename

        if cache_filepath.exists():
            _ = joblib.load(cache_filepath)
            print("Video already cached!")
            continue

        with video_path.open("rb") as video_file:
            video_data = video_file.read()

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
            ),
            contents=[
                types.Part.from_bytes(
                    data=video_data,
                    mime_type="video/mp4",
                ),
                "Follow the system prompt and return valid JSON only.",
            ],
        )
        # logging.log(logging.INFO, f"{response.text}")
        print(response.text)

        response_data = {
            "video_name": video_name.name,
            "video_path": str(video_path),
            "system_instruction": system_prompt,
            "response_text": response.text,
            "model": "gemini-2.5-flash",
        }

        joblib.dump(response_data, cache_filepath)


def get_accuracy_report() -> dict[str, dict[str, float]]:
    CACHE_PATH = Path("cache")
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
    # invoke_video_understanding_llm(client)
    print_accuracy_report()
    # invoke_image_llm(client, get_frame())
