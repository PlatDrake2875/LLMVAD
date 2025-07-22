import logging
import os
import pickle
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from .gemini_client import GeminiClient


class VideoProcessor:
    def __init__(
        self,
        video_directory: str,
        model_client: GeminiClient,
        frame_interval: int = 10,
        summarization_chunk_size: int = 3,
    ):
        self.video_directory = video_directory
        self.model_client = model_client
        self.frame_interval = frame_interval
        self.summarization_chunk_size = summarization_chunk_size

    def process_video(self):
        video_files = sorted(
            [f for f in os.listdir(self.video_directory) if f.lower().endswith(".mp4")]
        )

        if not video_files:
            logging.error(f"No video files found in {self.video_directory}")
            return

        for video_file in video_files:
            logging.info(f"Processing video: {video_file}")
            self._process_single_video(video_file)

    def _process_single_video(self, video_file: str):
        video_path = os.path.join(self.video_directory, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logging.error(f"Could not open video file: {video_path}")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        logging.info(f"Video info: {total_frames} frames, {fps:.2f} fps")

        frame_descriptions = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.frame_interval == 0:
                description = self._analyze_frame(frame, frame_count)
                if description:
                    frame_descriptions.append(description)
                    logging.info(f"Frame {frame_count}: {description[:100]}...")

            frame_count += 1

        cap.release()

        if frame_descriptions:
            self._create_chunked_summaries(frame_descriptions, video_file)
        else:
            logging.warning(f"No frame descriptions generated for {video_file}")

    def _analyze_frame(self, frame: np.ndarray, frame_number: int) -> Optional[str]:
        try:
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            prompt = f"""Analyze this video frame and provide a concise description of what you see. Focus on:
- People and their actions
- Objects and their interactions
- Any unusual or notable activities
- Overall scene context

Frame number: {frame_number}

Provide a clear, factual description in 1-2 sentences."""

            response = self.model_client.generate_text(
                prompt, temperature=0.3, max_tokens=100
            )

            return response.strip()

        except Exception as e:
            logging.error(f"Error analyzing frame {frame_number}: {e}")
            return None

    def _create_chunked_summaries(self, frame_descriptions: List[str], video_file: str):
        if not frame_descriptions:
            return

        chunked_summaries = []

        for i in range(0, len(frame_descriptions), self.summarization_chunk_size):
            chunk = frame_descriptions[i : i + self.summarization_chunk_size]
            summary = self._summarize_chunk(
                chunk, i // self.summarization_chunk_size + 1
            )
            if summary:
                chunked_summaries.append(summary)

        if chunked_summaries:
            self._save_summaries(chunked_summaries, video_file)
        else:
            logging.warning(f"No chunked summaries created for {video_file}")

    def _summarize_chunk(self, chunk: List[str], chunk_number: int) -> Optional[str]:
        if not chunk:
            return None

        chunk_text = "\n".join(
            [f"Frame {i + 1}: {desc}" for i, desc in enumerate(chunk)]
        )

        prompt = f"""Summarize this sequence of video frames into a coherent description of the events:

{chunk_text}

Provide a concise summary (2-3 sentences) that captures the key events and any notable activities."""

        try:
            response = self.model_client.generate_text(
                prompt, temperature=0.2, max_tokens=150
            )

            return response.strip()

        except Exception as e:
            logging.error(f"Error summarizing chunk {chunk_number}: {e}")
            return None

    def _save_summaries(self, summaries: List[str], video_file: str):
        file_name_without_ext = os.path.splitext(video_file)[0]
        summaries_filename = f"{file_name_without_ext}_chunked_summaries.pkl"
        summaries_path = os.path.join(self.video_directory, summaries_filename)

        try:
            with open(summaries_path, "wb") as f:
                pickle.dump(summaries, f)
            logging.info(f"Chunked summaries saved to: {summaries_path}")
        except Exception as e:
            logging.error(f"Error saving summaries: {e}")
