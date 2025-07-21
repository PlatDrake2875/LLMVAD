import logging
import os
import pickle
from typing import Generator

import cv2
from PIL import Image

from gemma_client import HuggingFaceGemmaClient


class VideoProcessor:
    """
    Processes video files, extracts frames, and sends them to a HuggingFaceGemmaClient
    for summarization.
    """

    def __init__(
        self,
        video_directory: str,
        model_client: HuggingFaceGemmaClient,
        frame_interval: int = 10,
        summarization_chunk_size: int = 3,
    ):
        """
        Initializes the VideoProcessor.
        """
        self.video_directory = video_directory
        self.model_client = model_client
        self.frame_interval = frame_interval
        self.summarization_chunk_size = summarization_chunk_size
        self.descriptions = []
        self.supported_formats = (".mp4", ".avi", ".mov", ".mkv")
        logging.info(
            f"VideoProcessor initialized for directory: {self.video_directory}, interval: {self.frame_interval}, summarization chunk size: {self.summarization_chunk_size}"
        )

    def _locate_video_file(self) -> Generator[str, None, None]:
        """
        A generator function that yields paths to supported video files
        found in the specified directory.
        """
        for file in sorted(os.listdir(self.video_directory)):
            if file.lower().endswith(self.supported_formats):
                video_path = os.path.join(self.video_directory, file)
                logging.info(f"Found video file: {video_path}")
                yield video_path
        logging.info(f"Finished searching for video files in: {self.video_directory}")

    def process_video(self):
        """
        Locates video(s), extracts frames at the specified interval,
        sends them to the Gemma client for summarization, and pickles
        the individual frame descriptions and the chunked summaries for each video.
        """
        logging.info("Starting video processing...")

        for video_file in self._locate_video_file():
            logging.info(f"Processing video: {video_file}")

            video_capture = cv2.VideoCapture(video_file)
            if not video_capture.isOpened():
                logging.error(
                    f"Error: Could not open video file {video_file}. Skipping to next video if available."
                )
                continue

            self.descriptions = []
            frame_count = 0

            while video_capture.isOpened():
                success, frame = video_capture.read()

                if not success:
                    logging.info(
                        f"Reached the end of video {video_file} or failed to read frame."
                    )
                    break

                if frame_count % self.frame_interval == 0:
                    logging.info(
                        f"Processing frame number {frame_count} from {video_file}..."
                    )
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)

                    summary = self.model_client.summarize_frame(pil_image, frame_count)
                    self.descriptions.append(summary)

                    model_output_text = f"\n{'=' * 25}\n    MODEL OUTPUT (VIDEO: {os.path.basename(video_file)}, FRAME: {frame_count})\n{'=' * 25}\n{summary}\n{'=' * 25}\n"
                    logging.info(model_output_text)

                frame_count += 1

            video_capture.release()
            logging.info(
                f"Finished processing video: {video_file}. Total frames read: {frame_count}. Total summaries generated: {len(self.descriptions)}."
            )

            video_basename = os.path.basename(video_file)
            file_name_without_ext = os.path.splitext(video_basename)[0]

            pickle_filename = f"{file_name_without_ext}_frame_descriptions.pkl"
            pickle_path = os.path.join(self.video_directory, pickle_filename)

            try:
                with open(pickle_path, "wb") as f:
                    pickle.dump(self.descriptions, f)
                logging.info(
                    f"Individual frame descriptions for '{video_basename}' pickled to: {pickle_path}"
                )
            except Exception as e:
                logging.error(
                    f"Error pickling individual frame descriptions for '{video_basename}': {e}"
                )

            if self.descriptions:
                logging.info(
                    f"Initiating chunk summarization for video: {video_basename}"
                )
                chunked_summaries = self.model_client.summarize_chunks(
                    self.descriptions,
                    chunk_size=self.summarization_chunk_size,
                )

                logging.info(f"Chunked summaries for '{video_basename}':")
                chunk_pickle_filename = f"{file_name_without_ext}_chunked_summaries.pkl"
                chunk_pickle_path = os.path.join(
                    self.video_directory, chunk_pickle_filename
                )
                try:
                    with open(chunk_pickle_path, "wb") as f:
                        pickle.dump(chunked_summaries, f)
                    logging.info(
                        f"Chunked summaries for '{video_basename}' pickled to: {chunk_pickle_path}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error pickling chunked summaries for '{video_basename}': {e}"
                    )
            else:
                logging.warning(
                    f"No individual frame descriptions found for '{video_basename}', skipping chunk summarization and pickling."
                )

        logging.info("Finished processing all videos.")

    def get_all_descriptions(self) -> list[str]:
        """
        Returns the list of all collected frame descriptions from the LAST PROCESSED video.
        Note: With _locate_video_file being a generator, this method will only return
        descriptions for the very last video processed.
        """
        return self.descriptions
