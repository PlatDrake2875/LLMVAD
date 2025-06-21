import logging
import os
import cv2
from PIL import Image
import pickle # Import the pickle module
from typing import Generator # Import Generator for type hinting

# Assuming OllamaClient is in the same package/directory level
from ollama_client import OllamaClient

class VideoProcessor:
    """
    Processes video files, extracts frames, and sends them to an OllamaClient
    for summarization.
    """
    def __init__(self, video_directory: str, ollama_client: OllamaClient, frame_interval: int = 10, summarization_chunk_size: int = 3):
        """
        Initializes the VideoProcessor.

        Args:
            video_directory (str): The directory containing the video files.
            ollama_client (OllamaClient): An instance of OllamaClient for API communication.
            frame_interval (int): Process every 'frame_interval'th frame.
            summarization_chunk_size (int): The number of frame descriptions to group into one chunk for higher-level summarization.
        """
        self.video_directory = video_directory
        self.ollama_client = ollama_client
        self.frame_interval = frame_interval
        self.summarization_chunk_size = summarization_chunk_size
        self.descriptions = [] # Stores all collected descriptions for the current video
        self.supported_formats = ('.mp4', '.avi', '.mov', '.mkv')
        logging.info(f"VideoProcessor initialized for directory: {self.video_directory}, interval: {self.frame_interval}, summarization chunk size: {self.summarization_chunk_size}")

    def _locate_video_file(self) -> Generator[str, None, None]:
        """
        A generator function that yields paths to supported video files
        found in the specified directory.

        Yields:
            str: The path to a video file.
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
        sends them to the Ollama client for summarization, and pickles
        the individual frame descriptions and the chunked summaries for each video.
        """
        logging.info("Starting video processing...")
        
        # Iterate over all video files found by the generator
        for video_file in self._locate_video_file():
            logging.info(f"Processing video: {video_file}")
            
            video_capture = cv2.VideoCapture(video_file)
            if not video_capture.isOpened():
                logging.error(f"Error: Could not open video file {video_file}. Skipping to next video if available.")
                continue # Skip to the next video if this one can't be opened

            self.descriptions = [] # Reset descriptions for the new video
            frame_count = 0
            
            while video_capture.isOpened():
                success, frame = video_capture.read()

                if not success:
                    logging.info(f"Reached the end of video {video_file} or failed to read frame.")
                    break

                # Process frame only if it's at the specified interval
                if frame_count % self.frame_interval == 0:
                    logging.info(f"Processing frame number {frame_count} from {video_file}...")
                    # OpenCV reads images in BGR format, convert to RGB for PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert the numpy array (frame) to a PIL Image
                    pil_image = Image.fromarray(frame_rgb)

                    summary = self.ollama_client.summarize_frame(pil_image, frame_count)
                    self.descriptions.append(summary)

                    print("\n" + "="*25)
                    print(f"    MODEL OUTPUT (VIDEO: {os.path.basename(video_file)}, FRAME: {frame_count})")
                    print("="*25)
                    print(summary)
                    print("="*25 + "\n")
                
                frame_count += 1

            video_capture.release()
            logging.info(f"Finished processing video: {video_file}. Total frames read: {frame_count}. Total summaries generated: {len(self.descriptions)}.")

            video_basename = os.path.basename(video_file)
            file_name_without_ext = os.path.splitext(video_basename)[0] 
            
            # --- Pickle individual frame descriptions ---
            pickle_filename = f"{file_name_without_ext}_frame_descriptions.pkl" # More specific name
            pickle_path = os.path.join(self.video_directory, pickle_filename) # Save in the same directory as videos

            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump(self.descriptions, f)
                logging.info(f"Individual frame descriptions for '{video_basename}' pickled to: {pickle_path}")
            except Exception as e:
                logging.error(f"Error pickling individual frame descriptions for '{video_basename}': {e}")

            # --- Summarize chunks and pickle them ---
            if self.descriptions: # Only attempt if there are descriptions
                logging.info(f"Initiating chunk summarization for video: {video_basename}")
                chunked_summaries = self.ollama_client.summarize_chunks(
                    self.descriptions, 
                    chunk_size=self.summarization_chunk_size # Use the instance variable for chunk size
                )
                
                logging.info(f"Chunked summaries for '{video_basename}':")
                # for j, summary in enumerate(chunked_summaries):
                #     print(f"\n--- Chunk Summary {j+1} (Video: {video_basename}) ---")
                #     print(summary)
                
                # Pickle the chunked summaries
                chunk_pickle_filename = f"{file_name_without_ext}_chunked_summaries.pkl"
                chunk_pickle_path = os.path.join(self.video_directory, chunk_pickle_filename)
                try:
                    with open(chunk_pickle_path, 'wb') as f:
                        pickle.dump(chunked_summaries, f)
                    logging.info(f"Chunked summaries for '{video_basename}' pickled to: {chunk_pickle_path}")
                except Exception as e:
                    logging.error(f"Error pickling chunked summaries for '{video_basename}': {e}")
            else:
                logging.warning(f"No individual frame descriptions found for '{video_basename}', skipping chunk summarization and pickling.")

        logging.info("Finished processing all videos.")


    def get_all_descriptions(self) -> list[str]:
        """
        Returns the list of all collected frame descriptions from the LAST PROCESSED video.
        Note: With _locate_video_file being a generator, this method will only return
        descriptions for the very last video processed.

        Returns:
            list[str]: A list of strings, where each string is a frame summary.
        """
        return self.descriptions

