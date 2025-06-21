import logging
import base64
import requests
import json
from PIL import Image
from io import BytesIO
from typing import List

class OllamaClient:
    """
    Client for interacting with the Ollama API to get image descriptions and summaries.
    """
    def __init__(self, api_url: str = "http://localhost:11434/api/chat",
                 model_name: str = "gemma3:4b-it-q4_K_M",
                 timeout: int = 30):
        """
        Initializes the OllamaClient.

        Args:
            api_url (str): The URL of the Ollama API chat endpoint.
            model_name (str): The name of the model to use for inference.
            timeout (int): Timeout for the API call in seconds.
        """
        self.api_url = api_url
        self.model_name = model_name
        self.timeout = timeout
        logging.info(f"OllamaClient initialized with API URL: {self.api_url}, model: {self.model_name}")

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Converts a PIL Image to a Base64 encoded string.

        Args:
            image (PIL.Image.Image): The PIL Image object to encode.

        Returns:
            str: Base64 encoded string of the image.
        """
        buffered = BytesIO()
        # Save image to an in-memory buffer in JPEG format
        image.save(buffered, format="JPEG")
        # Encode the bytes to a base64 string
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def summarize_frame(self, frame_image: Image.Image, frame_number: int) -> str:
        """
        Encodes a single video frame, sends it to the Ollama API for description,
        and returns the model's output.

        Args:
            frame_image (PIL.Image.Image): The PIL Image of the video frame.
            frame_number (int): The number of the frame being processed.

        Returns:
            str: The summarized text description from the Ollama model.
        """
        logging.info(f"Preparing frame {frame_number} for Ollama API...")
        base64_image = self._image_to_base64(frame_image)
        logging.info(f"Frame {frame_number} converted to Base64.")

        prompt_text = "Summarize what you see on screen. Format your response into a block of text. Don't say anything else."

        logging.info(f"Sending frame {frame_number} to Ollama API (model: {self.model_name})...")

        # Construct the payload for the /api/chat endpoint
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_text,
                    "images": [base64_image]
                }
            ],
            "stream": False
        }

        try:
            # Make the POST request
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=self.timeout
            )
            # Raise an exception for HTTP errors (e.g., 404, 500)
            response.raise_for_status()

            logging.info(f"Successfully received response for frame {frame_number}.")

            # Process and return the output
            response_data = response.json()
            generated_text = response_data.get("message", {}).get("content", "No response text found in API output.")
            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            logging.error(f"Error communicating with Ollama API for frame {frame_number}: {e}")
            return f"Error: Failed to get summary for frame {frame_number}. {e}"

    def summarize_chunks(self, video_descriptions: List[str], chunk_size: int = 3) -> List[str]:
        """
        Summarizes chunks of video frame descriptions using the Ollama API.

        Args:
            video_descriptions (List[str]): A list of individual frame descriptions.
            chunk_size (int): The number of frame descriptions to group into one chunk for summarization.

        Returns:
            List[str]: A list of summarized texts for each chunk.
        """
        logging.info(f"Starting summarization of video descriptions in chunks of size {chunk_size}...")
        chunk_summaries = []
        num_descriptions = len(video_descriptions)

        if num_descriptions == 0:
            logging.warning("No descriptions to summarize.")
            return []

        for i in range(0, num_descriptions, chunk_size - 1):
            # Get the chunk of frame descriptions
            current_chunk_descriptions = video_descriptions[i : i + chunk_size]
            # Join them into a single string for the prompt
            frame_descriptions_text = "\n".join(current_chunk_descriptions)

            prompt_text = (
                "Summarize the following video frame descriptions into a single, cohesive, "
                "long description. Focus on key actions, objects, and changes observed across the frames. "
                "Do not add any conversational filler, just the summary:\n\n"
                f"{frame_descriptions_text}"
            )

            logging.info(f"Summarizing chunk {i // chunk_size + 1} (frames {i} to {min(i + chunk_size, num_descriptions) - 1})...")

            # Construct the payload for the /api/chat endpoint
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                "stream": False
            }

            try:
                # Make the POST request
                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                response.raise_for_status() # Raise HTTP errors

                response_data = response.json()
                generated_summary = response_data.get("message", {}).get("content", "No summary found for this chunk.")
                chunk_summaries.append(generated_summary.strip())
                logging.info(f"Chunk {i // chunk_size + 1} summarized.")

            except requests.exceptions.RequestException as e:
                logging.error(f"Error summarizing chunk starting at index {i}: {e}")
                chunk_summaries.append(f"Error: Failed to summarize chunk. {e}")
        
        logging.info(f"Finished summarizing {len(chunk_summaries)} chunks.")
        return chunk_summaries

