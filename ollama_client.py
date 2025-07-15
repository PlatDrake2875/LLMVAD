import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from PIL import Image
from typing import List
import io
import base64

class HuggingFaceGemmaClient:
    """
    Client for using HuggingFace's Gemma 3 model to get image descriptions and summaries.
    """
    def __init__(self, model_name: str = "google/gemma-2-2b-it", device: str = "auto"):
        """
        Initializes the HuggingFaceGemmaClient.

        Args:
            model_name (str): The name of the HuggingFace model to use.
            device (str): Device to run the model on ('auto', 'cuda', 'cpu').
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        ) if self.device == "cuda" else None
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logging.info(f"HuggingFaceGemmaClient initialized with model: {self.model_name}, device: {self.device}")

    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """
        Generates text using the Gemma model.

        Args:
            prompt (str): The input prompt for text generation.
            max_length (int): Maximum length of generated text.

        Returns:
            str: Generated text response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Converts a PIL Image to a Base64 encoded string.

        Args:
            image (PIL.Image.Image): The PIL Image object to encode.

        Returns:
            str: Base64 encoded string of the image.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def summarize_frame(self, frame_image: Image.Image, frame_number: int) -> str:
        """
        Processes a single video frame and returns a text description.

        Args:
            frame_image (PIL.Image.Image): The PIL Image of the video frame.
            frame_number (int): The number of the frame being processed.

        Returns:
            str: The summarized text description from the Gemma model.
        """
        logging.info(f"Processing frame {frame_number} with Gemma model...")
        
        prompt = (
            "Describe what you see in this image in detail. Focus on people, objects, actions, "
            "and the overall scene. Provide a comprehensive but concise description:\n\n"
        )
        
        try:
            description = self._generate_text(prompt)
            logging.info(f"Successfully processed frame {frame_number}.")
            return description
        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {e}")
            return f"Error: Failed to get summary for frame {frame_number}. {e}"

    def summarize_chunks(self, video_descriptions: List[str], chunk_size: int = 3) -> List[str]:
        """
        Summarizes chunks of video frame descriptions using the Gemma model.

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
            current_chunk_descriptions = video_descriptions[i : i + chunk_size]
            frame_descriptions_text = "\n".join(current_chunk_descriptions)

            prompt = (
                "Summarize the following video frame descriptions into a single, cohesive, "
                "long description. Focus on key actions, objects, and changes observed across the frames. "
                "Do not add any conversational filler, just the summary:\n\n"
                f"{frame_descriptions_text}"
            )

            logging.info(f"Summarizing chunk {i // chunk_size + 1} (frames {i} to {min(i + chunk_size, num_descriptions) - 1})...")

            try:
                generated_summary = self._generate_text(prompt)
                chunk_summaries.append(generated_summary)
                logging.info(f"Chunk {i // chunk_size + 1} summarized.")

            except Exception as e:
                logging.error(f"Error summarizing chunk starting at index {i}: {e}")
                chunk_summaries.append(f"Error: Failed to summarize chunk. {e}")
        
        logging.info(f"Finished summarizing {len(chunk_summaries)} chunks.")
        return chunk_summaries

