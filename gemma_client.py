import logging
import torch
import torch._dynamo
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
from typing import List
import io
import base64
import os
from hf_auth import get_hf_token

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision("high")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


class HuggingFaceGemmaClient:
    """
    Client for using HuggingFace's Gemma 3 multimodal model to get image descriptions and summaries.
    """

    def __init__(self, model_name: str = "google/gemma-3-4b-it", device: str = "auto"):
        """
        Initializes the HuggingFaceGemmaClient.
        """
        self.model_name = model_name
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        hf_token = get_hf_token()

        self.processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager",
            token=hf_token,
        ).eval()

        logging.info(
            f"HuggingFaceGemmaClient initialized with model: {self.model_name}, device: {self.device}"
        )

    def _generate_text_from_image(
        self, image: Image.Image, prompt: str, max_new_tokens: int = 512
    ) -> str:
        """
        Generates text description from an image using the Gemma 3 multimodal model.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()

    def _generate_text(self, prompt: str, max_length: int = 512) -> str:
        """
        Generates text using the Gemma 3 model (text-only).
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(
            self.model.device,
            dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
        )

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]

        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()

    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Converts a PIL Image to a Base64 encoded string.
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def summarize_frame(self, frame_image: Image.Image, frame_number: int) -> str:
        """
        Processes a single video frame and returns a text description using Gemma 3's vision capabilities.
        """
        logging.info(
            f"Processing frame {frame_number} with Gemma 3 multimodal model..."
        )

        prompt = (
            "Describe what you see in this image in detail. Focus on people, objects, actions, "
            "and the overall scene. Provide a comprehensive but concise description of the video frame."
        )

        try:
            description = self._generate_text_from_image(frame_image, prompt)
            logging.info(f"Successfully processed frame {frame_number}.")
            return description
        except Exception as e:
            logging.error(f"Error processing frame {frame_number}: {e}")
            return f"Error: Failed to get summary for frame {frame_number}. {e}"

    def summarize_chunks(
        self, video_descriptions: List[str], chunk_size: int = 3
    ) -> List[str]:
        """
        Summarizes chunks of video frame descriptions using the Gemma 3 model.
        """
        logging.info(
            f"Starting summarization of video descriptions in chunks of size {chunk_size}..."
        )
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

            logging.info(
                f"Summarizing chunk {i // chunk_size + 1} (frames {i} to {min(i + chunk_size, num_descriptions) - 1})..."
            )

            try:
                generated_summary = self._generate_text(prompt)
                chunk_summaries.append(generated_summary)
                logging.info(f"Chunk {i // chunk_size + 1} summarized.")

            except Exception as e:
                logging.error(f"Error summarizing chunk starting at index {i}: {e}")
                chunk_summaries.append(f"Error: Failed to summarize chunk. {e}")

        logging.info(f"Finished summarizing {len(chunk_summaries)} chunks.")
        return chunk_summaries
