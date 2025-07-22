import logging
import os
from typing import Optional

import google.generativeai.types as types

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiClient:
    def __init__(
        self,
        credentials_path: str = "GeminiAPI/gcp_credentials.json",
        project_id: str = "devtest-autopilot",
        model_name: str = "gemini-1.5-pro",
    ):
        self.credentials_path = credentials_path
        self.project_id = project_id
        self.model_name = model_name

        self._setup_credentials()
        self._init_google_ai_client()

    def _setup_credentials(self):
        if not os.path.exists(self.credentials_path):
            raise FileNotFoundError(
                f"Credentials file not found: {self.credentials_path}"
            )

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(
            self.credentials_path
        )
        os.environ["GOOGLE_CLOUD_PROJECT"] = self.project_id

        logger.info(f"Credentials set up from: {self.credentials_path}")
        logger.info(f"Project ID: {self.project_id}")

    def _init_google_ai_client(self):
        try:
            import google.generativeai as genai

            genai.configure(api_key=None)

            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Google AI client initialized with model: {self.model_name}")

        except ImportError:
            raise ImportError(
                "Google AI Python SDK not installed. "
                "Install with: pip install google-generativeai"
            )

    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 40,
    ) -> str:
        try:
            return self._generate_text_google_ai(
                prompt, max_tokens, temperature, top_p, top_k
            )

        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise

    def generate_text_with_image(
        self,
        prompt: str,
        image,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 40,
    ) -> str:
        try:
            return self._generate_text_with_image_google_ai(
                prompt, image, max_tokens, temperature, top_p, top_k
            )

        except Exception as e:
            logger.error(f"Error generating text with image: {e}")
            raise

    def _generate_text_google_ai(
        self,
        prompt: str,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        safety_settings = [
            {
                "category": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if not response.text:
                if hasattr(response, "candidates") and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 2:  # BLOCKED
                        logger.warning("Content was blocked by safety filters")
                        return "Content blocked by safety filters"
                    elif finish_reason == 3:  # STOPPED
                        logger.warning("Generation stopped unexpectedly")
                        return "Generation stopped unexpectedly"

                logger.error("No text content in response")
                return "No response generated"

            return response.text

        except Exception as e:
            logger.error(f"Full error details: {e}")
            if "finish_reason" in str(e) and "2" in str(e):
                logger.warning("Content was blocked by safety filters")
                return "Content blocked by safety filters"
            elif "finish_reason" in str(e) and "3" in str(e):
                logger.warning("Generation stopped unexpectedly")
                return "Generation stopped unexpectedly"
            else:
                raise

    def _generate_text_with_image_google_ai(
        self,
        prompt: str,
        image,
        max_tokens: Optional[int],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens

        safety_settings = [
            {
                "category": types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
            {
                "category": types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": types.HarmBlockThreshold.BLOCK_NONE,
            },
        ]

        try:
            response = self.model.generate_content(
                [prompt, image],
                generation_config=generation_config,
                safety_settings=safety_settings,
            )

            if not response.text:
                if hasattr(response, "candidates") and response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 2:  # BLOCKED
                        logger.warning("Content was blocked by safety filters")
                        return "Content blocked by safety filters"
                    elif finish_reason == 3:  # STOPPED
                        logger.warning("Generation stopped unexpectedly")
                        return "Generation stopped unexpectedly"

                logger.error("No text content in response")
                return "No response generated"

            return response.text

        except Exception as e:
            logger.error(f"Full error details: {e}")
            if "finish_reason" in str(e) and "2" in str(e):
                logger.warning("Content was blocked by safety filters")
                return "Content blocked by safety filters"
            elif "finish_reason" in str(e) and "3" in str(e):
                logger.warning("Generation stopped unexpectedly")
                return "Generation stopped unexpectedly"
            else:
                raise


if __name__ == "__main__":
    print("=== Gemini Client Example ===")
    try:
        client = GeminiClient()
        response = client.generate_text("Explain quantum computing in simple terms.")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
